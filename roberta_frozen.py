"""Frozen RoBERTa representations, with a sentiment head fitted on the fixed train set."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from train_roberta import load_split

ROOT = Path(__file__).resolve().parent
BASE = "FacebookAI/roberta-base"
LABELS = ["negative", "neutral", "positive"]
MAX_LENGTH = 64


class RobertaEncoder:
    def __init__(self):
        torch.set_num_threads(min(6, os.cpu_count() or 4))
        result_path = ROOT / "artifacts/roberta_frozen/metrics.json"
        revision = json.loads(result_path.read_text(encoding="utf-8"))["revision"] if result_path.exists() else "main"
        self.tokenizer = AutoTokenizer.from_pretrained(BASE, revision=revision, cache_dir=ROOT.parent / ".hf-cache/hub")
        self.model = AutoModel.from_pretrained(BASE, use_safetensors=True, revision=revision, cache_dir=ROOT.parent / ".hf-cache/hub", add_pooling_layer=False).eval()
        self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

    def encode(self, texts, batch_size=24, progress=False):
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = self.tokenizer(texts[start:start + batch_size], padding=True,
                                         truncation=True, max_length=MAX_LENGTH,
                                         return_tensors="pt")
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                chunks.append(pooled.cpu().numpy())
                if progress and start % (batch_size * 20) == 0:
                    print(f"Encoded {start}/{len(texts)}", flush=True)
        return np.concatenate(chunks)


def run():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    train, valid = load_split()
    out = ROOT / "artifacts/roberta_frozen"
    out.mkdir(parents=True, exist_ok=True)
    encoder = RobertaEncoder()
    config = dict(model=BASE, revision=encoder.model.config._commit_hash,
                  pooling="attention-mask mean", quantization="dynamic int8 linear layers", max_length=MAX_LENGTH,
                  train_ids=train.textID.tolist(), validation_ids=valid.textID.tolist())
    identity = json.dumps(config, sort_keys=True)
    for name, frame in [("train", train), ("validation", valid)]:
        feature_path = out / f"{name}_features.npy"
        meta_path = out / f"{name}_identity.json"
        if not (feature_path.exists() and meta_path.exists() and meta_path.read_text() == identity):
            features = encoder.encode(frame.text.tolist(), progress=True)
            np.save(feature_path, features)
            meta_path.write_text(identity)
    x = np.load(out / "train_features.npy")
    xv = np.load(out / "validation_features.npy")
    classifier = LogisticRegression(C=1, max_iter=2000, random_state=42)
    classifier.fit(x, train.sentiment)
    probs = classifier.predict_proba(xv)
    valid["roberta_prediction"] = classifier.classes_[probs.argmax(1)]
    valid["roberta_confidence"] = probs.max(1)
    for i, label in enumerate(classifier.classes_):
        valid["roberta_" + label] = probs[:, i]
    metric = dict(model="RoBERTa frozen + LR", accuracy=accuracy_score(valid.sentiment, valid.roberta_prediction),
                  macro_f1=f1_score(valid.sentiment, valid.roberta_prediction, labels=LABELS, average="macro"),
                  confusion=confusion_matrix(valid.sentiment, valid.roberta_prediction, labels=LABELS).tolist(),
                  report=classification_report(valid.sentiment, valid.roberta_prediction, output_dict=True))
    config.update(train_rows=len(train), validation_rows=len(valid), experiment="Frozen pretrained encoder; fitted logistic regression head",
                  head_C=1, seed=42, metric=metric)
    config.pop("train_ids")
    config.pop("validation_ids")
    joblib.dump(classifier, out / "head.joblib")
    valid.to_csv(out / "validation_predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(metric), flush=True)


if __name__ == "__main__":
    run()
