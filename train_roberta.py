"""Fine-tune RoBERTa on the frozen TweetLens sentiment split."""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LABELS = ["negative", "neutral", "positive"]


def load_split():
    raw = pd.read_csv(ROOT / "data/train.csv", dtype={"textID": str})
    manifest = pd.read_csv(ROOT / "artifacts/split_manifest.csv", dtype=str)
    rows = manifest.merge(raw, on="textID", how="left", validate="one_to_one")
    if rows[["text", "sentiment"]].isna().any().any():
        raise ValueError("Missing text or label for a split ID")
    if not rows.sentiment.isin(LABELS).all():
        raise ValueError("Unknown sentiment label")
    train = rows[rows.split == "train"].copy()
    valid = rows[rows.split == "validation"].copy()
    if (len(train), len(valid), len(rows)) != (21931, 5481, 27412):
        raise ValueError("Unexpected split sizes")
    def groups(frame):
        return set(frame.text.str.lower().str.split().str.join(" "))
    if groups(train) & groups(valid):
        raise ValueError("Normalized text occurs in both splits")
    previous = pd.read_csv(ROOT / "artifacts/validation_predictions.csv",
                           dtype={"textID": str})
    aligned = valid.merge(previous[["textID", "sentiment", "Logistic regression"]],
                          on="textID", validate="one_to_one", suffixes=("", "_previous"))
    if len(aligned) != len(valid) or not aligned.sentiment.equals(aligned.sentiment_previous):
        raise ValueError("Baseline validation rows or labels differ")
    return train, aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-split", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/roberta")
    args = parser.parse_args()
    train, valid = load_split()
    print(f"Verified {len(train)} train / {len(valid)} validation; no text overlap.")
    if args.check_split:
        return

    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty --output directory for this run")
    set_seed(42)
    base = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base)
    ids = {name: i for i, name in enumerate(LABELS)}
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=3, label2id=ids, id2label=dict(enumerate(LABELS)))
    def encode(frame):
        data = Dataset.from_dict({"text": frame.text.tolist(),
                                  "labels": frame.sentiment.map(ids).tolist()})
        return data.map(lambda batch: tokenizer(batch["text"], truncation=True,
                                                max_length=128),
                        batched=True, remove_columns=["text"])
    settings = TrainingArguments(
        output_dir=str(args.output), learning_rate=2e-5, weight_decay=0.01,
        num_train_epochs=2, per_device_train_batch_size=8,
        gradient_accumulation_steps=2, per_device_eval_batch_size=16,
        save_strategy="no", report_to="none", seed=42)
    trainer = Trainer(model=model, args=settings, train_dataset=encode(train),
                      data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train()
    logits = trainer.predict(encode(valid)).predictions
    valid["RoBERTa"] = [LABELS[i] for i in np.argmax(logits, axis=1)]
    result = {}
    for name in ["Logistic regression", "RoBERTa"]:
        result[name] = dict(accuracy=accuracy_score(valid.sentiment, valid[name]),
                            macro_f1=f1_score(valid.sentiment, valid[name], labels=LABELS, average="macro"),
                            confusion_matrix=confusion_matrix(valid.sentiment, valid[name], labels=LABELS).tolist())
    trainer.save_model(str(args.output / "model"))
    tokenizer.save_pretrained(args.output / "model")
    valid[["textID", "text", "sentiment", "Logistic regression", "RoBERTa"]].to_csv(
        args.output / "validation_predictions.csv", index=False)
    summary = dict(base=base, revision=getattr(model.config, "_commit_hash", None),
                   labels=LABELS, train_rows=len(train), validation_rows=len(valid), scores=result)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
