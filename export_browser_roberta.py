"""Export the fitted TweetLens RoBERTa classifier for browser execution."""
import tempfile
import hashlib
import json
from pathlib import Path

import onnx
import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--temp-directory',type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
temp_directory=args.temp_directory or Path(tempfile.gettempdir())
temp_directory.mkdir(exist_ok=True)
tempfile.tempdir=str(temp_directory)

ROOT = Path(__file__).resolve().parent
OUT = args.output or ROOT / 'assets/roberta'
CACHE = ROOT.parent / '.hf-cache/hub'
OUT.mkdir(parents=True, exist_ok=True)
metadata = json.loads((ROOT/'artifacts/roberta_frozen/metrics.json').read_text(encoding='utf-8'))
tokenizer = AutoTokenizer.from_pretrained(metadata['model'], revision=metadata['revision'], cache_dir=CACHE, local_files_only=True)
encoder = AutoModel.from_pretrained(metadata['model'], revision=metadata['revision'], cache_dir=CACHE, local_files_only=True, add_pooling_layer=False, attn_implementation='eager').eval()
head = joblib.load(ROOT/'artifacts/roberta_frozen/head.joblib')

class BrowserClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(768, 3)
        self.head.weight.data.copy_(torch.tensor(head.coef_, dtype=torch.float32))
        self.head.bias.data.copy_(torch.tensor(head.intercept_, dtype=torch.float32))
    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        return torch.softmax(self.head(pooled), dim=-1)

torch.set_num_threads(6)
model=BrowserClassifier().eval()
sample=tokenizer(['I prefer to work alone.'],return_tensors='pt')
float_path=ROOT/'artifacts/roberta_frozen/browser_float.onnx'
quant_path=ROOT/'artifacts/roberta_frozen/browser_int8.onnx'
if not float_path.exists(): torch.onnx.export(model,(sample['input_ids'],sample['attention_mask']),str(float_path),input_names=['input_ids','attention_mask'],output_names=['probabilities'],dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},'probabilities':{0:'batch'}},opset_version=17,dynamo=False,external_data=False)
print('Float model exported',flush=True)
quantize_dynamic(onnx.load(str(float_path)),str(quant_path),weight_type=QuantType.QUInt8,op_types_to_quantize=['MatMul','Gemm','Gather'],per_channel=False,reduce_range=False)
print('Quantized',quant_path.stat().st_size,flush=True)
tokenizer.save_pretrained(OUT)
chunks=[]
with quant_path.open('rb') as source:
    index=0
    while data:=source.read(16*1024*1024):
        name=f'model-{index:02d}.bin'
        (OUT/name).write_bytes(data)
        chunks.append({'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        index+=1
(OUT/'manifest.json').write_text(json.dumps({'revision':metadata['revision'],'classes':head.classes_.tolist(),'max_length':64,'chunks':chunks,'bytes':quant_path.stat().st_size,'sha256':hashlib.sha256(quant_path.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
print('Browser assets ready',len(chunks),flush=True)
