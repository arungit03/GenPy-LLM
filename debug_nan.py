import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.tokenizer import GenPyTokenizer
from configs.tiny import config
from model.genpy_llm import GenPyLLM
from training.instruction_dataset import InstructionDataset

tokenizer = GenPyTokenizer("tokenizer/genpy_tokenizer.json")
import tempfile
import json
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    example = {"instruction": "Return exactly 42", "response": "return 42"}
    f.write(json.dumps(example) + "\n")
    dummy_file = f.name

ds = InstructionDataset(dummy_file, tokenizer, max_seq_len=16)
item = ds[0]

print("input_ids:", item["input_ids"])
print("labels:", item["labels"])

model = GenPyLLM(config)
out = model(item["input_ids"].unsqueeze(0), labels=item["labels"].unsqueeze(0))
print("loss:", out["loss"].item())
