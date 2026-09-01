import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import json
import torch

from configs.tiny import config
from tokenizer.tokenizer import GenPyTokenizer
from training.dataset import CodeDataset

TOKENIZER_PATH = "tokenizer/genpy_tokenizer.json"

@pytest.fixture
def tokenizer():
    tok = GenPyTokenizer(TOKENIZER_PATH)
    if tok.vocab_size == 0:
        pytest.skip("Tokenizer not trained yet. Run scripts/train_tokenizer.py first.")
    return tok

@pytest.fixture
def sample_jsonl():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for text in [
            "def foo():\n    return 1",
            "x = 1 + 2\nprint(x)",
            "class Bar:\n    pass",
        ]:
            f.write(json.dumps({"text": text}) + "\n")
        return f.name

def test_dataset_loads(tokenizer, sample_jsonl):
    ds = CodeDataset(sample_jsonl, tokenizer, max_seq_len=64)
    assert len(ds) == 3

def test_dataset_item_shapes(tokenizer, sample_jsonl):
    ds = CodeDataset(sample_jsonl, tokenizer, max_seq_len=64)
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["input_ids"].shape == (64,)
    assert item["labels"].shape == (64,)

def test_next_token_shift(tokenizer, sample_jsonl):
    ds = CodeDataset(sample_jsonl, tokenizer, max_seq_len=128)
    item = ds[0]
    input_ids = item["input_ids"]
    labels = item["labels"]
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
    # Dataset should output input_ids == labels (un-shifted)
    valid_len = (labels != -100).sum().item()
    for i in range(valid_len):
        assert input_ids[i] == labels[i]

def test_padding(tokenizer, sample_jsonl):
    max_seq = 64
    ds = CodeDataset(sample_jsonl, tokenizer, max_seq_len=max_seq)
    for i in range(len(ds)):
        item = ds[i]
        assert len(item["input_ids"]) == max_seq
        assert len(item["labels"]) == max_seq

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
