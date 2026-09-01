import sys
import os
import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import CodeDataset
from tokenizer.tokenizer import GenPyTokenizer

TOKENIZER_PATH = "tokenizer/genpy_tokenizer.json"

@pytest.fixture
def tokenizer():
    tok = GenPyTokenizer(TOKENIZER_PATH)
    if tok.vocab_size == 0:
        pytest.skip("Tokenizer not trained yet.")
    return tok

def test_objective_shifting(tokenizer, tmp_path):
    dummy_file = tmp_path / "dummy.jsonl"
    dummy_file.write_text('{"text": "A B C D E"}\n')
    
    ds = CodeDataset(str(dummy_file), tokenizer, max_seq_len=8)
    item = ds[0]
    
    input_ids = item["input_ids"]
    labels = item["labels"]
    
    # Since CodeDataset no longer shifts, input_ids and labels should be identical
    # The model shifts them internally!
    valid_len = (labels != tokenizer.pad_token_id).sum().item()
    for i in range(valid_len):
        assert input_ids[i] == labels[i], "Labels should not be pre-shifted by the dataset."

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
