import sys
import os
import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.instruction_dataset import InstructionDataset
from tokenizer.tokenizer import GenPyTokenizer

TOKENIZER_PATH = "tokenizer/genpy_tokenizer.json"

@pytest.fixture
def tokenizer():
    tok = GenPyTokenizer(TOKENIZER_PATH)
    if tok.vocab_size == 0:
        pytest.skip("Tokenizer not trained yet.")
    return tok

def test_sft_masking(tokenizer, tmp_path):
    dummy_file = tmp_path / "dummy_sft.jsonl"
    dummy_file.write_text('{"instruction": "Add 2+2", "response": "return 4"}\n')
    
    ds = InstructionDataset(str(dummy_file), tokenizer, max_seq_len=64)
    item = ds[0]
    
    labels = item["labels"]
    
    # We expect the instruction part to be masked out with -100
    assert labels[0] == -100, "Instruction prefix not masked."
    
    # We expect the response part to NOT be -100 (except padding at the end)
    # Let's count how many valid tokens exist
    valid_tokens = (labels != -100).sum().item()
    assert valid_tokens > 0, "No valid tokens found to train on!"
    
    # Verify the padding at the end is also -100
    assert labels[-1] == -100, "Padding tokens not masked out."

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
