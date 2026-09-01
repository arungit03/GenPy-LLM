import sys
import os
import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.tiny import config
from model.genpy_llm import GenPyLLM

def test_causal_masking():
    """Verify that future tokens cannot influence past tokens."""
    torch.manual_seed(42)
    model = GenPyLLM(config)
    model.eval()
    
    # Input sequences
    seq_A = torch.tensor([[10, 20, 30, 40, 50]])  # Shape (1, 5)
    seq_B = torch.tensor([[10, 20, 30, 99, 99]])  # First 3 tokens identical
    
    with torch.no_grad():
        out_A = model(seq_A)["logits"]
        out_B = model(seq_B)["logits"]
        
    # The logits for the first 3 tokens (indices 0, 1, 2) must be exactly identical
    # despite the sequences diverging at index 3.
    assert torch.allclose(out_A[0, :3, :], out_B[0, :3, :], atol=1e-6), \
        "Information leakage detected! Future tokens altered past token logits."

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
