import sys
import os
import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.tiny import config
from model.genpy_llm import GenPyLLM
from training.checkpoint import save_checkpoint, load_checkpoint
from training.optimizer import create_optimizer

def test_reproducibility(tmp_path):
    torch.manual_seed(42)
    model = GenPyLLM(config)
    optimizer = create_optimizer(model)
    
    # Do a dummy forward/backward to populate optimizer state
    dummy_input = torch.randint(0, config.vocab_size, (1, 8))
    loss = model(dummy_input, labels=dummy_input)["loss"]
    loss.backward()
    optimizer.step()
    
    ckpt_path = str(tmp_path / "ckpt.pt")
    
    # Save checkpoint
    save_checkpoint(model, optimizer, None, step=10, filepath=ckpt_path)
    
    # Load into new model
    model2 = GenPyLLM(config)
    opt2 = create_optimizer(model2)
    step = load_checkpoint(ckpt_path, model2, opt2, None)
    
    assert step == 10
    
    # Check model weights exactly match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1.data, p2.data), "Model weights did not restore perfectly."
        
    # Check optimizer state (e.g. momentum buffers)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        if p1 in optimizer.state and p2 in opt2.state:
            state1 = optimizer.state[p1]
            state2 = opt2.state[p2]
            for k in state1:
                if isinstance(state1[k], torch.Tensor):
                    assert torch.allclose(state1[k], state2[k]), f"Optimizer state mismatch for {k}"
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
