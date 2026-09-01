import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import torch

from configs.tiny import config
from model.genpy_llm import GenPyLLM
from training.checkpoint import save_checkpoint, load_checkpoint
from training.optimizer import create_optimizer
from training.scheduler import get_cosine_schedule_with_warmup

def test_checkpoint_save_and_load():
    model = GenPyLLM(config)
    optimizer = create_optimizer(model)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_checkpoint.pt")
        save_checkpoint(model, optimizer, scheduler, step=42, filepath=ckpt_path)
        
        assert os.path.exists(ckpt_path)
        
        # Load into a new model
        model2 = GenPyLLM(config)
        optimizer2 = create_optimizer(model2)
        scheduler2 = get_cosine_schedule_with_warmup(optimizer2, num_warmup_steps=10, num_training_steps=100)
        
        step = load_checkpoint(ckpt_path, model2, optimizer2, scheduler2)
        assert step == 42
        
        # Verify weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1.data, p2.data)

def test_checkpoint_missing():
    model = GenPyLLM(config)
    step = load_checkpoint("nonexistent.pt", model)
    assert step == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
