import sys
import os
import json
import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.tokenizer import GenPyTokenizer
from configs.tiny import config
from model.genpy_llm import GenPyLLM
from training.instruction_dataset import InstructionDataset
from training.optimizer import create_optimizer
from training.trainer import Trainer

TOKENIZER_PATH = "tokenizer/genpy_tokenizer.json"

@pytest.fixture
def tokenizer():
    tok = GenPyTokenizer(TOKENIZER_PATH)
    if tok.vocab_size == 0:
        pytest.skip("Tokenizer not trained yet.")
    return tok

def test_tiny_overfit(tokenizer, tmp_path):
    # 1. Create a tiny dataset of 10 identical examples
    dummy_file = tmp_path / "overfit.jsonl"
    example = {"instruction": "Return exactly 42", "response": "return 42"}
    with open(dummy_file, "w") as f:
        for _ in range(10):
            f.write(json.dumps(example) + "\n")
            
    # 2. Setup tiny model and trainer
    device = torch.device("cpu")
    model = GenPyLLM(config).to(device)
    
    # Check initial loss
    ds = InstructionDataset(str(dummy_file), tokenizer, max_seq_len=64)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=2)
    
    # Run a single batch to get initial loss
    model.eval()
    batch = next(iter(dataloader))
    with torch.no_grad():
        initial_loss = model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"].item()
        
    # 3. Train for 20 epochs to overfit
    optimizer = create_optimizer(model, learning_rate=1e-3)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataloader=dataloader,
        device=device,
        checkpoint_dir=str(tmp_path),
        save_every_steps=100
    )
    
    trainer.train(num_epochs=20)
    
    # 4. Check final loss
    model.eval()
    with torch.no_grad():
        final_loss = model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"].item()
        
    # Loss should decrease substantially
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    assert final_loss < 1.0, f"Failed to overfit on tiny dataset: final_loss={final_loss:.4f}"
    
    # 5. Generate and check if it compiled
    from inference.generate import Generator
    gen = Generator(model, tokenizer, device)
    output = gen.generate_code("Return exactly 42", max_new_tokens=10, temperature=0.0)
    
    # Check compilation
    try:
        compile(output, "<string>", "exec")
    except Exception as e:
        pytest.fail(f"Generated output failed to compile: {e}\nOutput was: {output}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
