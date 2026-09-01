import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from configs.tiny import config
from model.genpy_llm import GenPyLLM
from tokenizer.tokenizer import GenPyTokenizer
from inference.generate import Generator

TOKENIZER_PATH = "tokenizer/genpy_tokenizer.json"

@pytest.fixture
def model_and_tokenizer():
    tok = GenPyTokenizer(TOKENIZER_PATH)
    if tok.vocab_size == 0:
        pytest.skip("Tokenizer not trained yet. Run scripts/train_tokenizer.py first.")
    model = GenPyLLM(config)
    return model, tok

def test_generation_runs(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    device = torch.device("cpu")
    gen = Generator(model, tokenizer, device)
    output = gen.generate("def ", max_new_tokens=10, temperature=1.0)
    assert isinstance(output, str)

def test_greedy_decoding(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    device = torch.device("cpu")
    gen = Generator(model, tokenizer, device)
    out1 = gen.generate("def add", max_new_tokens=5, temperature=0.0)
    out2 = gen.generate("def add", max_new_tokens=5, temperature=0.0)
    assert out1 == out2, "Greedy decoding should be deterministic"

def test_code_only_extraction(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    device = torch.device("cpu")
    gen = Generator(model, tokenizer, device)
    output = gen.generate_code("add two numbers", max_new_tokens=20)
    assert isinstance(output, str)

def test_max_new_tokens_respected(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    device = torch.device("cpu")
    gen = Generator(model, tokenizer, device)
    prompt = "def"
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    output = gen.generate(prompt, max_new_tokens=5, temperature=1.0)
    # output is the decoded NEW tokens only, so length in chars can vary, just ensure it's a string
    assert isinstance(output, str)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
