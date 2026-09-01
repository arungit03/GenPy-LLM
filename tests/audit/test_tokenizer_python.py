import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.tokenizer import GenPyTokenizer

TOKENIZER_PATH = "tokenizer/genpy_tokenizer.json"

@pytest.fixture
def tokenizer():
    tok = GenPyTokenizer(TOKENIZER_PATH)
    if tok.vocab_size == 0:
        pytest.skip("Tokenizer not trained yet.")
    return tok

def test_python_round_trip(tokenizer):
    python_code = '''def factorial(n):
    # Base case
    if n == 0:
        return 1
    \t
    return n * factorial(n - 1)
'''
    encoded = tokenizer.encode(python_code, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    
    # Needs to match perfectly
    assert decoded == python_code, "Tokenizer corrupted Python syntax (indentation/newlines)."

def test_unknown_tokens(tokenizer):
    # Test unicode / weird chars to see fallback behavior
    weird_text = "def ƒunc(): 🚀"
    encoded = tokenizer.encode(weird_text)
    decoded = tokenizer.decode(encoded)
    
    # The standard behavior for BPE usually falls back to UNK or byte fallback
    assert len(encoded) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
