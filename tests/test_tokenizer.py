import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from evaluation.syntax_check import check_syntax, syntax_pass_rate

def test_valid_function():
    code = "def add(a, b):\n    return a + b"
    ok, err = check_syntax(code)
    assert ok is True
    assert err == ""

def test_invalid_syntax():
    code = "def broken(:\n    pass"
    ok, err = check_syntax(code)
    assert ok is False
    assert err != ""

def test_empty_string():
    code = ""
    ok, err = check_syntax(code)
    assert ok is True  # empty is valid Python

def test_class_valid():
    code = "class Foo:\n    def __init__(self):\n        self.x = 1"
    ok, err = check_syntax(code)
    assert ok is True

def test_pass_rate():
    codes = [
        "def f(): pass",
        "def broken(:",
        "x = 1 + 2",
    ]
    rate = syntax_pass_rate(codes)
    assert abs(rate - 2/3) < 0.01

def test_import_statement():
    code = "import os\nimport sys\nfrom pathlib import Path"
    ok, err = check_syntax(code)
    assert ok is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
