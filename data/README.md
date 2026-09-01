## Data Directory

This directory contains training data for GenPy-LLM.

### Structure

```
data/
├── raw/              # Raw JSONL files (Python source code)
│   └── sample.jsonl  # Small dummy dataset for quick tests
└── processed/        # Tokenized / preprocessed data (optional)
```

### Data Format

Each line in the JSONL file must contain a `text` key:

```json
{"text": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)"}
```

### Downloading Real Data

Use the provided script:

```bash
pip install datasets
python scripts/prepare_data.py --output data/raw/python_code.jsonl --num-examples 50000
```

This downloads from [CodeParrot](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid) on Hugging Face.
