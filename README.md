# GenPy-LLM

A **300M parameter decoder-only Transformer language model** built **from scratch** using PyTorch, specialized exclusively for **Python code generation**.

> **Natural-language programming request → correct Python code**

```
Input:  "Write Python code to check if a number is prime."

Output: def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
```

---

## Architecture

GenPy-LLM is a modern, efficient **decoder-only Transformer** featuring:

| Component | Implementation |
|-----------|---------------|
| Positional embeddings | **RoPE** (Rotary Position Embeddings) |
| Normalization | **RMSNorm** (pre-normalization) |
| Feed-forward | **SwiGLU** |
| Attention | Multi-head causal self-attention with `scaled_dot_product_attention` |
| Training | AdamW + cosine LR schedule with warmup |
| Precision | BF16/FP16 mixed precision |
| Weight tying | Token embedding ↔ LM head (optional) |

### Model Sizes

| Config | Layers | Hidden | Heads | FFN | Parameters |
|--------|--------|--------|-------|-----|------------|
| Tiny   | 4      | 256    | 8     | 1024 | ~12M      |
| Medium | 8      | 512    | 8     | 2048 | ~60M      |
| Large  | 20     | 1024   | 16    | 2730 | ~300M     |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Download a Python code dataset (requires `datasets` library):

```bash
pip install datasets
python scripts/prepare_data.py --output data/raw/python_code.jsonl --num-examples 50000
```

Or supply your own JSONL file with `{"text": "...python code..."}` format.

### 3. Train tokenizer

```bash
python scripts/train_tokenizer.py --data data/raw/python_code.jsonl --vocab-size 32000
```

### 4. Pretrain (Tiny — for local testing)

```bash
python scripts/pretrain.py --config configs/tiny.py --data data/raw/python_code.jsonl --epochs 3
```

### 5. Fine-tune on instructions

Prepare an instruction JSONL file with `{"instruction": "...", "response": "..."}` entries, then:

```bash
python scripts/finetune.py --config configs/tiny.py --data data/instructions.jsonl
```

### 6. Generate code

```bash
python scripts/generate.py \
    --prompt "Write Python code to check if a number is odd or even." \
    --max-new-tokens 200 \
    --code-only
```

### 7. Python API

```python
from genpy_llm import GenPyLLM

model = GenPyLLM.from_checkpoint("checkpoints/latest.pt")
code = model.generate_code("Write Python code to find the largest number in a list.")
print(code)
```

---

## Dataset Format

### Pretraining

JSONL file, one example per line:

```json
{"text": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)"}
```

### Instruction Fine-tuning

JSONL file:

```json
{"instruction": "Write a Python function to add two numbers.", "response": "def add(a, b):\n    return a + b"}
```

---

## Kaggle Setup (Training the 300M model)

### Recommended Kaggle settings

```
GPU: Tesla T4 or P100
GPU Memory: 16 GB
```

### Recommended training configuration

```python
# configs/large.py
config = ModelConfig(
    vocab_size=32000,
    hidden_size=1024,
    num_layers=20,
    num_heads=16,
    intermediate_size=2730,
    max_seq_len=2048,
    tie_embeddings=True
)
```

### Recommended Kaggle training args

```bash
python scripts/pretrain.py \
    --config configs/large.py \
    --data data/raw/python_code.jsonl \
    --batch-size 4 \
    --epochs 3 \
    --save-every 500
```

### Checkpoint Resume

Training is interrupted when Kaggle sessions end. Resume using:

```bash
python scripts/pretrain.py --config configs/large.py --resume
```

Checkpoints are saved as:
```
checkpoints/
    checkpoint_step_500.pt
    checkpoint_step_1000.pt
    latest.pt
```

---

## Evaluation

### Syntax check

```bash
python -c "
from evaluation.syntax_check import check_syntax
ok, err = check_syntax('def add(a, b):\n    return a + b')
print('Valid:', ok)
"
```

### Full benchmark (100 prompts)

```bash
python evaluation/benchmark.py --checkpoint checkpoints/latest.pt
```

Output:
```
BENCHMARK RESULTS
================
Total prompts:       100
Syntax accuracy:     XX/100 (XX%)
Functional accuracy: XX/100 (XX%)
Avg generation len:  XXX chars
```

---

## Training Curriculum

```
Phase 1: Python source-code pretraining (large corpus)
    ↓
Phase 2: Instruction fine-tuning (instruction/response pairs)
    ↓
Phase 3: Evaluation (benchmark)
    ↓
Phase 4: Iterate based on failures
```

---

## Running Tests

```bash
pytest
```

All 25 unit tests should pass before beginning a real training run.

---

## Limitations

- GenPy-LLM generates **Python only** — it is not a general-purpose LLM.
- Untrained models produce random-looking output (expected behavior).
- A proper training run on CodeParrot or The Stack requires Kaggle GPU.
- Evaluation uses a timeout subprocess — not a Docker sandbox.
- The tiny model (12M params) will not produce high-quality code without training on a large dataset.

---

## Project Structure

```
genpy-llm/
├── configs/           # tiny / medium / large ModelConfig
├── data/              # raw and processed dataset files
├── tokenizer/         # BPE tokenizer training and wrapper
├── model/             # Transformer components (RMSNorm, RoPE, Attention, SwiGLU, etc.)
├── training/          # Dataset, Trainer, Optimizer, Scheduler, Checkpoint
├── inference/         # Autoregressive Generator, sampling utilities
├── evaluation/        # Syntax checker, execution tests, 100-prompt benchmark
├── scripts/           # CLI entrypoints for all stages
└── tests/             # pytest unit tests for every component
```
