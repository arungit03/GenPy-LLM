import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.train_tokenizer import train_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/sample.jsonl", help="Path to training data")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--save-path", type=str, default="tokenizer/genpy_tokenizer.json", help="Path to save tokenizer")
    
    args = parser.parse_args()
    
    # Create dummy data if it doesn't exist for initial testing
    if not os.path.exists(args.data):
        print(f"Dummy data file created at {args.data} for testing.")
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        with open(args.data, 'w') as f:
            f.write('{"text": "def factorial(n):\\n    if n == 0:\\n        return 1\\n    return n * factorial(n - 1)"}\\n')
            f.write('{"text": "class MyClass:\\n    def __init__(self):\\n        self.value = 42"}\\n')
            f.write('{"text": "for i in range(10):\\n    print(i)"}\\n')
            f.write('{"text": "import os\\nimport sys"}\\n')
            
    train_tokenizer(args.data, args.vocab_size, args.save_path)
