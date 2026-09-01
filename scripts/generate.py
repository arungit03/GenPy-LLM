import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizer.tokenizer import GenPyTokenizer
from model.genpy_llm import GenPyLLM
from inference.generate import Generator

def main():
    parser = argparse.ArgumentParser(description="Generate Python code with GenPy-LLM")
    parser.add_argument("--prompt", type=str, required=True, help="Natural language prompt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/genpy_tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--code-only", action="store_true", help="Output code only (no conversational text)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (temperature=0)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = GenPyTokenizer(args.tokenizer)
    if tokenizer.vocab_size == 0:
        print("ERROR: Tokenizer not found. Run scripts/train_tokenizer.py first.")
        sys.exit(1)
        
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
        
    model = GenPyLLM.from_checkpoint(args.checkpoint).to(device)
    
    generator = Generator(model=model, tokenizer=tokenizer, device=device)
    
    temperature = 0.0 if args.greedy else args.temperature
    
    output = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        code_only=args.code_only,
    )
    
    print(output)

if __name__ == "__main__":
    main()
