import sys
import os
import json
import argparse
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizer.tokenizer import GenPyTokenizer
from model.genpy_llm import GenPyLLM
from inference.generate import Generator
from evaluation.syntax_check import check_syntax
from evaluation.execution_test import run_code_in_subprocess

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts.json")

def run_benchmark(generator, prompts, max_new_tokens=256):
    results = {
        'total': len(prompts),
        'syntax_pass': 0,
        'functional_pass': 0,
        'by_category': defaultdict(lambda: {'total': 0, 'syntax': 0, 'functional': 0}),
        'details': []
    }
    
    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        tests = item.get("tests", "")
        category = item.get("category", "Unknown")
        
        print(f"[{i+1}/{len(prompts)}] {prompt[:60]}...")
        
        code = generator.generate_code(prompt, max_new_tokens=max_new_tokens)
        
        syntax_ok, syntax_err = check_syntax(code)
        functional_ok = False
        func_err = ""
        
        if syntax_ok and tests:
            functional_ok, func_err = run_code_in_subprocess(code, tests)
            
        results['by_category'][category]['total'] += 1
        if syntax_ok:
            results['syntax_pass'] += 1
            results['by_category'][category]['syntax'] += 1
        if functional_ok:
            results['functional_pass'] += 1
            results['by_category'][category]['functional'] += 1
            
        results['details'].append({
            'prompt': prompt,
            'code': code,
            'category': category,
            'syntax_ok': syntax_ok,
            'functional_ok': functional_ok,
            'syntax_error': syntax_err,
            'functional_error': func_err,
        })
        
        status = "PASS" if functional_ok else ("SYNTAX_OK" if syntax_ok else "FAIL")
        print(f"  [{status}] {len(code)} chars")
        
    total = results['total']
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total prompts:       {total}")
    print(f"Syntax accuracy:     {results['syntax_pass']}/{total} ({100*results['syntax_pass']/max(total,1):.1f}%)")
    print(f"Functional accuracy: {results['functional_pass']}/{total} ({100*results['functional_pass']/max(total,1):.1f}%)")
    
    avg_len = sum(len(d['code']) for d in results['details']) / max(total, 1)
    print(f"Avg generation len:  {avg_len:.0f} chars")
    
    print("\nBy Category:")
    for cat, counts in sorted(results['by_category'].items()):
        t = counts['total']
        s = counts['syntax']
        f = counts['functional']
        print(f"  {cat:25s}: Syntax {s}/{t}  Functional {f}/{t}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/genpy_tokenizer.json")
    parser.add_argument("--prompts", type=str, default=PROMPTS_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = GenPyTokenizer(args.tokenizer)
    model = GenPyLLM.from_checkpoint(args.checkpoint).to(device)
    generator = Generator(model=model, tokenizer=tokenizer, device=device)
    
    with open(args.prompts, 'r') as f:
        prompts = json.load(f)
        
    results = run_benchmark(generator, prompts, args.max_new_tokens)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
