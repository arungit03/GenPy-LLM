import sys
import os
import argparse
import json
import hashlib
from datasets import load_dataset
from tqdm import tqdm

def get_hash(text):
    """Compute MD5 hash for exact duplicate detection."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def download_codeparrot_sample(output_dir, num_examples=50000, val_split=0.05, max_len=100000):
    """Download a sample of CodeParrot Python dataset with deduplication and validation split."""
    print(f"Downloading CodeParrot sample ({num_examples} examples)...")
    print("Source: https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid")
    print("License: Please verify the license of individual repositories within CodeParrot.")
    
    dataset = load_dataset(
        "codeparrot/codeparrot-clean-valid",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    
    seen_hashes = set()
    train_count = 0
    val_count = 0
    
    num_val = int(num_examples * val_split)
    num_train = num_examples - num_val
    
    with open(train_path, 'w', encoding='utf-8') as f_train, open(val_path, 'w', encoding='utf-8') as f_val:
        for item in tqdm(dataset, desc="Processing"):
            if train_count >= num_train and val_count >= num_val:
                break
                
            content = item.get("content", "").strip()
            
            # 1. Length filtering (drop empty and extremely long)
            if not content or len(content) < 50 or len(content) > max_len:
                continue
                
            # 2. Quality filter (must look like Python)
            if "def " not in content and "class " not in content and "import " not in content:
                continue
                
            # 3. Exact Duplicate Detection
            h = get_hash(content)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            
            # Prepare metadata to preserve provenance
            repo = item.get("repo_name", "unknown")
            path = item.get("path", "unknown")
            license_info = item.get("license", "unknown")
            
            entry = {
                "text": content,
                "meta": {
                    "repo": repo,
                    "path": path,
                    "license": license_info,
                    "source": "codeparrot-clean-valid"
                }
            }
            
            # 4. Train / Val split
            # Fill val first to ensure we get exactly num_val if possible
            if val_count < num_val:
                f_val.write(json.dumps(entry, ensure_ascii=False) + "\n")
                val_count += 1
            else:
                f_train.write(json.dumps(entry, ensure_ascii=False) + "\n")
                train_count += 1
                
    print(f"Saved {train_count} train examples to {train_path}")
    print(f"Saved {val_count} validation examples to {val_path}")
    print(f"Removed duplicates: {len(seen_hashes)} unique files kept out of larger streamed set.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--num-examples", type=int, default=50000)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--source", type=str, default="codeparrot", choices=["codeparrot"])
    args = parser.parse_args()
    
    if args.source == "codeparrot":
        try:
            download_codeparrot_sample(args.output_dir, args.num_examples, args.val_split)
        except ImportError:
            print("ERROR: 'datasets' library not installed. Run: pip install datasets")
            print("       Then re-run this script.")
            sys.exit(1)
    
if __name__ == "__main__":
    main()
