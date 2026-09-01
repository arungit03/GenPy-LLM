import json
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os

def train_tokenizer(data_path, vocab_size, save_path):
    print(f"Training tokenizer on {data_path} with vocab size {vocab_size}")
    
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|unk|>", "<|pad|>"],
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet()
    )

    def batch_iterator(batch_size=1000):
        with open(data_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        batch.append(text)
                except:
                    pass
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")


