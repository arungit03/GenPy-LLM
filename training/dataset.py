import json
import torch
from torch.utils.data import Dataset, DataLoader

class CodeDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []
        
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        self.examples.append(text)
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {len(self.examples)} examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            
        input_ids = list(tokens)
        labels = list(tokens)
        
        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len) # -100 is standard PyTorch ignore_index
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def create_dataloader(data_path, tokenizer, max_seq_len, batch_size, shuffle=True, num_workers=0):
    dataset = CodeDataset(data_path, tokenizer, max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
