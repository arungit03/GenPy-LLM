import json
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    """Dataset for supervised fine-tuning using instruction/response pairs."""
    
    PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Python Code:\n{response}"
    
    def __init__(self, data_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []
        
        print(f"Loading instruction dataset from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "instruction" in data and "response" in data:
                        self.examples.append(data)
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {len(self.examples)} instruction examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        text = self.PROMPT_TEMPLATE.format(
            instruction=item["instruction"],
            response=item["response"]
        )
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            
        input_ids = list(tokens)
        labels = list(tokens)
        
        # Compute where the response starts so we can mask the prompt
        prompt = self.PROMPT_TEMPLATE.format(
            instruction=item["instruction"],
            response=""
        )
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        
        # Mask out instruction part from loss (set to -100)
        labels_masked = [-100] * min(prompt_len, len(labels)) + labels[prompt_len:]
        
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            labels_masked.extend([-100] * pad_len)
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels_masked, dtype=torch.long),
        }
