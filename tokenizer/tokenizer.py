import os
from tokenizers import Tokenizer

class GenPyTokenizer:
    def __init__(self, path=None):
        self.tokenizer = None
        self.pad_token_id = 2  # Based on special_tokens=["<|endoftext|>", "<|unk|>", "<|pad|>"]
        self.eos_token_id = 0
        self.unk_token_id = 1

        if path and os.path.exists(path):
            self.load(path)
            
    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
        
    def save(self, path):
        if self.tokenizer is not None:
            self.tokenizer.save(path)
            
    def encode(self, text, add_special_tokens=True):
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids
        
    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        
    @property
    def vocab_size(self):
        if self.tokenizer is not None:
            return self.tokenizer.get_vocab_size()
        return 0
