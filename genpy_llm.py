"""
GenPy-LLM Public API
~~~~~~~~~~~~~~~~~~~~
Simple top-level interface for loading and generating code.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from model.genpy_llm import GenPyLLM
from tokenizer.tokenizer import GenPyTokenizer
from inference.generate import Generator


class GenPyLLMApi:
    """High-level public API for GenPy-LLM code generation."""
    
    def __init__(self, model: GenPyLLM, tokenizer: GenPyTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._generator = Generator(model=model, tokenizer=tokenizer, device=device)
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, tokenizer_path: str = "tokenizer/genpy_tokenizer.json"):
        """Load GenPy-LLM from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            tokenizer_path: Path to the tokenizer JSON file.
            
        Returns:
            Initialized GenPyLLMApi instance.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
            
        model = GenPyLLM.from_checkpoint(checkpoint_path).to(device)
        tokenizer = GenPyTokenizer(tokenizer_path)
        
        return cls(model=model, tokenizer=tokenizer, device=device)
        
    def generate_code(
        self,
        instruction: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate Python code from a natural language instruction.
        
        Args:
            instruction: Natural language description of what the code should do.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy, higher = more random).
            top_k: Top-k sampling cutoff.
            top_p: Top-p (nucleus) sampling cutoff.
            repetition_penalty: Penalty for repeating tokens.
            
        Returns:
            Generated Python code as a string.
            
        Example:
            >>> model = GenPyLLMApi.from_checkpoint("checkpoints/genpy-llm-300m.pt")
            >>> code = model.generate_code("Write Python code to find the largest number in a list.")
            >>> print(code)
        """
        return self._generator.generate_code(
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

# Allow: from genpy_llm import GenPyLLM
GenPyLLM = GenPyLLMApi
