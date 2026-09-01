import torch
import torch.nn.functional as F

from inference.sampling import top_k_top_p_filter, apply_repetition_penalty

class Generator:
    """Autoregressive text generator for GenPy-LLM."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        stop_tokens: list = None,
        code_only: bool = False,
    ) -> str:
        """Generate text autoregressively from a prompt."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated_ids = list(input_ids)
        
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_token_id]
            
        max_seq = self.model.config.max_seq_len
        
        for _ in range(max_new_tokens):
            # Truncate context if it exceeds max_seq_len
            ctx = input_tensor[:, -max_seq:]
            
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                outputs = self.model(input_ids=ctx)
                
            logits = outputs["logits"][:, -1, :]  # (1, vocab_size)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
                
            if temperature == 0.0:
                # Greedy decoding
                next_token_id = torch.argmax(logits, dim=-1).item()
            else:
                logits = logits.float() / temperature  # cast to float32 before sampling
                logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
                logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                probs = F.softmax(logits, dim=-1)
                # Final safety guard
                probs = probs.clamp(min=0)
                if probs.sum() == 0:
                    probs = torch.ones_like(probs) / probs.shape[-1]
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
            generated_ids.append(next_token_id)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=self.device)], dim=1)
            
            if next_token_id in stop_tokens:
                break
                
        # Decode only the generated part (not the prompt)
        new_ids = generated_ids[len(input_ids):]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        
        if code_only:
            text = _extract_code_only(text)
            
        return text

    def generate_code(
        self,
        instruction: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate Python code from a natural language instruction using the SFT prompt template."""
        prompt = f"### Instruction:\n{instruction}\n\n### Python Code:\n"
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            code_only=True,
        )


def _extract_code_only(text: str) -> str:
    """Strip conversational wrapper text, returning only the code block."""
    # If a markdown code block is present, extract its content
    import re
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # Remove common conversational prefixes
    for prefix in ["Sure! Here is the code:", "Here is the Python code:", "Sure,"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            
    return text.strip()
