import torch
import torch.nn.functional as F

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Apply top-k and top-p (nucleus) filtering to logits."""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Zero out all logits below the k-th largest
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above top_p (shift so first token above threshold is kept)
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        logits.scatter_(-1, sorted_indices, sorted_indices_to_remove.float() * float('-inf'))
        
    return logits

def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list, penalty: float) -> torch.Tensor:
    """Penalize previously generated tokens."""
    for token_id in set(generated_ids):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits
