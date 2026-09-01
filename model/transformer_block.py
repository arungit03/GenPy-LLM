import torch
import torch.nn as nn

from model.config import ModelConfig
from model.attention import CausalSelfAttention
from model.mlp import SwiGLUMLP
from model.normalization import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        # Pre-LN + Attention + Residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        
        # Pre-LN + MLP + Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
