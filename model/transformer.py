import torch
import torch.nn as nn

from model.config import ModelConfig
from model.transformer_block import TransformerBlock
from model.normalization import RMSNorm
from model.embeddings import RotaryEmbedding

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_seq_len,
            base=config.rope_theta,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        bsz, seq_len = input_ids.shape
        
        hidden_states = self.tok_embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
            )
            
        hidden_states = self.norm(hidden_states)
        return hidden_states
