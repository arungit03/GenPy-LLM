from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_seq_len: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    dropout: float = 0.1

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads
