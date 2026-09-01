from model.config import ModelConfig

config = ModelConfig(
    vocab_size=32000,
    hidden_size=1024,
    num_layers=20,
    num_heads=16,
    intermediate_size=3072,
    max_seq_len=2048,
    tie_embeddings=True
)
