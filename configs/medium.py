from model.config import ModelConfig

config = ModelConfig(
    vocab_size=32000,
    hidden_size=512,
    num_layers=8,
    num_heads=8,
    intermediate_size=2048,
    max_seq_len=1024,
    tie_embeddings=True
)
