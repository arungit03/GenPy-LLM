from model.config import ModelConfig

config = ModelConfig(
    vocab_size=32000,
    hidden_size=256,
    num_layers=4,
    num_heads=8,
    intermediate_size=1024,
    max_seq_len=512,
    tie_embeddings=True
)
