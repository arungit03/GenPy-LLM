import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from configs.tiny import config
from model.normalization import RMSNorm
from model.embeddings import RotaryEmbedding, apply_rotary_pos_emb
from model.attention import CausalSelfAttention
from model.mlp import SwiGLUMLP
from model.transformer_block import TransformerBlock
from model.transformer import Transformer
from model.genpy_llm import GenPyLLM, count_parameters

BATCH = 2
SEQ = 16

def test_rmsnorm():
    norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    x = torch.randn(BATCH, SEQ, config.hidden_size)
    out = norm(x)
    assert out.shape == x.shape, "RMSNorm output shape mismatch"
    assert not torch.isnan(out).any(), "RMSNorm output contains NaN"

def test_rope():
    rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
    x = torch.randn(BATCH, config.num_heads, SEQ, config.head_dim)
    cos, sin = rope(x, seq_len=SEQ)
    
    q = torch.randn_like(x)
    k = torch.randn_like(x)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape, "RoPE query shape mismatch"
    assert k_rot.shape == k.shape, "RoPE key shape mismatch"
    assert not torch.isnan(q_rot).any(), "RoPE output contains NaN"

def test_causal_attention():
    rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
    dummy_input = torch.randn(BATCH, SEQ, config.hidden_size)
    cos, sin = rope(dummy_input, seq_len=SEQ)
    
    attn = CausalSelfAttention(config)
    position_ids = torch.arange(SEQ)
    out = attn(dummy_input, position_ids=position_ids, cos=cos, sin=sin)
    assert out.shape == dummy_input.shape, "Attention output shape mismatch"

def test_swiglu():
    mlp = SwiGLUMLP(config)
    x = torch.randn(BATCH, SEQ, config.hidden_size)
    out = mlp(x)
    assert out.shape == x.shape, "SwiGLU output shape mismatch"
    assert not torch.isnan(out).any(), "SwiGLU output contains NaN"

def test_transformer_block():
    rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
    block = TransformerBlock(config)
    x = torch.randn(BATCH, SEQ, config.hidden_size)
    cos, sin = rope(x, seq_len=SEQ)
    position_ids = torch.arange(SEQ)
    
    out = block(x, position_ids=position_ids, cos=cos, sin=sin)
    assert out.shape == x.shape, "TransformerBlock output shape mismatch"

def test_full_model_forward():
    model = GenPyLLM(config)
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ))
    labels = torch.randint(0, config.vocab_size, (BATCH, SEQ))
    
    output = model(input_ids, labels=labels)
    assert "logits" in output
    assert "loss" in output
    assert output["logits"].shape == (BATCH, SEQ, config.vocab_size)
    assert output["loss"].item() > 0

def test_parameter_count():
    model = GenPyLLM(config)
    total, trainable = count_parameters(model)
    assert total > 0
    assert trainable > 0
    assert total == trainable  # all params trainable for tiny model

def test_weight_tying():
    model = GenPyLLM(config)
    if config.tie_embeddings:
        assert model.lm_head.weight.data_ptr() == model.transformer.tok_embeddings.weight.data_ptr(), \
            "Weight tying not working"

def test_no_nan_in_forward():
    model = GenPyLLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    output = model(input_ids)
    assert not torch.isnan(output["logits"]).any(), "Logits contain NaN"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
