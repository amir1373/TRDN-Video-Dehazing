import torch
from src.config import TRDNConfig
from src.train import build_temporal_modules


def test_nondefault_temporal_width_reaches_all_modules():
    cfg = TRDNConfig(temporal_hidden_dim=32, seq_len=3, transformer_token_dim=32, transformer_num_layers=1, transformer_pool_size=2)
    memory, transformer, selector, adapter = build_temporal_modules(cfg, 64, 'cpu')
    frames = torch.rand(1, 3, 3, 16, 16)
    result = transformer(frames, memory(frames))
    features = selector(frames[:, :-1], result['enhanced_memory'])
    tokens = adapter(result['enhanced_memory'], features['reference_feature'])
    assert tokens.shape == (1, 16, 64)
