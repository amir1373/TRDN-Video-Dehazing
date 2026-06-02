import torch
import torch.nn as nn
import torch.nn.functional as F

from .assertions import assert_frames, assert_reference_weights, assert_temporal_memory


class TemporalRetrievalTransformer(nn.Module):
    """Temporal reasoning over aligned video references.

    The transformer does not directly dehaze. It learns temporal reliability
    context and returns an enhanced memory map plus reference prior logits.
    """

    def __init__(
        self,
        input_channels: int = 3,
        memory_dim: int = 64,
        token_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        pool_size: int = 8,
        max_seq_len: int = 10,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.max_seq_len = max_seq_len
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, memory_dim, 3, padding=1),
            nn.GroupNorm(8, memory_dim),
            nn.SiLU(),
            nn.Conv2d(memory_dim, memory_dim, 3, padding=1),
            nn.GroupNorm(8, memory_dim),
            nn.SiLU(),
        )
        self.frame_to_token = nn.Linear(memory_dim, token_dim)
        self.memory_to_token = nn.Linear(memory_dim, token_dim)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, max_seq_len, 1, token_dim))
        self.spatial_embedding = nn.Parameter(torch.zeros(1, 1, pool_size * pool_size, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(token_dim)
        self.memory_decoder = nn.Sequential(
            nn.Conv2d(token_dim, memory_dim, 1),
            nn.GroupNorm(8, memory_dim),
            nn.SiLU(),
            nn.Conv2d(memory_dim, memory_dim, 3, padding=1),
        )
        self.reference_prior = nn.Linear(token_dim, 1)
        nn.init.normal_(self.temporal_embedding, std=0.02)
        nn.init.normal_(self.spatial_embedding, std=0.02)

    def forward(self, aligned_frames: torch.Tensor, temporal_memory: torch.Tensor) -> dict:
        assert_frames(aligned_frames, name="aligned_frames")
        assert_temporal_memory(temporal_memory, batch=aligned_frames.shape[0])
        batch, seq_len, _, height, width = aligned_frames.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"max_seq_len={self.max_seq_len}, got sequence length {seq_len}")

        frame_features = self.frame_encoder(aligned_frames.reshape(batch * seq_len, 3, height, width))
        pooled = F.adaptive_avg_pool2d(frame_features, (self.pool_size, self.pool_size))
        pooled = pooled.reshape(batch, seq_len, -1, self.pool_size * self.pool_size).permute(0, 1, 3, 2)
        frame_tokens = self.frame_to_token(pooled)

        memory_tokens = F.adaptive_avg_pool2d(temporal_memory, (self.pool_size, self.pool_size))
        memory_tokens = self.memory_to_token(memory_tokens.flatten(2).transpose(1, 2)).unsqueeze(1)

        tokens = frame_tokens + memory_tokens + self.temporal_embedding[:, :seq_len] + self.spatial_embedding
        encoded = self.encoder(tokens.reshape(batch, seq_len * self.pool_size * self.pool_size, -1))
        encoded = self.out_norm(encoded).reshape(batch, seq_len, self.pool_size * self.pool_size, -1)

        current = encoded[:, -1].transpose(1, 2).reshape(batch, -1, self.pool_size, self.pool_size)
        enhanced = self.memory_decoder(current)
        enhanced = F.interpolate(enhanced, size=(height, width), mode="bilinear", align_corners=False)
        enhanced_memory = temporal_memory + enhanced

        prior = self.reference_prior(encoded[:, :-1]).squeeze(-1)
        prior = prior.reshape(batch, seq_len - 1, self.pool_size, self.pool_size)
        prior = F.interpolate(prior, size=(height, width), mode="bilinear", align_corners=False)
        assert_reference_weights(torch.softmax(prior, dim=1), seq_len=seq_len, name="transformer_reference_prior")
        return {
            "enhanced_memory": enhanced_memory,
            "reference_prior_logits": prior,
            "tokens": encoded.reshape(batch, seq_len * self.pool_size * self.pool_size, -1),
        }
