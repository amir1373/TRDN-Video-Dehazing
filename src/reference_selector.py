import torch
import torch.nn as nn

from .assertions import assert_reference_weights, assert_temporal_memory, assert_warped_references


class ReferenceSelectionModule(nn.Module):
    """Learn per-pixel reliability weights over warped reference frames."""

    def __init__(self, num_references: int = 9, memory_dim: int = 64, feature_dim: int = 64):
        super().__init__()
        self.num_references = num_references
        self.ref_encoder = nn.Sequential(
            nn.Conv2d(3, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(),
        )
        self.memory_proj = nn.Sequential(nn.Conv2d(memory_dim, feature_dim, 1), nn.SiLU())
        self.score = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feature_dim, 1, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(),
        )

    def forward(self, warped_refs: torch.Tensor, temporal_memory: torch.Tensor, prior_logits: torch.Tensor | None = None) -> dict:
        assert_warped_references(warped_refs, seq_len=self.num_references + 1)
        assert_temporal_memory(temporal_memory, batch=warped_refs.shape[0])
        batch, num_refs, channels, height, width = warped_refs.shape
        if num_refs != self.num_references:
            raise ValueError(f"Expected {self.num_references} references, got {num_refs}")
        ref_feats = self.ref_encoder(warped_refs.reshape(batch * num_refs, channels, height, width))
        ref_feats = ref_feats.reshape(batch, num_refs, -1, height, width)
        memory = self.memory_proj(temporal_memory).unsqueeze(1).expand(-1, num_refs, -1, -1, -1)
        logits = self.score(torch.cat([ref_feats, memory], dim=2).reshape(batch * num_refs, -1, height, width))
        logits = logits.reshape(batch, num_refs, height, width)
        if prior_logits is not None:
            if tuple(prior_logits.shape) != tuple(logits.shape):
                raise ValueError(f"prior_logits must match reference logits {tuple(logits.shape)}, got {tuple(prior_logits.shape)}")
            logits = logits + prior_logits
        weights = torch.softmax(logits, dim=1)
        assert_reference_weights(weights, seq_len=self.num_references + 1)
        weighted_reference = (weights.unsqueeze(2) * warped_refs).sum(dim=1)
        reference_feature = self.out_proj((weights.unsqueeze(2) * ref_feats).sum(dim=1))
        return {
            "weights": weights,
            "weighted_reference": weighted_reference,
            "reference_feature": reference_feature,
            "logits": logits,
        }
