import torch
import torch.nn.functional as F
import torch.nn as nn
import pytest

from src.losses import LossBundle, weighted_total_loss
from src.warp import warp_with_flow


class _Config:
    w_diffusion = 1.0
    w_l1 = 0.25
    w_lpips = 0.05
    w_temporal = 0.05
    w_flow = 0.05
    w_reference = 0.0


class _DifferenceLPIPS(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - target).mean(dim=(1, 2, 3), keepdim=True)


class _ZeroLPIPS(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pred.new_zeros((pred.shape[0], 1, 1, 1))


def test_lpips_startup_probe_is_nonzero_for_different_tensors():
    bundle = LossBundle.__new__(LossBundle)
    nn.Module.__init__(bundle)
    bundle.lpips_model = _DifferenceLPIPS()

    assert bundle._assert_lpips_nonzero("cpu") > 0.0


def test_lpips_startup_probe_rejects_silent_zero():
    bundle = LossBundle.__new__(LossBundle)
    nn.Module.__init__(bundle)
    bundle.lpips_model = _ZeroLPIPS()

    with pytest.raises(RuntimeError, match="returned zero"):
        bundle._assert_lpips_nonzero("cpu")


def test_legacy_temporal_consistency_loss_matches_old_formula():
    pred = torch.rand(2, 3, 8, 8)
    warped_refs = torch.rand(2, 3, 3, 8, 8)
    weights = torch.softmax(torch.rand(2, 3, 8, 8), dim=1)
    expected_weighted_ref = (weights.unsqueeze(2) * warped_refs).sum(dim=1)
    expected = F.l1_loss(pred, expected_weighted_ref.detach())
    actual = LossBundle.legacy_temporal_consistency_loss(pred, warped_refs, weights)
    assert torch.allclose(actual, expected)


def test_predictive_temporal_consistency_loss_uses_warp_with_flow():
    pred_current = torch.rand(2, 3, 8, 8)
    pred_previous = torch.rand(2, 3, 8, 8)
    flow = torch.randn(2, 2, 8, 8) * 0.5
    expected = F.l1_loss(pred_current, warp_with_flow(pred_previous, flow).detach())
    actual = LossBundle.predictive_temporal_consistency_loss(pred_current, pred_previous, flow)
    assert torch.allclose(actual, expected)


def test_predictive_temporal_consistency_loss_zero_flow_identity_prediction():
    # With zero flow (no motion) and prev == current prediction, the loss must be 0.
    pred = torch.rand(1, 3, 6, 6)
    flow = torch.zeros(1, 2, 6, 6)
    loss = LossBundle.predictive_temporal_consistency_loss(pred, pred, flow)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_reference_preservation_loss_is_noop_under_full_mask():
    pred = torch.rand(1, 3, 6, 6)
    weighted_reference = torch.rand(1, 3, 6, 6)
    full_mask = torch.ones(1, 1, 6, 6)
    loss = LossBundle.reference_preservation_loss(pred, weighted_reference, full_mask)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_weighted_total_loss_no_keyerror_when_reference_absent():
    parts = {
        "diffusion": torch.tensor(1.0),
        "l1": torch.tensor(1.0),
        "lpips": torch.tensor(1.0),
        "temporal": torch.tensor(1.0),
        "flow": torch.tensor(1.0),
    }
    total = weighted_total_loss(_Config(), parts)  # must not KeyError on "reference"
    expected = _Config.w_diffusion + _Config.w_l1 + _Config.w_lpips + _Config.w_temporal + _Config.w_flow
    assert torch.allclose(total, torch.tensor(expected))


def test_weighted_total_loss_includes_reference_when_present():
    parts = {
        "diffusion": torch.tensor(0.0),
        "l1": torch.tensor(0.0),
        "lpips": torch.tensor(0.0),
        "temporal": torch.tensor(0.0),
        "flow": torch.tensor(0.0),
        "reference": torch.tensor(1.0),
    }
    config = _Config()
    config.w_reference = 0.5
    total = weighted_total_loss(config, parts)
    assert torch.allclose(total, torch.tensor(0.5))
