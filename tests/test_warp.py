import torch
import pytest

from src.flow import assert_raft_flow_sane, compute_raft_flow
from src.warp import warp_with_flow


def test_warp_with_flow_zero_flow_is_identity():
    source = torch.rand(1, 3, 10, 10)
    flow = torch.zeros(1, 2, 10, 10)
    warped = warp_with_flow(source, flow)
    assert torch.allclose(warped, source, atol=1e-5)


def test_warp_with_flow_supports_two_channel_tensors():
    # Needed by the eval script's forward-backward consistency check, which
    # warps a 2-channel flow field (not just 3-channel RGB predictions).
    flow_field = torch.randn(1, 2, 12, 12)
    warp_by = torch.zeros(1, 2, 12, 12)
    warped = warp_with_flow(flow_field, warp_by)
    assert warped.shape == (1, 2, 12, 12)
    assert torch.allclose(warped, flow_field, atol=1e-5)


def test_raft_flow_sanity_rejects_nonfinite_values():
    flow = torch.zeros(1, 2, 8, 8)
    flow[0, 0, 0, 0] = torch.nan

    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        assert_raft_flow_sane(flow, 8, 8)


def test_raft_flow_sanity_rejects_implausible_magnitude():
    flow = torch.full((1, 2, 8, 8), 100.0)

    with pytest.raises(FloatingPointError, match="implausible"):
        assert_raft_flow_sane(flow, 8, 8, max_flow_factor=2.0)


def test_raft_flow_sanity_can_be_disabled():
    class ImplausibleRaft(torch.nn.Module):
        _trdn_validate_flow = False

        def forward(self, source, _target, num_flow_updates):
            return [source.new_full((source.shape[0], 2, source.shape[2], source.shape[3]), 100.0)]

    frame = torch.zeros(1, 3, 8, 8)

    flow = compute_raft_flow(ImplausibleRaft(), frame, frame)

    assert flow.shape == (1, 2, 8, 8)
