import torch

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
