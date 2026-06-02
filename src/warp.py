import torch
import torch.nn.functional as F

from .assertions import assert_flow, assert_image


def warp_with_flow(source: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border") -> torch.Tensor:
    """Backward-warp source using a pixel flow field.

    Shapes:
        source: [B, C, H, W]
        flow: [B, 2, H, W], x/y pixel displacement
    """
    assert_image(source, channels=3, name="source")
    assert_flow(flow, name="flow")
    batch, _, height, width = source.shape
    if tuple(flow.shape) != (batch, 2, height, width):
        raise ValueError(f"flow shape {tuple(flow.shape)} must match source [B,2,H,W] with B/H/W={batch}/{height}/{width}")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
    sample = base - flow.to(dtype=source.dtype)
    grid_x = 2.0 * sample[:, 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * sample[:, 1] / max(height - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(source, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
