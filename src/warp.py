import torch
import torch.nn.functional as F


def warp_with_flow(source: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border") -> torch.Tensor:
    """Backward-warp source using a pixel flow field.

    Shapes:
        source: [B, C, H, W]
        flow: [B, 2, H, W], x/y pixel displacement
    """
    batch, _, height, width = source.shape
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
