"""Learned haze-severity map for the inpainting mask channel (experiment M1).

In dehazing mode TRDN feeds the UNet an all-ones mask. M1 replaces it with a spatial severity
map M(x) in [0, 1] predicted from the hazy current frame by a small CNN. The CNN is supervised
with a pseudo-target derived from the training pair through the atmospheric scattering model,
I = J t + A (1 - t)  =>  t = |I - A| / |J - A|  (channel mean), severity = 1 - t, where the
airlight A is the dark-channel-prior estimate from the hazy frame. Pixels whose clean radiance
is close to A carry no information about t and are excluded from the loss. (A plain |I - J|
target would confuse dark objects with dense haze and bright objects with clear air.)
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def estimate_airlight(hazy: torch.Tensor, patch: int = 15, top_fraction: float = 0.001) -> torch.Tensor:
    """Dark-channel-prior airlight, [B, 3, 1, 1]: mean hazy colour of the brightest dark-channel pixels."""
    dark = -F.max_pool2d(-hazy.min(dim=1, keepdim=True).values, patch, stride=1, padding=patch // 2)
    batch = hazy.shape[0]
    flat_dark = dark.reshape(batch, -1)
    k = max(1, int(flat_dark.shape[1] * top_fraction))
    index = flat_dark.topk(k, dim=1).indices                       # [B, k]
    flat_hazy = hazy.reshape(batch, 3, -1)
    picked = torch.gather(flat_hazy, 2, index.unsqueeze(1).expand(-1, 3, -1))
    return picked.mean(dim=2).view(batch, 3, 1, 1)


@torch.no_grad()
def pseudo_haze_severity(hazy: torch.Tensor, clean: torch.Tensor, smooth: int = 15, min_contrast: float = 0.05):
    """Return (severity [B,1,H,W] in [0,1], valid [B,1,H,W] float) from a paired hazy/clean frame."""
    hazy, clean = hazy.float().clamp(0, 1), clean.float().clamp(0, 1)
    airlight = estimate_airlight(hazy)
    num = (hazy - airlight).abs().mean(dim=1, keepdim=True)
    den = (clean - airlight).abs().mean(dim=1, keepdim=True)
    valid = (den > min_contrast).float()
    transmission = (num / den.clamp(min=min_contrast)).clamp(0, 1)
    # Smooth only over valid pixels (normalised box filter), so excluded pixels do not leak in.
    kernel = torch.ones(1, 1, smooth, smooth, device=hazy.device) / (smooth * smooth)
    pad = smooth // 2
    weighted = F.conv2d(F.pad(transmission * valid, (pad,) * 4, mode="replicate"), kernel)
    weight = F.conv2d(F.pad(valid, (pad,) * 4, mode="replicate"), kernel)
    transmission = torch.where(weight > 1e-3, weighted / weight.clamp(min=1e-3), transmission)
    return 1.0 - transmission, valid


class HazeMapEstimator(nn.Module):
    """Small CNN: hazy frame [B,3,H,W] in [0,1] -> severity map [B,1,H,W] in [0,1] (~40k parameters)."""

    def __init__(self, width: int = 32, initial_severity: float = 0.98) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, width // 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )
        # Start close to the all-ones mask the pretrained model was fine-tuned with.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, float(torch.logit(torch.tensor(initial_severity))))

    def forward(self, hazy: torch.Tensor) -> torch.Tensor:
        logits = self.body(hazy.float())
        return torch.sigmoid(F.interpolate(logits, size=hazy.shape[-2:], mode="bilinear", align_corners=False))
