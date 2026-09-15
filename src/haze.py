import random

import torch
import torch.nn.functional as F


def gaussian_blur_tensor(x: torch.Tensor, kernel_size: int = 9, sigma: float = 3.0) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(x, kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels)


def make_dust_overlay(batch: int, height: int, width: int, device: torch.device, density: float = 0.015) -> torch.Tensor:
    dust = torch.zeros(batch, 1, height, width, device=device)
    count = max(1, int(height * width * density))
    for bidx in range(batch):
        ys = torch.randint(0, height, (count,), device=device)
        xs = torch.randint(0, width, (count,), device=device)
        dust[bidx, 0, ys, xs] = torch.rand(count, device=device) * 0.8 + 0.2
    dust = F.avg_pool2d(dust, kernel_size=7, stride=1, padding=3)
    dust = F.avg_pool2d(dust, kernel_size=7, stride=1, padding=3)
    return (dust / (dust.amax(dim=(2, 3), keepdim=True) + 1e-8)).clamp(0, 1)


def simulate_realistic_haze(clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Create synthetic haze for BCHW or CHW clean images in [0, 1]."""
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    clean = clean.float().clamp(0, 1)
    mask = mask.float().to(clean.device).clamp(0, 1)
    batch, _, height, width = clean.shape

    atmospheric_light = torch.empty(batch, 3, 1, 1, device=clean.device).uniform_(0.72, 1.0)
    haze_strength = torch.empty(batch, 1, 1, 1, device=clean.device).uniform_(0.35, 0.85)
    transmission = (1.0 - haze_strength * mask).clamp(0.08, 1.0)
    hazy = clean * transmission + atmospheric_light * (1.0 - transmission)

    blurred = gaussian_blur_tensor(
        hazy,
        kernel_size=random.choice([7, 9, 11]),
        sigma=random.uniform(1.5, 4.0),
    )
    hazy = hazy * (1.0 - 0.45 * mask) + blurred * (0.45 * mask)
    dust = make_dust_overlay(batch, height, width, clean.device, density=random.uniform(0.004, 0.02))
    dust_color = torch.empty(batch, 3, 1, 1, device=clean.device).uniform_(0.75, 1.0)
    return (hazy + dust_color * dust * mask * random.uniform(0.03, 0.12)).clamp(0, 1)
