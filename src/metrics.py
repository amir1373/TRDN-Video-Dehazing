import math

import numpy as np
import torch
import torch.nn.functional as F


def psnr_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.float().clamp(0, 1), target.float().clamp(0, 1)).item()
    return 99.0 if mse <= 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def _to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    x = x.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255.0).round().astype(np.uint8)


def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(_to_uint8_hwc(pred), _to_uint8_hwc(target), channel_axis=2, data_range=255))
    except Exception:
        x = pred.detach().float().clamp(0, 1)
        y = target.detach().float().clamp(0, 1)
        c1, c2 = 0.01**2, 0.03**2
        mux, muy = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = ((x - mux) * (y - muy)).mean()
        return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2)))
