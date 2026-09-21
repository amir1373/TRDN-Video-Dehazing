"""Variable-size occlusion masks with explicit coverage control.

The stock generators in masks.py have hard-coded size ranges, which measure
(empirically, 256x256, 40 samples each):
    rectangle 15.1% mean / 36.4% max      ellipse 16.5% / 31.7%
    blob      15.9% / 25.2%               perlin  18.2% / 24.3%
so "how much is occluded" is an uncontrolled side effect of shape parameters.

These generators instead SAMPLE A TARGET COVERAGE first, then build a shape that
achieves it. That gives genuinely variable occlusion size across a chosen range
(e.g. 5%-85%) and makes coverage a reportable experimental variable rather than
an accident of the shape code.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np
import torch


def _as_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask).unsqueeze(0).float().clamp(0, 1)


def variable_rectangle(h: int, w: int, target: float) -> np.ndarray:
    """Rectangle whose AREA is `target` of the frame, with random aspect ratio."""
    area = target * h * w
    aspect = random.uniform(0.45, 2.2)                 # w/h
    rh = min(h, max(1, int(round(math.sqrt(area / aspect)))))
    rw = min(w, max(1, int(round(area / rh))))
    top = random.randint(0, max(0, h - rh))
    left = random.randint(0, max(0, w - rw))
    m = np.zeros((h, w), dtype=np.float32)
    m[top:top + rh, left:left + rw] = 1.0
    return m


def variable_ellipse(h: int, w: int, target: float) -> np.ndarray:
    """Ellipse with area = target*h*w  (pi*a*b = area)."""
    area = target * h * w
    aspect = random.uniform(0.55, 1.8)
    # cv2.ellipse takes SEMI-axes, so solve area = pi * sa * sb directly for them.
    # (Computing full axes and halving them quarters the area - that bug made
    #  ellipse top out at ~20% coverage when 85% was requested.)
    sb = math.sqrt(area / (math.pi * aspect))
    sa = aspect * sb
    sa = int(max(2, min(w * 0.75, sa)))
    sb = int(max(2, min(h * 0.75, sb)))
    cx = random.randint(0, w - 1)
    cy = random.randint(0, h - 1)
    m = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(m, (cx, cy), (sa, sb), random.uniform(0, 180), 0, 360, 1.0, -1)
    return m


def variable_blob(h: int, w: int, target: float) -> np.ndarray:
    """Accumulate overlapping circles until measured coverage reaches target."""
    m = np.zeros((h, w), dtype=np.float32)
    base = max(4, int(min(h, w) * random.uniform(0.06, 0.16)))
    for _ in range(400):
        if m.mean() >= target:
            break
        cv2.circle(m, (random.randint(0, w - 1), random.randint(0, h - 1)),
                   random.randint(base, int(base * 1.9)), 1.0, -1)
    return m


def variable_perlin(h: int, w: int, target: float) -> np.ndarray:
    """Smooth noise field thresholded (bisection) to hit the target coverage."""
    grid = random.choice([4, 6, 8, 12])
    noise = np.random.rand(grid, grid).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = (noise - noise.min()) / max(1e-6, noise.max() - noise.min())
    lo, hi = 0.0, 1.0
    for _ in range(24):                       # bisect the threshold
        mid = (lo + hi) / 2
        if (noise >= mid).mean() > target:
            lo = mid
        else:
            hi = mid
    return (noise >= (lo + hi) / 2).astype(np.float32)


_SHAPES = {"rectangle": variable_rectangle, "ellipse": variable_ellipse,
           "blob": variable_blob, "perlin": variable_perlin}


def variable_occlusion_mask(height: int, width: int, mode: str = "mixed",
                            coverage_min: float = 0.05,
                            coverage_max: float = 0.85) -> torch.Tensor:
    """Mask of shape [1,H,W]; occluded fraction ~ U(coverage_min, coverage_max).

    Sampling coverage uniformly means the model sees small patches AND
    near-total occlusion within one run, so it cannot specialise to one size.
    """
    target = random.uniform(coverage_min, coverage_max)
    shape = random.choice(list(_SHAPES)) if mode in ("mixed", "auto") else mode
    if shape not in _SHAPES:
        raise ValueError(f"Unknown variable mask mode: {mode}")
    return _as_tensor(_SHAPES[shape](height, width, target))
