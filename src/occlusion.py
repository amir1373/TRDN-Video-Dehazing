"""Lens-fixed opaque occluders for the partial-occlusion experiment.

An occluder models an opaque obstruction attached to the camera (dust, mud or debris on the
lens): it covers the same image region in every frame of a window, while the camera moves
past the scene behind it. Occluded pixels are replaced by mid-grey (0.5 in [0, 1], i.e. 0 in
the [-1, 1] range Stable Diffusion inpainting uses for masked pixels), and the mask marks them
with 1.

Training samples a coverage uniformly from a range; evaluation fixes the coverage and derives
the shape deterministically from the window identity, so every model is scored on identical
occluders.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

from .masks_variable import _SHAPES, _as_tensor

# Rectangles are excluded: a lens obstruction is irregular, not axis-aligned.
OCCLUDER_SHAPES = ("ellipse", "blob", "perlin")
OCCLUDER_FILL = 0.5
MAX_DRAWS = 30


def occluder_mask(height: int, width: int, coverage: float, shape: str | None = None) -> torch.Tensor:
    """[1, H, W] opaque-occluder mask with the requested fraction of pixels covered."""
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie in (0, 1); got {coverage}")
    shape = shape or random.choice(OCCLUDER_SHAPES)
    if shape not in OCCLUDER_SHAPES:
        raise ValueError(f"Unknown occluder shape: {shape}")
    # Shapes placed near the border are clipped and cover less than requested, so redraw
    # until the measured coverage is within 10 % (relative) of the target, keeping the closest.
    best, best_gap = None, float("inf")
    for _ in range(MAX_DRAWS):
        candidate = _SHAPES[shape](height, width, coverage)
        gap = abs(float(candidate.mean()) - coverage)
        if gap < best_gap:
            best, best_gap = candidate, gap
        if gap <= 0.10 * coverage:
            break
    return _as_tensor(best)


def seeded_occluder_mask(height: int, width: int, coverage: float, key: str) -> torch.Tensor:
    """Deterministic occluder for evaluation: the same key and coverage give the same mask."""
    seed = int.from_bytes(hashlib.sha256(f"{key}|{coverage:.4f}".encode()).digest()[:8], "little")
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        return occluder_mask(height, width, coverage)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def apply_occluder(frames: torch.Tensor, mask: torch.Tensor, fill: float = OCCLUDER_FILL) -> torch.Tensor:
    """Replace occluded pixels of [..., 3, H, W] frames with `fill`; mask is [1, H, W]."""
    return frames * (1.0 - mask) + fill * mask
