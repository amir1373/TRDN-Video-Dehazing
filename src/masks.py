import random

import cv2
import numpy as np
import torch


def random_rectangle_mask(height: int, width: int, min_frac: float = 0.15, max_frac: float = 0.65) -> torch.Tensor:
    mask = torch.zeros(1, height, width)
    rect_h = random.randint(max(1, int(height * min_frac)), max(2, int(height * max_frac)))
    rect_w = random.randint(max(1, int(width * min_frac)), max(2, int(width * max_frac)))
    top = random.randint(0, max(0, height - rect_h))
    left = random.randint(0, max(0, width - rect_w))
    mask[:, top : top + rect_h, left : left + rect_w] = 1.0
    return mask


def random_ellipse_mask(height: int, width: int) -> torch.Tensor:
    mask = np.zeros((height, width), dtype=np.float32)
    center = (random.randint(width // 4, 3 * width // 4), random.randint(height // 4, 3 * height // 4))
    axes = (
        random.randint(max(4, width // 10), max(5, width // 3)),
        random.randint(max(4, height // 10), max(5, height // 3)),
    )
    cv2.ellipse(mask, center, axes, random.uniform(0, 180), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return torch.from_numpy(mask).unsqueeze(0).clamp(0, 1)


def random_blob_mask(height: int, width: int, blobs: int | None = None) -> torch.Tensor:
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(blobs or random.randint(4, 12)):
        center = (random.randint(0, width - 1), random.randint(0, height - 1))
        radius = random.randint(max(3, min(height, width) // 20), max(4, min(height, width) // 7))
        cv2.circle(mask, center, radius, random.uniform(0.45, 1.0), -1)
    kernel = random.choice([15, 21, 31, 41])
    mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return torch.from_numpy(mask).unsqueeze(0).clamp(0, 1)


def perlin_like_mask(height: int, width: int, grid: int = 8, threshold: float = 0.45) -> torch.Tensor:
    noise = np.random.rand(max(2, height // grid), max(2, width // grid)).astype(np.float32)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=grid / 2)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    mask = np.clip((noise - threshold) / max(1e-4, 1.0 - threshold), 0, 1)
    return torch.from_numpy(mask).unsqueeze(0).float()


def generate_haze_mask(height: int, width: int, mode: str = "mixed") -> torch.Tensor:
    if mode == "rectangle":
        return random_rectangle_mask(height, width)
    if mode == "ellipse":
        return random_ellipse_mask(height, width)
    if mode == "blob":
        return random_blob_mask(height, width)
    if mode == "perlin":
        return perlin_like_mask(height, width)
    if mode == "mixed":
        return generate_haze_mask(height, width, random.choice(["rectangle", "ellipse", "blob", "perlin"]))
    raise ValueError(f"Unknown mask mode: {mode}")
