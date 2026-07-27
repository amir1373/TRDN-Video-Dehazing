from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def make_fake_revide_root(
    root: Path, sequence_names: List[str], num_frames: int = 6, size: int = 16
) -> Path:
    """Build a tiny on-disk REVIDE-style tree with real, distinguishable images.

    Each sequence gets hazy/ and gt/ subfolders. Hazy frames are filled with a
    value that is unambiguously different from the corresponding clean/gt
    frame, so tests can assert on pixel content (not just paths) to prove no
    ground-truth leakage.
    """
    root.mkdir(parents=True, exist_ok=True)
    for seq_name in sequence_names:
        hazy_dir = root / seq_name / "hazy"
        gt_dir = root / seq_name / "gt"
        hazy_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in range(num_frames):
            # Hazy frames: low-ish, hazy-looking constant-ish value with mild
            # per-frame variation. Clean frames: a very different value range.
            hazy_val = 60 + (frame_idx * 5) % 40
            clean_val = 200 + (frame_idx * 3) % 40
            hazy_array = np.full((size, size, 3), hazy_val, dtype=np.uint8)
            clean_array = np.full((size, size, 3), clean_val, dtype=np.uint8)
            Image.fromarray(hazy_array).save(hazy_dir / f"{frame_idx:04d}.png")
            Image.fromarray(clean_array).save(gt_dir / f"{frame_idx:04d}.png")
    return root
