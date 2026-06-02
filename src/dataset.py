import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .haze import simulate_realistic_haze
from .masks import generate_haze_mask

CLEAN_DIR_NAMES = {"gt", "GT", "clean", "Clean", "clear", "Clear", "target", "targets", "groundtruth", "ground_truth"}
HAZY_DIR_NAMES = {"hazy", "Hazy", "input", "Input", "inputs", "fog", "Fog", "degraded", "Degraded"}


def image_to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def list_images(folder: Path, extensions: Tuple[str, ...]) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def discover_revide_sequences(root: Path, split: Optional[str], extensions: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Discover common REVIDE clean/hazy sequence layouts."""
    search_roots = []
    if split and (root / split).exists():
        search_roots.append(root / split)
    search_roots.append(root)

    sequences: List[Dict[str, Any]] = []
    seen = set()
    for base in search_roots:
        if not base.exists():
            continue

        for seq_dir in [path for path in base.rglob("*") if path.is_dir()]:
            children = {child.name: child for child in seq_dir.iterdir() if child.is_dir()}
            clean_dirs = [children[name] for name in children if name in CLEAN_DIR_NAMES]
            hazy_dirs = [children[name] for name in children if name in HAZY_DIR_NAMES]
            for clean_dir in clean_dirs:
                for hazy_dir in hazy_dirs:
                    clean_files = list_images(clean_dir, extensions)
                    hazy_files = list_images(hazy_dir, extensions)
                    if clean_files and hazy_files:
                        key = (str(hazy_dir.resolve()), str(clean_dir.resolve()))
                        if key not in seen:
                            seen.add(key)
                            sequences.append(
                                {
                                    "name": seq_dir.name,
                                    "hazy_dir": hazy_dir,
                                    "clean_dir": clean_dir,
                                    "hazy_files": hazy_files,
                                    "clean_files": clean_files,
                                }
                            )

        children = {child.name: child for child in base.iterdir() if child.is_dir()}
        clean_roots = [children[name] for name in children if name in CLEAN_DIR_NAMES]
        hazy_roots = [children[name] for name in children if name in HAZY_DIR_NAMES]
        for clean_root in clean_roots:
            for hazy_root in hazy_roots:
                for hazy_seq in [path for path in hazy_root.iterdir() if path.is_dir()]:
                    clean_seq = clean_root / hazy_seq.name
                    if not clean_seq.exists():
                        continue
                    key = (str(hazy_seq.resolve()), str(clean_seq.resolve()))
                    if key in seen:
                        continue
                    seen.add(key)
                    sequences.append(
                        {
                            "name": hazy_seq.name,
                            "hazy_dir": hazy_seq,
                            "clean_dir": clean_seq,
                            "hazy_files": list_images(hazy_seq, extensions),
                            "clean_files": list_images(clean_seq, extensions),
                        }
                    )
    return sequences


class REVIDESequenceDataset(Dataset):
    """REVIDE sequence dataset returning the canonical TRDN tensors."""

    def __init__(
        self,
        root: str,
        split: Optional[str] = "train",
        seq_len: int = 10,
        crop_size: int = 256,
        random_crop: bool = True,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        synthetic_if_empty: bool = True,
        max_sequences: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.seq_len = seq_len
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.extensions = extensions
        self.synthetic_if_empty = synthetic_if_empty
        self.sequences = discover_revide_sequences(self.root, split, extensions)
        if max_sequences is not None:
            self.sequences = self.sequences[:max_sequences]

        self.index: List[Tuple[int, int]] = []
        for seq_idx, sequence in enumerate(self.sequences):
            count = min(len(sequence["hazy_files"]), len(sequence["clean_files"]))
            for end_idx in range(seq_len - 1, count):
                self.index.append((seq_idx, end_idx))
        self.synthetic_len = 8 if not self.index and synthetic_if_empty else 0

    def __len__(self) -> int:
        return len(self.index) if self.index else self.synthetic_len

    def _load_real_clip(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
        seq_idx, end_idx = self.index[idx]
        sequence = self.sequences[seq_idx]
        start_idx = end_idx - self.seq_len + 1
        hazy_paths = sequence["hazy_files"][start_idx : end_idx + 1]
        frames = torch.stack([image_to_tensor(path) for path in hazy_paths], dim=0)
        target = image_to_tensor(sequence["clean_files"][end_idx])
        return frames, target, [str(path) for path in hazy_paths], sequence["name"]

    def _load_synthetic_clip(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
        height = width = max(self.crop_size, 256)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
        clean_frames = []
        for tidx in range(self.seq_len):
            shift = 0.02 * (idx + tidx)
            clean_frames.append(
                torch.stack(
                    [
                        (xx + shift).fmod(1.0),
                        (yy + 0.5 * shift).fmod(1.0),
                        (0.5 * xx + 0.5 * yy + shift).fmod(1.0),
                    ],
                    dim=0,
                )
            )
        target = clean_frames[-1].clone()
        frames = torch.stack([torch.clamp(frame * 0.65 + 0.35, 0, 1) for frame in clean_frames], dim=0)
        return frames, target, [f"synthetic_{idx}_{tidx}.png" for tidx in range(self.seq_len)], "synthetic"

    def _crop(self, frames: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = frames.shape
        crop = min(self.crop_size, height, width)
        if height == crop and width == crop:
            return frames, target
        top = random.randint(0, height - crop) if self.random_crop else (height - crop) // 2
        left = random.randint(0, width - crop) if self.random_crop else (width - crop) // 2
        return frames[:, :, top : top + crop, left : left + crop], target[:, top : top + crop, left : left + crop]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.index:
            frames, target, paths, name = self._load_real_clip(idx)
        else:
            frames, target, paths, name = self._load_synthetic_clip(idx)
        frames, target = self._crop(frames, target)
        _, height, width = target.shape
        mask = generate_haze_mask(height, width, mode="mixed").float()
        corrupted = simulate_realistic_haze(target.unsqueeze(0), mask.unsqueeze(0))[0]
        return {
            "frames": frames.float(),
            "current_frame": frames[-1].float(),
            "target_frame": target.float(),
            "mask": mask.float(),
            "corrupted_frame": corrupted.float(),
            "warped_references": frames[:-1].clone().float(),
            "sequence_name": name,
            "frame_paths": paths,
        }
