import hashlib
import logging
import random
import re
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

# "reconstruct" is the legacy name for what is now called "reconstruct_synthetic".
# It leaks ground truth into the temporal references (see the branch below) and
# is kept only so previously-reported numbers can be explained/reproduced.
TRAIN_MODES = {"dehaze", "reconstruct_synthetic"}
_LEGACY_TRAIN_MODE_ALIASES = {"reconstruct": "reconstruct_synthetic"}

logger = logging.getLogger(__name__)
_NATURAL_PARTS = re.compile(r"(\d+)")
_STEM_TOKENS = re.compile(r"[a-z]+|\d+")
_MODALITY_TOKENS = {
    "clean",
    "clear",
    "degraded",
    "fog",
    "ground",
    "groundtruth",
    "gt",
    "hazy",
    "input",
    "target",
    "truth",
}


def _normalize_train_mode(train_mode: str) -> str:
    if train_mode in _LEGACY_TRAIN_MODE_ALIASES:
        resolved = _LEGACY_TRAIN_MODE_ALIASES[train_mode]
        logger.warning(
            "train_mode=%r is a deprecated alias for %r. Update callers to use %r directly.",
            train_mode,
            resolved,
            resolved,
        )
        return resolved
    if train_mode not in TRAIN_MODES:
        raise ValueError(f"train_mode must be one of {sorted(TRAIN_MODES)}, got {train_mode!r}")
    return train_mode


def image_to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def natural_sort_key(value: str | Path) -> tuple:
    """Case-insensitive natural key, so frame_2 sorts before frame_10."""
    text = Path(value).name if isinstance(value, Path) else str(value)
    parts = tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in _NATURAL_PARTS.split(text)
        if part
    )
    return (*parts, (2, text.casefold()))


def _normalized_frame_stem(path: Path) -> tuple:
    tokens = _STEM_TOKENS.findall(path.stem.casefold())
    normalized = [
        int(token) if token.isdigit() else token
        for token in tokens
        if token not in _MODALITY_TOKENS
    ]
    return tuple(normalized)


def frame_pairing_report(
    hazy_files: List[Path],
    clean_files: List[Path],
    *,
    sample_size: int = 5,
) -> Dict[str, Any]:
    """Validate natural-position pairing using exact or modality-normalized stems.

    Files are natural-sorted first. Each positional pair must then have either
    the same case-insensitive stem, or the same token sequence after removing
    only documented modality tokens (for example ``hazy_2`` and ``clean_2``).
    """
    pairs = list(zip(hazy_files, clean_files))
    exact_matches = [
        hazy.stem.casefold() == clean.stem.casefold()
        for hazy, clean in pairs
    ]
    normalized_matches = [
        bool(_normalized_frame_stem(hazy))
        and _normalized_frame_stem(hazy) == _normalized_frame_stem(clean)
        for hazy, clean in pairs
    ]
    consistent = (
        len(hazy_files) == len(clean_files)
        and all(exact or normalized for exact, normalized in zip(exact_matches, normalized_matches))
    )
    differing_pairs = [
        {
            "hazy": hazy.name,
            "clean": clean.name,
            "hazy_normalized_stem": list(_normalized_frame_stem(hazy)),
            "clean_normalized_stem": list(_normalized_frame_stem(clean)),
        }
        for (hazy, clean), exact in zip(pairs, exact_matches)
        if not exact
    ]
    return {
        "rule": "natural_sort_then_exact_or_known_modality_normalized_stem",
        "consistent": consistent,
        "exact_stems": bool(pairs) and all(exact_matches),
        "inference_used": bool(differing_pairs),
        "sample_inferred_pairs": differing_pairs[:sample_size],
    }


def list_images(folder: Path, extensions: Tuple[str, ...]) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=natural_sort_key,
    )


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


def _sequence_name_fraction(name: str, seed: int) -> float:
    """Deterministic pseudo-random value in [0, 1) derived from (seed, name)."""
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_train_val_sequence_names(
    sequence_names: List[str], val_fraction: float = 0.1, seed: int = 1234
) -> Tuple[List[str], List[str]]:
    """Deterministically partition sequence names into (train, val) with no overlap.

    Whole sequences are assigned to one side or the other (never split within a
    sequence), so clips never leak across the train/val boundary. The partition
    only depends on (seed, sequence name), not on filesystem order, so it is
    stable across reruns and across changes to how many sequences exist.
    """
    unique_names = sorted(set(sequence_names))
    if not unique_names:
        return [], []
    ranked = sorted(unique_names, key=lambda name: _sequence_name_fraction(name, seed))
    val_count = round(len(unique_names) * val_fraction)
    if val_fraction > 0 and val_count == 0 and len(unique_names) > 1:
        val_count = 1  # guarantee a non-empty val set whenever a fraction was requested and it's feasible
    val_count = min(val_count, len(unique_names) - 1) if len(unique_names) > 1 else 0
    val_names = set(ranked[:val_count])
    train_names = [name for name in unique_names if name not in val_names]
    return sorted(train_names), sorted(val_names)


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
        train_mode: str = "dehaze",
        mask_mode: str = "auto",
        val_fraction: float = 0.1,
        split_seed: int = 1234,
        include_prev_frame: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.seq_len = seq_len
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.extensions = extensions
        self.synthetic_if_empty = synthetic_if_empty
        self.train_mode = _normalize_train_mode(train_mode)
        if self.train_mode == "reconstruct_synthetic":
            logger.warning(
                "=" * 88
                + "\nREVIDESequenceDataset: train_mode='reconstruct_synthetic' is ACTIVE.\n"
                "Temporal references are GROUND-TRUTH CLEAN frames, and the 'hazy' input is\n"
                "the clean target frame with SYNTHETIC haze painted inside a random mask.\n"
                "Real REVIDE haze never reaches the model in this mode. This is NOT a\n"
                "dehazing evaluation; it exists only to reproduce previously-reported numbers.\n"
                "Use train_mode='dehaze' to measure real video dehazing.\n" + "=" * 88
            )
        self.mask_mode = mask_mode

        # Predictive temporal-consistency training (see src/losses.py) needs the
        # previous frame's own T-length window so a genuine "previous frame
        # prediction" can be computed and warped forward. Only dehaze mode uses
        # this; reconstruct_synthetic keeps its original sample count/pairing
        # unchanged so legacy numbers stay reproducible. Evaluation scripts that
        # compute temporal consistency directly from consecutive real
        # predictions (not needed at training time) can pass
        # include_prev_frame=False to skip the extra I/O.
        self.needs_prev_frame = self.train_mode == "dehaze" and include_prev_frame

        self.sequences = discover_revide_sequences(self.root, split, extensions)
        if max_sequences is not None:
            self.sequences = self.sequences[:max_sequences]

        if split and split.lower() in {"train", "training", "val", "valid", "validation"} and val_fraction > 0:
            all_names = [sequence["name"] for sequence in self.sequences]
            train_names, val_names = split_train_val_sequence_names(all_names, val_fraction, split_seed)
            keep = set(val_names) if split.lower() in {"val", "valid", "validation"} else set(train_names)
            self.sequences = [sequence for sequence in self.sequences if sequence["name"] in keep]

        self.index: List[Tuple[int, int]] = []
        self.skipped_sequences: List[Dict[str, Any]] = []
        min_window = seq_len + 1 if self.needs_prev_frame else seq_len
        for seq_idx, sequence in enumerate(self.sequences):
            hazy_count = len(sequence["hazy_files"])
            clean_count = len(sequence["clean_files"])
            if hazy_count != clean_count:
                self.skipped_sequences.append(
                    {
                        "sequence_name": sequence["name"],
                        "reason": "hazy_clean_count_mismatch",
                        "hazy_frames": hazy_count,
                        "clean_frames": clean_count,
                    }
                )
                continue
            pairing = frame_pairing_report(
                sequence["hazy_files"],
                sequence["clean_files"],
            )
            sequence["pairing_report"] = pairing
            if not pairing["consistent"]:
                self.skipped_sequences.append(
                    {
                        "sequence_name": sequence["name"],
                        "reason": "frame_pairing_mismatch",
                        "pairing_report": pairing,
                    }
                )
                continue
            count = hazy_count
            if count < min_window:
                self.skipped_sequences.append(
                    {
                        "sequence_name": sequence["name"],
                        "reason": "too_short_for_seq_len",
                        "hazy_frames": hazy_count,
                        "clean_frames": clean_count,
                        "required_frames": min_window,
                    }
                )
                continue
            for end_idx in range(min_window - 1, count):
                self.index.append((seq_idx, end_idx))
        self.synthetic_len = 8 if not self.index and synthetic_if_empty else 0

    def layout_inventory(self) -> Dict[str, Any]:
        skipped = {
            item["sequence_name"]: item
            for item in self.skipped_sequences
        }
        sequences = []
        for sequence in self.sequences:
            status = skipped.get(sequence["name"])
            pairing = sequence.get("pairing_report") or frame_pairing_report(
                sequence["hazy_files"],
                sequence["clean_files"],
            )
            sequences.append(
                {
                    "sequence_name": sequence["name"],
                    "hazy_dir": str(sequence["hazy_dir"]),
                    "clean_dir": str(sequence["clean_dir"]),
                    "hazy_frames": len(sequence["hazy_files"]),
                    "clean_frames": len(sequence["clean_files"]),
                    "pairing": pairing,
                    "status": "skipped" if status else "eligible",
                    "reason": status.get("reason") if status else None,
                }
            )
        return {
            "root": str(self.root.resolve()),
            "sequence_count": len(self.sequences),
            "eligible_sequence_count": len(self.sequences) - len(self.skipped_sequences),
            "skipped_sequence_count": len(self.skipped_sequences),
            "sequences": sequences,
        }

    def assert_valid_structure(self, split: str) -> None:
        if not self.sequences:
            raise RuntimeError(
                f"{split} dataset structure mismatch at {self.root}: "
                "no directories containing both hazy and clean image folders were found."
            )
        count_mismatches = [
            item
            for item in self.skipped_sequences
            if item["reason"] == "hazy_clean_count_mismatch"
        ]
        if count_mismatches:
            details = "; ".join(
                f"{item['sequence_name']}: hazy={item['hazy_frames']} clean={item['clean_frames']}"
                for item in count_mismatches
            )
            raise RuntimeError(
                f"{split} dataset structure mismatch: hazy/clean frame counts differ: {details}"
            )
        pairing_mismatches = [
            item
            for item in self.skipped_sequences
            if item["reason"] == "frame_pairing_mismatch"
        ]
        if pairing_mismatches:
            raise RuntimeError(
                f"{split} dataset frame pairing mismatch under the documented natural-sort "
                "and stem-normalization rule. Inferred sample pairs: "
                + str(
                    [
                        {
                            "sequence_name": item["sequence_name"],
                            **item["pairing_report"],
                        }
                        for item in pairing_mismatches[:5]
                    ]
                )
            )
        if not self.index:
            shortest = "; ".join(
                f"{item['sequence_name']}: found={item['hazy_frames']} "
                f"required={item.get('required_frames', self.seq_len)}"
                for item in self.skipped_sequences
            )
            raise RuntimeError(
                f"{split} dataset has no sequence long enough for seq_len={self.seq_len}. "
                f"{shortest or 'No paired frames were found.'}"
            )
        for sequence in self.sequences:
            if any(
                item["sequence_name"] == sequence["name"]
                for item in self.skipped_sequences
            ):
                continue
            expected_size = None
            for hazy_path, clean_path in zip(
                sequence["hazy_files"],
                sequence["clean_files"],
            ):
                try:
                    with Image.open(hazy_path) as hazy_image:
                        hazy_size = hazy_image.size
                        hazy_image.verify()
                    with Image.open(clean_path) as clean_image:
                        clean_size = clean_image.size
                        clean_image.verify()
                except (OSError, ValueError) as exc:
                    raise RuntimeError(
                        f"{split} dataset decode failure in sequence "
                        f"{sequence['name']}: {type(exc).__name__}: {exc}"
                    ) from exc
                if hazy_size != clean_size:
                    raise RuntimeError(
                        f"{split} dataset spatial mismatch in sequence "
                        f"{sequence['name']}: {hazy_path.name}={hazy_size} "
                        f"{clean_path.name}={clean_size}"
                    )
                if expected_size is None:
                    expected_size = hazy_size
                elif hazy_size != expected_size:
                    raise RuntimeError(
                        f"{split} dataset frame-size mismatch within sequence "
                        f"{sequence['name']}: expected={expected_size} "
                        f"found={hazy_size} at {hazy_path.name}"
                    )

    def _resolve_mask_mode(self) -> str:
        if self.mask_mode != "auto":
            return self.mask_mode
        return "full" if self.train_mode == "dehaze" else "mixed"

    def __len__(self) -> int:
        return len(self.index) if self.index else self.synthetic_len

    def _load_real_clip(self, idx: int) -> Dict[str, Any]:
        seq_idx, end_idx = self.index[idx]
        sequence = self.sequences[seq_idx]
        start_idx = end_idx - self.seq_len + 1
        ext_start = start_idx - 1 if self.needs_prev_frame else start_idx
        hazy_paths_ext = sequence["hazy_files"][ext_start : end_idx + 1]
        clean_paths_ext = sequence["clean_files"][ext_start : end_idx + 1]
        hazy_ext = torch.stack([image_to_tensor(path) for path in hazy_paths_ext], dim=0)
        clean_ext = torch.stack([image_to_tensor(path) for path in clean_paths_ext], dim=0)

        result: Dict[str, Any] = {
            "hazy_frames": hazy_ext[1:] if self.needs_prev_frame else hazy_ext,
            "clean_frames": clean_ext[1:] if self.needs_prev_frame else clean_ext,
            "paths": [str(path) for path in (hazy_paths_ext[1:] if self.needs_prev_frame else hazy_paths_ext)],
            "name": sequence["name"],
        }
        if self.needs_prev_frame:
            result["prev_hazy_frames"] = hazy_ext[:-1]
            result["prev_clean_frames"] = clean_ext[:-1]
        return result

    def _load_synthetic_clip(self, idx: int) -> Dict[str, Any]:
        height = width = max(self.crop_size, 256)
        total_len = self.seq_len + 1 if self.needs_prev_frame else self.seq_len
        yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
        clean_frames_ext = []
        for tidx in range(total_len):
            shift = 0.02 * (idx + tidx)
            clean_frames_ext.append(
                torch.stack(
                    [
                        (xx + shift).fmod(1.0),
                        (yy + 0.5 * shift).fmod(1.0),
                        (0.5 * xx + 0.5 * yy + shift).fmod(1.0),
                    ],
                    dim=0,
                )
            )
        clean_ext = torch.stack(clean_frames_ext, dim=0)
        hazy_ext = torch.stack([torch.clamp(frame * 0.65 + 0.35, 0, 1) for frame in clean_frames_ext], dim=0)

        offset = 1 if self.needs_prev_frame else 0
        result: Dict[str, Any] = {
            "hazy_frames": hazy_ext[offset:],
            "clean_frames": clean_ext[offset:],
            "paths": [f"synthetic_{idx}_{tidx}.png" for tidx in range(offset, total_len)],
            "name": "synthetic",
        }
        if self.needs_prev_frame:
            result["prev_hazy_frames"] = hazy_ext[:-1]
            result["prev_clean_frames"] = clean_ext[:-1]
        return result

    def _crop_pair(self, hazy_frames: torch.Tensor, clean_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        _, _, height, width = hazy_frames.shape
        crop = min(self.crop_size, height, width)
        if height == crop and width == crop:
            return hazy_frames, clean_frames, 0, 0, crop
        top = random.randint(0, height - crop) if self.random_crop else (height - crop) // 2
        left = random.randint(0, width - crop) if self.random_crop else (width - crop) // 2
        return (
            hazy_frames[:, :, top : top + crop, left : left + crop],
            clean_frames[:, :, top : top + crop, left : left + crop],
            top,
            left,
            crop,
        )

    def _build_window(self, hazy_frames: torch.Tensor, clean_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        target = clean_frames[-1]
        _, height, width = target.shape
        mask = generate_haze_mask(height, width, mode=self._resolve_mask_mode()).float()
        if self.train_mode == "dehaze":
            frames = hazy_frames
            corrupted = hazy_frames[-1]
        else:
            frames = clean_frames.clone()
            corrupted = simulate_realistic_haze(target.unsqueeze(0), mask.unsqueeze(0))[0]
            frames[-1] = corrupted
        return {
            "frames": frames.float(),
            "current_frame": frames[-1].float(),
            "target_frame": target.float(),
            "mask": mask.float(),
            "corrupted_frame": corrupted.float(),
            "warped_references": frames[:-1].clone().float(),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip = self._load_real_clip(idx) if self.index else self._load_synthetic_clip(idx)
        hazy_frames, clean_frames = clip["hazy_frames"], clip["clean_frames"]

        if self.needs_prev_frame:
            # Crop the extended (prev + current) window jointly so both share the
            # exact same spatial crop before splitting.
            hazy_ext = torch.cat([clip["prev_hazy_frames"][:1], hazy_frames], dim=0)
            clean_ext = torch.cat([clip["prev_clean_frames"][:1], clean_frames], dim=0)
            hazy_ext, clean_ext, *_ = self._crop_pair(hazy_ext, clean_ext)
            hazy_frames, clean_frames = hazy_ext[1:], clean_ext[1:]
            prev_hazy = torch.cat([hazy_ext[:1], hazy_ext[1:-1]], dim=0)
            prev_clean = torch.cat([clean_ext[:1], clean_ext[1:-1]], dim=0)
        else:
            hazy_frames, clean_frames, *_ = self._crop_pair(hazy_frames, clean_frames)

        sample = self._build_window(hazy_frames, clean_frames)
        sample.update(
            {
                "clean_frames": clean_frames.float(),
                "hazy_frames": hazy_frames.float(),
                "train_mode": self.train_mode,
                "sequence_name": clip["name"],
                "frame_paths": clip["paths"],
            }
        )
        if self.needs_prev_frame:
            prev_window = self._build_window(prev_hazy, prev_clean)
            sample["prev_frames"] = prev_window["frames"]
            sample["prev_current_frame"] = prev_window["current_frame"]
            sample["prev_target_frame"] = prev_window["target_frame"]
            sample["prev_mask"] = prev_window["mask"]
            sample["prev_corrupted_frame"] = prev_window["corrupted_frame"]
        return sample
