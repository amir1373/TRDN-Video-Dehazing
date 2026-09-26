"""Long-range causal reference retrieval (experiment R1), shared verbatim by both models.

For each target frame, the references are not simply the nine preceding frames: the up-to-30
preceding frames of the same video are scored from the hazy input alone and the best nine are
kept, in time order. The score of a candidate c for target f is

    cosine(thumb_f, thumb_c) + CLARITY_WEIGHT * contrast_c / max(contrast over candidates)

where thumb is the mean-subtracted 64x64 luminance of the central 256x256 crop (the evaluated
region) and contrast is its RMS contrast, so frames that show the same content, and show it
more clearly, are preferred. No clean frame is used. build_index() writes a JSON map from the
resolved target-frame path to the resolved paths of its nine references.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

SEARCH_WINDOW = 30
NUM_REFERENCES = 9
CLARITY_WEIGHT = 0.25
CROP = 256
THUMB = 64


def frame_descriptor(path: str) -> tuple[np.ndarray, float]:
    with Image.open(path) as image:
        width, height = image.size
        crop = min(CROP, width, height)
        top, left = (height - crop) // 2, (width - crop) // 2
        patch = image.convert("L").crop((left, top, left + crop, top + crop)).resize((THUMB, THUMB), Image.BILINEAR)
    lum = np.asarray(patch, dtype=np.float32) / 255.0
    contrast = float(lum.std())
    centred = (lum - lum.mean()).ravel()
    norm = float(np.linalg.norm(centred))
    return centred / max(norm, 1e-6), contrast


def select_references(descriptors: Sequence[tuple[np.ndarray, float]], end: int) -> List[int]:
    """Indices of the NUM_REFERENCES retrieved references for target index `end` (time order)."""
    start = max(0, end - SEARCH_WINDOW)
    candidates = list(range(start, end))
    if len(candidates) <= NUM_REFERENCES:
        return candidates
    target = descriptors[end][0]
    max_contrast = max(descriptors[c][1] for c in candidates) or 1.0
    scores = [float(target @ descriptors[c][0]) + CLARITY_WEIGHT * descriptors[c][1] / max_contrast for c in candidates]
    best = sorted(range(len(candidates)), key=lambda i: (-scores[i], -candidates[i]))[:NUM_REFERENCES]
    return sorted(candidates[i] for i in best)


def build_index(videos: Dict[str, List[str]], out_path: str) -> Dict[str, object]:
    """videos: name -> ordered hazy frame paths of one continuous video. Writes the JSON index."""
    index: Dict[str, List[str]] = {}
    same_as_contiguous, spans = 0, []
    for name, paths in videos.items():
        real = [os.path.realpath(p) for p in paths]
        descriptors = [frame_descriptor(p) for p in real]
        for end in range(NUM_REFERENCES, len(real)):
            refs = select_references(descriptors, end)
            index[real[end]] = [real[r] for r in refs]
            same_as_contiguous += refs == list(range(end - NUM_REFERENCES, end))
            spans.append(end - refs[0])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(index))
    stats = {"targets": len(index), "identical_to_contiguous": same_as_contiguous,
             "mean_reach_frames": float(np.mean(spans)) if spans else 0.0,
             "max_reach_frames": int(max(spans)) if spans else 0,
             "search_window": SEARCH_WINDOW, "num_references": NUM_REFERENCES, "clarity_weight": CLARITY_WEIGHT}
    Path(out_path + ".stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def load_index(path: str) -> Dict[str, List[str]]:
    return json.loads(Path(path).read_text())


def clean_path_for(hazy_path: str) -> str:
    """REVIDE_sequences keeps pairs as <split>/hazy/<seq>/<frame> and <split>/gt/<seq>/<frame>."""
    return hazy_path.replace("/hazy/", "/gt/")
