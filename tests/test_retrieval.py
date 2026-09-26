import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import REVIDESequenceDataset
from src.retrieval import build_index, select_references


def test_select_references_is_causal_sorted_and_prefers_similar_frames():
    rng = np.random.default_rng(0)
    base = [rng.standard_normal(16) for _ in range(3)]
    descs = []
    for i in range(40):
        v = base[i % 3] + 0.01 * rng.standard_normal(16)
        descs.append((v / np.linalg.norm(v), 0.1))
    refs = select_references(descs, 39)
    assert len(refs) == 9 and refs == sorted(refs) and all(9 <= r < 39 for r in refs)
    assert all(r % 3 == 39 % 3 for r in refs)          # the same "content" as the target
    assert select_references(descs, 5) == [0, 1, 2, 3, 4]


def _fake_root(tmp: Path, n: int = 14) -> Path:
    root = tmp / "train"
    for kind in ("hazy", "gt"):
        (root / "seq_a" / kind).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for i in range(n):
        img = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(root / "seq_a" / "gt" / f"{i:03d}.png")
        Image.fromarray((img * 0.5 + 100).astype(np.uint8)).save(root / "seq_a" / "hazy" / f"{i:03d}.png")
    return root


def test_dataset_uses_retrieved_references(tmp_path: Path):
    root = _fake_root(tmp_path)
    ds = REVIDESequenceDataset(str(root), split="train", seq_len=10, crop_size=32, random_crop=False,
                               train_mode="dehaze", val_fraction=0.0)
    seq = ds.sequences[0]
    # A hand-made index: target 12 retrieves frames 0..8 (a long reach).
    import os
    index = {os.path.realpath(seq["hazy_files"][12]): [os.path.realpath(seq["hazy_files"][i]) for i in range(9)]}
    path = tmp_path / "idx.json"
    path.write_text(json.dumps(index))
    ds_r = REVIDESequenceDataset(str(root), split="train", seq_len=10, crop_size=32, random_crop=False,
                                 train_mode="dehaze", val_fraction=0.0, retrieval_index=str(path))
    target_pos = [i for i, (s, e) in enumerate(ds_r.index) if e == 12][0]
    sample = ds_r[target_pos]
    assert Path(sample["frame_paths"][0]).name == "000.png" and Path(sample["frame_paths"][-1]).name == "012.png"
    plain = ds[[i for i, (s, e) in enumerate(ds.index) if e == 12][0]]
    assert torch.equal(sample["target_frame"], plain["target_frame"])
    assert torch.equal(sample["frames"][-1], plain["frames"][-1])
    assert "prev_frames" in sample and sample["prev_frames"].shape == sample["frames"].shape
    stats = build_index({"v": [str(p) for p in seq["hazy_files"]]}, str(tmp_path / "built.json"))
    assert stats["targets"] == len(seq["hazy_files"]) - 9
