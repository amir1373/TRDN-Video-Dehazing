from pathlib import Path

import torch

from src.dataset import REVIDESequenceDataset
from tests.conftest import make_fake_revide_root


def test_dehaze_mode_no_ground_truth_leak(tmp_path: Path):
    """Problem 1 fix: in dehaze mode, nothing handed to the model may equal the
    ground-truth target, and every reference/current frame must come from a
    hazy_files path, never a gt/clean path."""
    root = make_fake_revide_root(tmp_path / "train", ["seq_a", "seq_b"], num_frames=6)
    dataset = REVIDESequenceDataset(
        str(root), split="train", seq_len=3, crop_size=8, random_crop=False, train_mode="dehaze", val_fraction=0.0
    )
    assert len(dataset) > 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        frames = sample["frames"]
        target = sample["target_frame"]
        corrupted = sample["corrupted_frame"]

        # No frame handed to the model (references or current) equals the target.
        for t in range(frames.shape[0]):
            assert not torch.equal(frames[t], target), f"frame {t} leaked ground truth at sample {idx}"
        assert not torch.equal(corrupted, target)

        # Every path used to build this sample is a hazy_files path, not gt.
        for path in sample["frame_paths"]:
            assert "hazy" in Path(path).parts, f"expected a hazy path, got {path}"
            assert "gt" not in Path(path).parts


def test_reconstruct_synthetic_mode_still_available_and_warns(tmp_path: Path, caplog):
    """The legacy path must still work (needed to reproduce old numbers) and
    must warn loudly that it is not a dehazing evaluation."""
    import logging

    root = make_fake_revide_root(tmp_path / "train", ["seq_a"], num_frames=6)
    with caplog.at_level(logging.WARNING):
        dataset = REVIDESequenceDataset(
            str(root),
            split="train",
            seq_len=3,
            crop_size=8,
            random_crop=False,
            train_mode="reconstruct_synthetic",
            val_fraction=0.0,
        )
    assert any("NOT a" in record.message or "not a dehazing" in record.message.lower() for record in caplog.records)
    sample = dataset[0]
    # Legacy behavior preserved: references are clean, current is corrupted-clean.
    assert torch.equal(sample["frames"][0], sample["clean_frames"][0])


def test_legacy_train_mode_alias_warns_and_resolves(tmp_path: Path, caplog):
    import logging

    root = make_fake_revide_root(tmp_path / "train", ["seq_a"], num_frames=6)
    with caplog.at_level(logging.WARNING):
        dataset = REVIDESequenceDataset(
            str(root), split="train", seq_len=3, crop_size=8, random_crop=False, train_mode="reconstruct", val_fraction=0.0
        )
    assert dataset.train_mode == "reconstruct_synthetic"
    assert any("deprecated alias" in record.message for record in caplog.records)
