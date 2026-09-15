from pathlib import Path

from src.config import TRDNConfig
from src.dataset import REVIDESequenceDataset, split_train_val_sequence_names
from tests.conftest import make_fake_revide_root


def test_split_function_disjoint_and_deterministic():
    names = [f"seq_{i:03d}" for i in range(40)]
    train_a, val_a = split_train_val_sequence_names(names, val_fraction=0.1, seed=1234)
    train_b, val_b = split_train_val_sequence_names(names, val_fraction=0.1, seed=1234)

    assert set(train_a).isdisjoint(set(val_a))
    assert set(train_a) | set(val_a) == set(names)
    assert train_a == train_b and val_a == val_b  # deterministic given the same seed
    assert 2 <= len(val_a) <= 6  # ~10% of 40, tolerant of rounding


def test_split_function_different_seed_gives_different_partition():
    names = [f"seq_{i:03d}" for i in range(40)]
    _, val_a = split_train_val_sequence_names(names, val_fraction=0.1, seed=1234)
    _, val_b = split_train_val_sequence_names(names, val_fraction=0.1, seed=9999)
    assert val_a != val_b


def test_dataset_train_val_split_shares_no_sequence(tmp_path: Path):
    """Problem 2 fix: val is a held-out subset of TRAIN sequences, disjoint from
    the train split actually used, and root_for_split('val') must not be
    test_root."""
    sequence_names = [f"seq_{i:03d}" for i in range(20)]
    train_root = make_fake_revide_root(tmp_path / "Train", sequence_names, num_frames=4, size=8)
    test_root = make_fake_revide_root(tmp_path / "Test", ["holdout_seq_a", "holdout_seq_b"], num_frames=4, size=8)

    config = TRDNConfig(train_root=str(train_root), test_root=str(test_root), val_fraction=0.1, split_seed=1234)
    assert config.root_for_split("val") == config.train_root
    assert config.root_for_split("validation") == config.train_root
    assert config.root_for_split("test") == config.test_root
    assert config.root_for_split("test") != config.root_for_split("val")

    train_dataset = REVIDESequenceDataset(
        config.root_for_split("train"),
        split="train",
        seq_len=2,
        crop_size=8,
        random_crop=False,
        train_mode="dehaze",
        val_fraction=config.val_fraction,
        split_seed=config.split_seed,
    )
    val_dataset = REVIDESequenceDataset(
        config.root_for_split("val"),
        split="val",
        seq_len=2,
        crop_size=8,
        random_crop=False,
        train_mode="dehaze",
        val_fraction=config.val_fraction,
        split_seed=config.split_seed,
    )
    test_dataset = REVIDESequenceDataset(
        config.root_for_split("test"),
        split="test",
        seq_len=2,
        crop_size=8,
        random_crop=False,
        train_mode="dehaze",
        val_fraction=0.0,  # test split is not partitioned; it is a separate root entirely
    )

    train_names = {s["name"] for s in train_dataset.sequences}
    val_names = {s["name"] for s in val_dataset.sequences}
    test_names = {s["name"] for s in test_dataset.sequences}

    assert train_names, "expected non-empty train split"
    assert val_names, "expected non-empty val split"
    assert test_names, "expected non-empty test split"
    assert train_names.isdisjoint(val_names)
    assert train_names.isdisjoint(test_names)
    assert val_names.isdisjoint(test_names)
    assert train_names | val_names == set(sequence_names)
