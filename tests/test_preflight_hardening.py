from pathlib import Path

import pytest

from scripts.preflight import assert_checkpoint_storage_fits
from scripts.smoke_test import make_fake_revide_tree
from src.config import TRDNConfig
from src.dataset import REVIDESequenceDataset
from src.provenance import (
    ensure_project_config_compatible,
    find_seed_mismatches,
    write_json,
)


def test_preflight_disk_projection_fails_when_retention_will_not_fit():
    with pytest.raises(RuntimeError, match="Insufficient free disk"):
        assert_checkpoint_storage_fits(
            checkpoint_bytes=1_000,
            retained_count=4,
            free_disk_bytes=3_999,
        )


def test_dataset_count_mismatch_has_specific_early_error(tmp_path: Path):
    make_fake_revide_tree(tmp_path)
    (tmp_path / "Train" / "train_0" / "clean" / "0003.png").unlink()
    dataset = REVIDESequenceDataset(
        str(tmp_path / "Train"),
        split=None,
        seq_len=2,
        crop_size=16,
        synthetic_if_empty=False,
        train_mode="dehaze",
    )

    with pytest.raises(RuntimeError, match=r"train_0: hazy=4 clean=3"):
        dataset.assert_valid_structure("train")


def test_project_root_rejects_different_ablation_config(tmp_path: Path):
    config = TRDNConfig(project_root=str(tmp_path), use_raft_alignment=True)
    ensure_project_config_compatible(config)
    changed = TRDNConfig(project_root=str(tmp_path), use_raft_alignment=False)

    with pytest.raises(RuntimeError, match="Output directory collision"):
        ensure_project_config_compatible(changed)

    changed.allow_output_collision = True
    ensure_project_config_compatible(changed)


def test_project_root_checks_existing_manifest_without_marker(tmp_path: Path):
    existing = TRDNConfig(project_root=str(tmp_path), use_raft_alignment=True)
    write_json(
        tmp_path / "logs" / "runs" / "old" / "run_manifest.json",
        {"config": existing.to_dict()},
    )
    changed = TRDNConfig(project_root=str(tmp_path), use_raft_alignment=False)

    with pytest.raises(RuntimeError, match=r"run_manifest\.json"):
        ensure_project_config_compatible(changed)

    assert not (tmp_path / "project_config.json").exists()


def test_seed_warning_finds_sibling_project_manifest(tmp_path: Path):
    current = TRDNConfig(project_root=str(tmp_path / "full"), seed=1234)
    sibling = TRDNConfig(project_root=str(tmp_path / "no_raft"), seed=999)
    write_json(
        tmp_path / "no_raft" / "logs" / "runs" / "run" / "run_manifest.json",
        {"config": sibling.to_dict()},
    )

    mismatches = find_seed_mismatches(tmp_path / "full" / "logs", current)

    assert len(mismatches) == 1
    assert mismatches[0]["existing_seed"] == 999
    assert mismatches[0]["current_seed"] == 1234
