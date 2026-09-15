import argparse
from pathlib import Path

import torch

from scripts.evaluate_full_test import evaluate, load_runtime_for_eval
from src.config import TRDNConfig
from scripts.smoke_test import (
    SmokeLossBundle,
    make_fake_revide_tree,
    tiny_backbone,
    tiny_diffusion_inference,
)


def test_diffusion_only_runtime_never_constructs_temporal_or_raft(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(
        "scripts.evaluate_full_test.validate_checkpoint_modes",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "scripts.evaluate_full_test.load_diffusion_backbone",
        tiny_backbone,
    )
    monkeypatch.setattr(
        "scripts.evaluate_full_test._load_unet_only_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.evaluate_full_test.build_temporal_modules",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("temporal modules must not be constructed")
        ),
    )
    monkeypatch.setattr(
        "scripts.evaluate_full_test.load_raft",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("RAFT must not be loaded")
        ),
    )

    runtime = load_runtime_for_eval(
        TRDNConfig(mixed_precision="no"),
        str(checkpoint),
        "cpu",
        diffusion_only=True,
    )

    assert isinstance(runtime["diffusion"]["unet"], torch.nn.Module)
    assert runtime["temporal_memory"] is None
    assert runtime["temporal_transformer"] is None
    assert runtime["reference_selector"] is None
    assert runtime["conditioning_adapter"] is None
    assert runtime["model_raft"] is None


def test_diffusion_only_evaluation_uses_metric_raft_but_no_model_raft(
    tmp_path: Path,
    monkeypatch,
):
    dataset_root = tmp_path / "REVIDE"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    make_fake_revide_tree(dataset_root)
    monkeypatch.setattr(
        "scripts.evaluate_full_test.validate_checkpoint_modes",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "scripts.evaluate_full_test.load_diffusion_backbone",
        tiny_backbone,
    )
    monkeypatch.setattr(
        "scripts.evaluate_full_test._load_unet_only_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.evaluate_full_test.build_temporal_modules",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("temporal model components must not be constructed")
        ),
    )
    metric_raft = object()
    raft_loads = []

    def load_metric_raft(*_args, **_kwargs):
        raft_loads.append(True)
        return metric_raft

    monkeypatch.setattr("scripts.evaluate_full_test.load_raft", load_metric_raft)
    monkeypatch.setattr("scripts.evaluate_full_test.LossBundle", SmokeLossBundle)
    monkeypatch.setattr(
        "scripts.evaluate_full_test.infer_diffusion_only_batch",
        tiny_diffusion_inference,
    )
    monkeypatch.setattr(
        "scripts.evaluate_full_test.flow_warped_temporal_consistency_error",
        lambda previous, current, raft: (
            float(torch.abs(previous - current).mean()),
            1.0,
        )
        if raft is metric_raft
        else (_ for _ in ()).throw(AssertionError("wrong RAFT instance")),
    )
    config = TRDNConfig(
        dataset_root=str(dataset_root),
        train_root=str(dataset_root / "Train"),
        test_root=str(dataset_root / "Test"),
        mixed_precision="no",
        seq_len=2,
        crop_size=16,
    )
    args = argparse.Namespace(
        checkpoint=str(checkpoint),
        crop_size=16,
        train_mode="dehaze",
        mask_mode="auto",
        diffusion_only=True,
        num_steps=1,
        seed=1234,
        variant="diffusion_only",
        use_ema=False,
    )

    report = evaluate(config, args, "cpu")

    assert len(raft_loads) == 1
    assert report["aggregate"]["temporal_consistency_l1"]["mean"] is not None
    assert report["per_clip"][0]["temporal_consistency_l1_mean"] is not None
    assert report["temporal_metric"]["metric_raft_is_independent_of_model"] is True
