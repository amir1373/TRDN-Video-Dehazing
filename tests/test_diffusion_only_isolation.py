from pathlib import Path

import torch

from scripts.evaluate_full_test import load_runtime_for_eval
from src.config import TRDNConfig
from scripts.smoke_test import tiny_backbone


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
    assert runtime["raft_model"] is None
