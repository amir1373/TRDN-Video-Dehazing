import json
from contextlib import ExitStack
from pathlib import Path

import pytest
from accelerate.utils import load

import src.train as train_module
from scripts.smoke_test import (
    _patch_pipeline,
    make_fake_revide_tree,
    run_smoke,
    tiny_training_loss,
)
from src.config import TRDNConfig


def test_complete_cpu_smoke_pipeline(tmp_path: Path):
    report = run_smoke(tmp_path)

    assert report["status"] == "PASS"
    assert report["resumed_training_step"] == 4.0
    assert report["evaluation_accounting"] == {
        "clips_total_found": 2,
        "clips_evaluated": 1,
        "clips_skipped": 1,
    }


def test_crash_resume_restores_optimizer_and_appends_metrics(tmp_path: Path):
    dataset_root = tmp_path / "REVIDE"
    project_root = tmp_path / "run"
    make_fake_revide_tree(dataset_root)
    config = TRDNConfig(
        project_root=str(project_root),
        mixed_precision="no",
        seq_len=2,
        crop_size=16,
        batch_size=1,
        num_workers=0,
        max_train_steps=4,
        num_epochs=2,
        validate_every=99,
        checkpoint_every=1,
        log_every=1,
        num_inference_steps=1,
        enable_ema=True,
        run_name="crash",
    )
    config.override_dataset_root(str(dataset_root))
    calls = 0

    def crash_after_checkpoint(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated abrupt worker failure")
        return tiny_training_loss(*args, **kwargs)

    with ExitStack() as stack:
        _patch_pipeline(stack)
        stack.enter_context(
            pytest.MonkeyPatch.context()
        ).setattr(train_module, "compute_training_loss", crash_after_checkpoint)
        with pytest.raises(RuntimeError, match="simulated abrupt worker failure"):
            train_module.train_trdn(config)

    checkpoint = project_root / "checkpoints" / "step_000002"
    assert (checkpoint / "ema_weights.pt").is_file()
    optimizer_before = load(str(checkpoint / "optimizer.bin"), map_location="cpu")
    assert {int(state["step"].item()) for state in optimizer_before["state"].values()} == {2}

    resume = TRDNConfig(**config.to_dict())
    resume.resume_from_checkpoint = str(checkpoint)
    with ExitStack() as stack:
        _patch_pipeline(stack)
        result = train_module.train_trdn(resume)

    assert result["step"] == 4.0
    final_optimizer = load(
        str(project_root / "checkpoints" / "last" / "optimizer.bin"),
        map_location="cpu",
    )
    assert {int(state["step"].item()) for state in final_optimizer["state"].values()} == {4}
    assert (project_root / "checkpoints" / "last" / "ema_weights.pt").is_file()
    metrics_path = project_root / "logs" / "runs" / "crash" / "metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in rows if row["event"] == "step"] == [1, 2, 3, 4]
    manifest = json.loads(
        (metrics_path.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["resumes"][-1]["from_step"] == 2
