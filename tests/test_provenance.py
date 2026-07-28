import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import TRDNConfig
from src.provenance import (
    JsonlMetricLogger,
    checkpoint_metadata,
    find_numerics_mismatches,
    numerics_settings,
    prune_step_checkpoints,
    validate_checkpoint_modes,
)
from src.train import save_checkpoint, train_trdn


def test_dataset_root_override_preserves_train_test_directories(tmp_path: Path):
    (tmp_path / "Train").mkdir()
    (tmp_path / "Test").mkdir()
    config = TRDNConfig()

    config.override_dataset_root(str(tmp_path))

    assert Path(config.train_root) == tmp_path / "Train"
    assert Path(config.test_root) == tmp_path / "Test"


def test_resume_mode_mismatch_raises(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps({"train_mode": "reconstruct_synthetic", "mask_mode": "mixed"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="train_mode"):
        validate_checkpoint_modes(checkpoint, TRDNConfig(train_mode="dehaze", mask_mode="auto"))


def test_train_resume_fails_before_model_load(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps({"train_mode": "reconstruct_synthetic", "mask_mode": "mixed"}),
        encoding="utf-8",
    )
    model_load_attempted = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal model_load_attempted
        model_load_attempted = True
        raise AssertionError("model loading should not be reached")

    monkeypatch.setattr("src.train.load_diffusion_backbone", fail_if_called)
    config = TRDNConfig(
        project_root=str(tmp_path / "run"),
        resume_from_checkpoint=str(checkpoint),
        train_mode="dehaze",
        mask_mode="auto",
    )

    with pytest.raises(ValueError, match="train_mode"):
        train_trdn(config)
    assert model_load_attempted is False


def test_resume_mode_mismatch_requires_explicit_override(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps({"train_mode": "reconstruct_synthetic", "mask_mode": "mixed"}),
        encoding="utf-8",
    )
    config = TRDNConfig(train_mode="dehaze", mask_mode="auto", allow_mode_mismatch=True)

    metadata = validate_checkpoint_modes(checkpoint, config)

    assert metadata["train_mode"] == "reconstruct_synthetic"


def test_checkpoint_metadata_records_reproducibility_fields(tmp_path: Path):
    config = TRDNConfig(
        train_mode="dehaze",
        mask_mode="auto",
        seed=17,
        dataset_root="/data/revide",
        seq_len=7,
        crop_size=192,
        w_flow=0.125,
    )

    metadata = checkpoint_metadata(config, 12, 20.0, 0.8, tmp_path / "run_manifest.json")

    assert metadata["train_mode"] == "dehaze"
    assert metadata["mask_mode"] == "full"
    assert metadata["seed"] == 17
    assert metadata["dataset_root"] == "/data/revide"
    assert metadata["seq_len"] == 7
    assert metadata["crop_size"] == 192
    assert metadata["loss_weights"]["flow"] == 0.125
    assert metadata["numerics"] == numerics_settings(config)
    assert metadata["git_commit_sha"]


def test_existing_run_with_different_numerics_is_reported(tmp_path: Path):
    config = TRDNConfig(mixed_precision="bf16", batch_size=2)
    manifest_path = tmp_path / "runs" / "older" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    existing = numerics_settings(config)
    existing["mixed_precision"] = "fp16"
    existing["batch_size"] = 1
    manifest_path.write_text(json.dumps({"numerics": existing}), encoding="utf-8")

    mismatches = find_numerics_mismatches(tmp_path, config)

    assert len(mismatches) == 1
    assert mismatches[0]["differences"]["mixed_precision"] == {
        "existing": "fp16",
        "current": "bf16",
    }
    assert mismatches[0]["differences"]["batch_size"] == {
        "existing": 1,
        "current": 2,
    }


def test_checkpoint_retention_keeps_newest_steps_and_all_named_checkpoints(tmp_path: Path):
    for name in ("step_000010", "step_000020", "step_000030", "step_000040", "best_psnr", "best_ssim", "last"):
        (tmp_path / name).mkdir()

    removed = prune_step_checkpoints(tmp_path, keep_last_n=2)

    assert {path.name for path in removed} == {"step_000010", "step_000020"}
    assert {path.name for path in tmp_path.iterdir()} == {
        "step_000030",
        "step_000040",
        "best_psnr",
        "best_ssim",
        "last",
    }


def test_save_checkpoint_enforces_retention(tmp_path: Path):
    class FakeAccelerator:
        is_main_process = True

        @staticmethod
        def save_state(path: str):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.bin").write_bytes(b"state")

    config = TRDNConfig(keep_last_n_checkpoints=2)
    checkpoint_dir = tmp_path / "checkpoints"
    manifest_path = tmp_path / "run" / "run_manifest.json"
    for step in (10, 20, 30, 40):
        save_checkpoint(
            FakeAccelerator(),
            checkpoint_dir,
            step,
            1.0,
            0.5,
            config,
            manifest_path,
        )
    save_checkpoint(
        FakeAccelerator(),
        checkpoint_dir,
        40,
        1.0,
        0.5,
        config,
        manifest_path,
        "best_psnr",
    )

    assert {path.name for path in checkpoint_dir.iterdir()} == {
        "step_000030",
        "step_000040",
        "best_psnr",
    }


def test_jsonl_metric_logger_appends(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonlMetricLogger(path)

    logger.append({"event": "step", "step": 1, "total_loss": 2.0})
    logger.append({"event": "epoch", "epoch": 1, "total_loss": 1.5})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["step", "epoch"]
    assert rows[1]["total_loss"] == 1.5


def test_lightweight_training_smoke_writes_manifest_and_metrics(tmp_path: Path, monkeypatch):
    class TinyDataset(Dataset):
        def __init__(self, length: int):
            self.length = length
            self.sequences = [{"name": "tiny"}]
            self.index = [(0, index) for index in range(length)]
            self.synthetic_len = 0

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            return {"index": torch.tensor(index)}

    class FakeAccelerator:
        is_main_process = True
        sync_gradients = True
        device = torch.device("cpu")

        def __init__(self, **_kwargs):
            self.backward_calls = 0

        @property
        def optimizer_step_was_skipped(self):
            return self.backward_calls == 2

        def init_trackers(self, *_args, **_kwargs):
            pass

        def prepare(self, *items):
            return items

        def accumulate(self, _module):
            return nullcontext()

        def autocast(self):
            return nullcontext()

        def backward(self, loss):
            self.backward_calls += 1
            loss.backward()

        @staticmethod
        def clip_grad_norm_(parameters, max_norm):
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

        @staticmethod
        def save_state(path):
            destination = Path(path)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "state.bin").write_bytes(b"state")

        def log(self, *_args, **_kwargs):
            pass

        def end_training(self):
            pass

    class TinyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.5]))

        def forward(self, value):
            return value * self.weight

    unet = TinyModule()
    unet.config = SimpleNamespace(cross_attention_dim=8)
    diffusion = {
        "unet": unet,
        "vae": TinyModule(),
        "text_encoder": TinyModule(),
    }
    modules = [TinyModule() for _ in range(4)]
    train_dataset = TinyDataset(4)
    val_dataset = TinyDataset(2)
    test_dataset = TinyDataset(3)

    monkeypatch.setattr("src.train.Accelerator", FakeAccelerator)
    monkeypatch.setattr("src.train.load_diffusion_backbone", lambda *_args, **_kwargs: diffusion)
    monkeypatch.setattr("src.train.build_temporal_modules", lambda *_args, **_kwargs: tuple(modules))
    monkeypatch.setattr("src.train.LossBundle", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "src.train.make_dataloaders",
        lambda _config: (
            DataLoader(train_dataset, batch_size=1),
            DataLoader(val_dataset, batch_size=1),
        ),
    )
    monkeypatch.setattr("src.train.make_test_dataset_for_manifest", lambda _config: test_dataset)

    def tiny_loss(_accelerator, diffusion_arg, *_args, **_kwargs):
        base = diffusion_arg["unet"].weight.square().sum()
        batch = _args[6]
        if int(batch["index"].item()) == 1:
            base = base * torch.tensor(float("nan"))
        parts = {
            "diffusion": base,
            "l1": base,
            "lpips": base,
            "temporal": base,
            "flow": base,
        }
        return base, parts

    monkeypatch.setattr("src.train.compute_training_loss", tiny_loss)
    config = TRDNConfig(
        project_root=str(tmp_path),
        mixed_precision="no",
        max_train_steps=3,
        num_epochs=2,
        num_workers=0,
        log_every=1,
        validate_every=99,
        checkpoint_every=2,
        run_name="smoke",
    )

    result = train_trdn(config)

    run_dir = tmp_path / "logs" / "runs" / "smoke"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result["step"] == 3.0
    assert manifest["status"] == "completed"
    assert manifest["dataset_sizes"]["train"]["num_clips"] == 4
    assert manifest["training"]["wall_clock_seconds"] >= 0.0
    assert manifest["training"]["non_finite_loss_steps"] == 1
    assert manifest["training"]["non_finite_loss_terms"] == 6
    assert manifest["training"]["optimizer_steps_skipped"] == 1
    assert [row["event"] for row in rows].count("step") == 3
    assert [row["event"] for row in rows].count("epoch") == 1
    assert [row["event"] for row in rows].count("nonfinite_loss") == 1
    assert [row["event"] for row in rows].count("optimizer_step_skipped") == 1
