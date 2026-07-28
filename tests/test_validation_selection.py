import json
from contextlib import ExitStack
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import src.train as train_module
from scripts.smoke_test import _patch_pipeline, make_fake_revide_tree
from src.config import TRDNConfig
from src.validate import validate_trdn


class ValidationDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        target = torch.full((3, 8, 8), 0.25 + index * 0.01)
        return {
            "frames": target.unsqueeze(0).repeat(2, 1, 1, 1),
            "target_frame": target,
            "mask": torch.ones(1, 8, 8),
            "corrupted_frame": target + 0.05,
            "sequence_name": f"sequence_{index}",
        }


class ValidationLoss:
    @staticmethod
    def lpips_loss(prediction, target):
        return torch.abs(prediction - target).mean()


def test_validation_is_seeded_deterministic_and_counts_exact_samples(monkeypatch):
    calls = []

    def deterministic_inference(
        _frames,
        _mask,
        corrupted,
        _diffusion,
        *_args,
        seed,
        sample_ids,
        num_steps,
        **_kwargs,
    ):
        calls.append((seed, tuple(sample_ids), num_steps))
        return {"prediction": corrupted.clamp(0, 1)}

    monkeypatch.setattr(
        "src.validate.infer_dehazed_batch",
        deterministic_inference,
    )
    modules = [torch.nn.Linear(1, 1) for _ in range(4)]
    diffusion = {"unet": torch.nn.Linear(1, 1)}
    loader = DataLoader(ValidationDataset(), batch_size=2, shuffle=False)

    first = validate_trdn(
        loader,
        diffusion,
        modules[0],
        None,
        modules[1],
        modules[2],
        ValidationLoss(),
        "cpu",
        num_samples=3,
        num_steps=27,
        seed=777,
    )
    first_calls = list(calls)
    calls.clear()
    second = validate_trdn(
        loader,
        diffusion,
        modules[0],
        None,
        modules[1],
        modules[2],
        ValidationLoss(),
        "cpu",
        num_samples=3,
        num_steps=27,
        seed=777,
    )

    assert first["num_samples"] == second["num_samples"] == 3
    assert first["num_inference_steps"] == second["num_inference_steps"] == 27
    assert first["seed"] == second["seed"] == 777
    assert first["psnr"] == second["psnr"]
    assert first["ssim"] == second["ssim"]
    assert first["lpips"] == second["lpips"]
    assert first_calls == calls


def test_early_stopping_is_opt_in_and_records_selection_state(tmp_path: Path):
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
        validate_every=1,
        checkpoint_every=99,
        log_every=1,
        num_inference_steps=1,
        validation_num_samples=2,
        validation_num_inference_steps=1,
        checkpoint_selection_metric="ssim",
        enable_early_stopping=True,
        early_stopping_patience=1,
        run_name="early-stop",
    )
    config.override_dataset_root(str(dataset_root))

    with ExitStack() as stack:
        _patch_pipeline(stack)
        result = train_module.train_trdn(config)

    assert result["step"] == 2.0
    assert result["selection_metric"] == "ssim"
    assert result["selection_value"] == 0.8
    assert result["stopped_early"] is True
    manifest = json.loads(
        (
            project_root
            / "logs"
            / "runs"
            / "early-stop"
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["checkpoint_selection"]["metric"] == "ssim"
    assert manifest["checkpoint_selection"]["stopped_early"] is True
    assert len(manifest["validation_passes"]) == 2
    assert manifest["validation_passes"][-1]["num_samples"] == 2
    assert manifest["validation_passes"][-1]["num_inference_steps"] == 1
    metadata = json.loads(
        (
            project_root
            / "checkpoints"
            / "best_ssim"
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["checkpoint_selection"]["metric"] == "ssim"
    assert metadata["checkpoint_selection"]["value"] == 0.8
    assert metadata["checkpoint_selection"]["validation_num_samples"] == 2
    assert metadata["checkpoint_selection"]["validation_num_inference_steps"] == 1
