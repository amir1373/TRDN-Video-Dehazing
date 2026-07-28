"""Run the complete TRDN orchestration on CPU with tiny local stubs."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.evaluate_full_test as evaluate_script
import scripts.make_paper_figures as figures_script
import scripts.preflight as preflight_script
import src.train as train_module
from scripts.emit_paper_tables import load_reports, render_markdown, write_csv
from src.config import TRDNConfig
from src.provenance import write_json


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.5]))
        self.config = SimpleNamespace(cross_attention_dim=4)

    def forward(self, value):
        return value * self.weight


class SmokeLossBundle:
    def __init__(self, _device="cpu", **_kwargs):
        pass

    @staticmethod
    def lpips_loss(prediction, target):
        return torch.abs(prediction - target).mean()


def tiny_backbone(_config, device="cpu"):
    return {
        "unet": TinyModule().to(device),
        "vae": TinyModule().to(device),
        "text_encoder": TinyModule().to(device),
        "tokenizer": object(),
        "noise_scheduler": object(),
        "inference_scheduler": object(),
    }


def tiny_temporal(config, _cross_attention_dim, device):
    modules = [TinyModule().to(device) for _ in range(4)]
    return modules[0], (modules[1] if config.use_temporal_transformer else None), modules[2], modules[3]


def tiny_training_loss(
    _accelerator,
    diffusion,
    temporal_memory,
    temporal_transformer,
    reference_selector,
    conditioning_adapter,
    _raft_model,
    _loss_bundle,
    _batch,
    _config,
    **_kwargs,
):
    modules = [
        diffusion["unet"],
        temporal_memory,
        temporal_transformer,
        reference_selector,
        conditioning_adapter,
    ]
    total = sum(
        (parameter.square().sum() for module in modules if module is not None for parameter in module.parameters()),
        torch.tensor(0.0, device=diffusion["unet"].weight.device),
    )
    parts = {
        "diffusion": total,
        "l1": total,
        "lpips": total,
        "temporal": total,
        "flow": total,
    }
    return total, parts


def tiny_validation(*_args, **_kwargs):
    return {"psnr": 20.0, "ssim": 0.8, "lpips": 0.1, "first_output": None}


def tiny_eval_runtime(_config, _checkpoint, _device, *, diffusion_only=False, use_ema=False):
    return {
        "diffusion": {"unet": TinyModule()},
        "temporal_memory": None if diffusion_only else TinyModule(),
        "temporal_transformer": None if diffusion_only else TinyModule(),
        "reference_selector": None if diffusion_only else TinyModule(),
        "conditioning_adapter": None if diffusion_only else TinyModule(),
        "raft_model": None,
        "ema": None,
    }


def tiny_full_inference(
    frames,
    _mask,
    corrupted,
    _diffusion,
    _temporal_memory,
    _temporal_transformer,
    _reference_selector,
    _conditioning_adapter,
    _device,
    **_kwargs,
):
    batch, seq_len = frames.shape[:2]
    return {
        "prediction": (corrupted * 0.8).clamp(0, 1),
        "reference_weights": torch.full(
            (batch, seq_len - 1, 1, 1),
            1.0 / max(seq_len - 1, 1),
        ),
    }


def tiny_diffusion_inference(_mask, corrupted, _diffusion, _device, **_kwargs):
    return {"prediction": (corrupted * 0.75).clamp(0, 1)}


def _write_sequence(root: Path, name: str, frame_count: int, size: int = 16) -> None:
    clean_dir = root / name / "clean"
    hazy_dir = root / name / "hazy"
    clean_dir.mkdir(parents=True)
    hazy_dir.mkdir(parents=True)
    for index in range(frame_count):
        clean = np.full((size, size, 3), 40 + index * 10, dtype=np.uint8)
        hazy = np.clip(clean.astype(np.int16) + 35, 0, 255).astype(np.uint8)
        Image.fromarray(clean).save(clean_dir / f"{index:04d}.png")
        Image.fromarray(hazy).save(hazy_dir / f"{index:04d}.png")


def make_fake_revide_tree(root: Path) -> None:
    for index in range(4):
        _write_sequence(root / "Train", f"train_{index}", 4)
    _write_sequence(root / "Test", "test_complete", 3)
    _write_sequence(root / "Test", "test_too_short", 1)


def _assert_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"Expected non-empty output: {path}")


def _patch_pipeline(stack: ExitStack) -> None:
    replacements = (
        ("src.train.load_diffusion_backbone", tiny_backbone),
        ("src.train.build_temporal_modules", tiny_temporal),
        ("src.train.LossBundle", SmokeLossBundle),
        ("src.train.compute_training_loss", tiny_training_loss),
        ("src.train.validate_trdn", tiny_validation),
        ("scripts.preflight.load_diffusion_backbone", tiny_backbone),
        ("scripts.preflight.build_temporal_modules", tiny_temporal),
        ("scripts.preflight.LossBundle", SmokeLossBundle),
        ("scripts.preflight.compute_training_loss", tiny_training_loss),
        ("scripts.evaluate_full_test.load_runtime_for_eval", tiny_eval_runtime),
        ("scripts.evaluate_full_test.LossBundle", SmokeLossBundle),
        ("scripts.evaluate_full_test.load_raft", lambda *_args, **_kwargs: object()),
        (
            "scripts.evaluate_full_test.flow_warped_temporal_consistency_error",
            lambda previous, current, _raft: (
                float(torch.abs(previous - current).mean()),
                1.0,
            ),
        ),
        ("scripts.evaluate_full_test.infer_dehazed_batch", tiny_full_inference),
        ("scripts.evaluate_full_test.infer_diffusion_only_batch", tiny_diffusion_inference),
        ("scripts.make_paper_figures.load_runtime_for_eval", tiny_eval_runtime),
        ("scripts.make_paper_figures.infer_dehazed_batch", tiny_full_inference),
        ("scripts.make_paper_figures.infer_diffusion_only_batch", tiny_diffusion_inference),
    )
    for target, replacement in replacements:
        stack.enter_context(patch(target, replacement))


def run_smoke(work_root: Path) -> dict:
    dataset_root = work_root / "REVIDE"
    project_root = work_root / "run"
    artifact_root = project_root / "paper_artifacts"
    make_fake_revide_tree(dataset_root)

    with ExitStack() as stack:
        _patch_pipeline(stack)
        preflight_args = preflight_script.build_parser().parse_args(
            [
                "--dataset-root",
                str(dataset_root),
                "--project-root",
                str(project_root),
                "--max-train-steps",
                "4",
                "--num-epochs",
                "2",
                "--seq-len",
                "2",
                "--crop-size",
                "16",
                "--batch-size",
                "1",
                "--num-workers",
                "0",
                "--mixed-precision",
                "no",
                "--num-steps",
                "1",
                "--integrity-batches",
                "1",
                "--warmup-steps",
                "1",
                "--timed-steps",
                "1",
                "--enable-ema",
            ]
        )
        preflight_report = preflight_script.run_preflight(preflight_args)
        preflight_path = project_root / "preflight_report.json"
        write_json(preflight_path, preflight_report)

        config = TRDNConfig(
            project_root=str(project_root),
            dataset_root=str(dataset_root),
            mixed_precision="no",
            seq_len=2,
            crop_size=16,
            batch_size=1,
            num_workers=0,
            max_train_steps=2,
            num_epochs=2,
            validate_every=1,
            checkpoint_every=1,
            log_every=1,
            num_inference_steps=1,
            enable_ema=True,
            run_name="smoke",
        )
        config.override_dataset_root(str(dataset_root))
        first_result = train_module.train_trdn(config)
        checkpoint = project_root / "checkpoints" / "last"

        resume_config = TRDNConfig(**config.to_dict())
        resume_config.max_train_steps = 4
        resume_config.resume_from_checkpoint = str(checkpoint)
        resumed_result = train_module.train_trdn(resume_config)

        eval_paths = []
        for diffusion_only, name in ((False, "full"), (True, "diffusion_only")):
            eval_args = argparse.Namespace(
                checkpoint=str(checkpoint),
                crop_size=16,
                train_mode="dehaze",
                mask_mode="auto",
                diffusion_only=diffusion_only,
                num_steps=1,
                seed=1234,
                variant=name,
                use_ema=False,
            )
            report = evaluate_script.evaluate(resume_config, eval_args, "cpu")
            output = project_root / "evaluation" / f"{name}.json"
            write_json(output, report)
            evaluate_script.record_evaluation_manifest(str(checkpoint), output, report)
            eval_paths.append(output)

        figure_args = figures_script.build_parser().parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--eval-json",
                *(str(path) for path in eval_paths),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(artifact_root),
                "--seed",
                "1234",
                "--num-samples",
                "1",
            ]
        )
        figures_script.run(figure_args)

        reports = load_reports(eval_paths)
        markdown_path = artifact_root / "paper_results.md"
        csv_path = artifact_root / "paper_results.csv"
        markdown_path.write_text(render_markdown(reports), encoding="utf-8")
        write_csv(csv_path, reports)

    manifest_path = project_root / "logs" / "runs" / "smoke" / "run_manifest.json"
    metrics_path = manifest_path.parent / "metrics.jsonl"
    expected = [
        preflight_path,
        manifest_path,
        metrics_path,
        checkpoint / "ema_weights.pt",
        *eval_paths,
        artifact_root / "variant_metrics.png",
        artifact_root / "variant_metrics.sidecar.json",
        artifact_root / "temporal_window_comparison.png",
        artifact_root / "temporal_window_comparison.sidecar.json",
        artifact_root / "reference_weights.png",
        artifact_root / "reference_weights.sidecar.json",
        artifact_root / "sample_selection.json",
        artifact_root / "qualitative_grid.png",
        artifact_root / "qualitative_grid.sidecar.json",
        artifact_root / "variant_qualitative_comparison.png",
        artifact_root / "variant_qualitative_comparison.sidecar.json",
        markdown_path,
        csv_path,
    ]
    for path in expected:
        _assert_nonempty(path)
    for figure_path in artifact_root.glob("*.png"):
        _assert_nonempty(figure_path.with_suffix(".sidecar.json"))

    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    step_numbers = [row["step"] for row in metrics if row["event"] == "step"]
    if step_numbers != [1, 2, 3, 4]:
        raise AssertionError(f"Resume did not append exact step sequence: {step_numbers}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["resumes"][-1]["from_step"] != 2:
        raise AssertionError("Resume manifest did not record the source step.")
    full_report = json.loads(eval_paths[0].read_text(encoding="utf-8"))
    if full_report["clips_total_found"] != 2:
        raise AssertionError("Smoke evaluation did not discover both test sequences.")
    if full_report["clips_evaluated"] != 1 or full_report["clips_skipped"] != 1:
        raise AssertionError("Smoke evaluation clip accounting is incorrect.")
    if full_report["skipped_clips"][0]["reason"] != "too_short_for_seq_len":
        raise AssertionError("Too-short sequence was not reported with the expected reason.")

    return {
        "status": "PASS",
        "work_root": str(work_root.resolve()),
        "first_training_step": first_result["step"],
        "resumed_training_step": resumed_result["step"],
        "expected_outputs": [str(path.resolve()) for path in expected],
        "evaluation_accounting": {
            key: full_report[key]
            for key in ("clips_total_found", "clips_evaluated", "clips_skipped")
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Keep outputs here. By default a temporary directory is removed after success.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.work_root is not None:
        args.work_root.mkdir(parents=True, exist_ok=True)
        report = run_smoke(args.work_root)
    else:
        temporary = Path(tempfile.mkdtemp(prefix="trdn_smoke_"))
        try:
            report = run_smoke(temporary)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
