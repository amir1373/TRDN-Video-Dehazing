"""Real-data, real-weight safety and cost preflight for TRDN training."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_full_test import evaluate
from src.config import TRDNConfig
from src.flow import load_raft
from src.losses import LossBundle
from src.provenance import (
    dataset_size,
    effective_mask_mode,
    loss_weights,
    peak_gpu_memory_bytes,
    runtime_environment,
    trainable_parameter_counts,
    validate_checkpoint_modes,
    write_json,
)
from src.train import (
    build_optimizer,
    build_temporal_modules,
    compute_training_loss,
    make_datasets,
    make_test_dataset_for_manifest,
)
from src.diffusion_adapter import load_diffusion_backbone


def _banner(label: str) -> None:
    border = "=" * 88
    print(f"\n{border}\nTRDN PREFLIGHT {label}\n{border}")


def _checkpoint_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _sequence_names(dataset: Any) -> set[str]:
    return {sequence["name"] for sequence in dataset.sequences}


def _require_real_dataset(dataset: Any, split: str) -> None:
    if not dataset.sequences or not dataset.index:
        raise RuntimeError(
            f"{split} dataset has no real sequences/clips at {dataset.root}. "
            "Preflight never uses synthetic fallback."
        )


def _check_reference_integrity(dataset: Any, batch_size: int, num_batches: int) -> list[Dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    hazy_roots = {
        str(path.resolve())
        for sequence in dataset.sequences
        for path in sequence["hazy_files"]
    }
    checks = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= num_batches:
            break
        frames = batch["frames"]
        target = batch["target_frame"].unsqueeze(1)
        differences = torch.abs(frames - target)
        per_input_max = differences.flatten(2).amax(dim=2)
        if torch.any(per_input_max <= 1e-4):
            raise AssertionError(
                f"Batch {batch_index} contains an input frame equal or nearly equal to target_frame."
            )

        for frame_group in batch["frame_paths"]:
            for path in frame_group:
                if str(Path(path).resolve()) not in hazy_roots:
                    raise AssertionError(f"Reference path is not a discovered hazy file: {path}")

        maximum = float(differences.max())
        print(f"Reference integrity batch {batch_index}: max |input - target| = {maximum:.8f}")
        checks.append({"batch": batch_index, "max_abs_input_target": maximum})
    if len(checks) < num_batches:
        raise RuntimeError(f"Requested {num_batches} integrity batches but only found {len(checks)}.")
    return checks


def _prepare_timing_runtime(config: TRDNConfig, train_loader: DataLoader, checkpoint: str):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    diffusion = load_diffusion_backbone(config, device=str(accelerator.device))
    temporal_memory, temporal_transformer, reference_selector, conditioning_adapter = build_temporal_modules(
        config, diffusion["unet"].config.cross_attention_dim, str(accelerator.device)
    )
    modules = {
        "unet": diffusion["unet"],
        "temporal_memory": temporal_memory,
        "temporal_transformer": temporal_transformer,
        "reference_selector": reference_selector,
        "conditioning_adapter": conditioning_adapter,
    }
    optimizer = build_optimizer(
        config,
        diffusion["unet"],
        temporal_memory,
        temporal_transformer,
        reference_selector,
        conditioning_adapter,
    )
    counts = trainable_parameter_counts(modules, optimizer)
    if temporal_transformer is not None:
        (
            diffusion["unet"],
            temporal_memory,
            temporal_transformer,
            reference_selector,
            conditioning_adapter,
            optimizer,
            train_loader,
        ) = accelerator.prepare(
            diffusion["unet"],
            temporal_memory,
            temporal_transformer,
            reference_selector,
            conditioning_adapter,
            optimizer,
            train_loader,
        )
    else:
        (
            diffusion["unet"],
            temporal_memory,
            reference_selector,
            conditioning_adapter,
            optimizer,
            train_loader,
        ) = accelerator.prepare(
            diffusion["unet"],
            temporal_memory,
            reference_selector,
            conditioning_adapter,
            optimizer,
            train_loader,
        )
    diffusion["vae"].to(accelerator.device)
    diffusion["text_encoder"].to(accelerator.device)
    accelerator.load_state(checkpoint)
    raft_model = (
        load_raft(str(accelerator.device), config.freeze_raft)
        if config.use_raft_alignment and torch.cuda.is_available()
        else None
    )
    return {
        "accelerator": accelerator,
        "diffusion": diffusion,
        "temporal_memory": temporal_memory,
        "temporal_transformer": temporal_transformer,
        "reference_selector": reference_selector,
        "conditioning_adapter": conditioning_adapter,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "loss_bundle": LossBundle(str(accelerator.device)),
        "raft_model": raft_model,
        "parameter_counts": counts,
    }


def _time_training_steps(
    runtime: Dict[str, Any],
    config: TRDNConfig,
    warmup_steps: int,
    timed_steps: int,
) -> Dict[str, Any]:
    accelerator = runtime["accelerator"]
    optimizer = runtime["optimizer"]
    loader = runtime["train_loader"]
    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    def run_step(temporal_enabled: bool) -> float:
        batch = next_batch()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with accelerator.accumulate(runtime["diffusion"]["unet"]):
            total_loss, _parts = compute_training_loss(
                accelerator,
                runtime["diffusion"],
                runtime["temporal_memory"],
                runtime["temporal_transformer"],
                runtime["reference_selector"],
                runtime["conditioning_adapter"],
                runtime["raft_model"],
                runtime["loss_bundle"],
                batch,
                config,
                temporal_loss_enabled=temporal_enabled,
            )
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    ],
                    config.max_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - started

    for _ in range(warmup_steps):
        run_step(True)
    enabled = [run_step(True) for _ in range(timed_steps)]
    for _ in range(warmup_steps):
        run_step(False)
    disabled = [run_step(False) for _ in range(timed_steps)]

    enabled_median = statistics.median(enabled)
    disabled_median = statistics.median(disabled)
    overhead = 100.0 * (enabled_median / disabled_median - 1.0)
    return {
        "warmup_steps_per_mode": warmup_steps,
        "timed_steps_per_mode": timed_steps,
        "temporal_enabled_seconds_per_step_median": enabled_median,
        "temporal_disabled_seconds_per_step_median": disabled_median,
        "temporal_loss_overhead_percent": overhead,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes(),
    }


def run_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    config = TRDNConfig(
        project_root=args.project_root,
        max_train_steps=args.max_train_steps,
        num_epochs=args.num_epochs,
        resume_from_checkpoint=args.resume_from_checkpoint,
        train_mode=args.train_mode,
        mask_mode=args.mask_mode,
        seed=args.seed,
        seq_len=args.seq_len,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        num_inference_steps=args.num_steps,
    )
    if args.dataset_root:
        config.override_dataset_root(args.dataset_root)

    resolved_mask = effective_mask_mode(config.train_mode, config.mask_mode)
    print(f"Resolved train_mode={config.train_mode!r}, mask_mode={resolved_mask!r}")
    if config.train_mode != "dehaze" or resolved_mask != "full":
        raise RuntimeError("Preflight requires train_mode='dehaze' and effective mask_mode='full'.")
    if config.resume_from_checkpoint and not args.allow_resume:
        raise RuntimeError(
            "resume_from_checkpoint must be empty for preflight unless --allow-resume is explicit."
        )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Real checkpoint directory does not exist: {checkpoint}")
    validate_checkpoint_modes(checkpoint, config, allow_mode_mismatch=False)

    train_dataset, val_dataset = make_datasets(config)
    test_dataset = make_test_dataset_for_manifest(config)
    for split, dataset in (
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
        _require_real_dataset(dataset, split)

    names = {
        "train": _sequence_names(train_dataset),
        "val": _sequence_names(val_dataset),
        "test": _sequence_names(test_dataset),
    }
    overlaps = {
        "train_val": sorted(names["train"] & names["val"]),
        "train_test": sorted(names["train"] & names["test"]),
        "val_test": sorted(names["val"] & names["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Dataset sequence overlap detected: {overlaps}")
    sizes = {
        "train": dataset_size(train_dataset),
        "val": dataset_size(val_dataset),
        "test": dataset_size(test_dataset),
    }
    print("Dataset split counts:", json.dumps(sizes, indent=2, sort_keys=True))

    integrity = _check_reference_integrity(
        train_dataset,
        batch_size=config.batch_size,
        num_batches=args.integrity_batches,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timing_runtime = _prepare_timing_runtime(config, train_loader, str(checkpoint))
    timing = _time_training_steps(
        timing_runtime,
        config,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
    )

    total_steps = (
        config.max_train_steps
        if config.max_train_steps > 0
        else config.num_epochs * len(timing_runtime["train_loader"])
    )
    timing["configured_total_steps"] = total_steps
    timing["estimated_wall_clock_hours"] = (
        timing["temporal_enabled_seconds_per_step_median"] * total_steps / 3600.0
    )
    timing["estimated_wall_clock_hours_temporal_disabled"] = (
        timing["temporal_disabled_seconds_per_step_median"] * total_steps / 3600.0
    )

    checkpoint_bytes = _checkpoint_size_bytes(checkpoint)
    periodic_saves = total_steps // max(config.checkpoint_every, 1)
    retained_step_checkpoints = min(periodic_saves, config.keep_last_n_checkpoints)
    retained_named_checkpoints = 3  # last, best_psnr, best_ssim; best_* are never pruned
    retained_count_upper_bound = retained_step_checkpoints + retained_named_checkpoints
    storage_bytes = checkpoint_bytes * retained_count_upper_bound
    storage = {
        "measured_checkpoint_path": str(checkpoint.resolve()),
        "measured_checkpoint_bytes": checkpoint_bytes,
        "retained_checkpoint_count_upper_bound": retained_count_upper_bound,
        "estimated_retained_storage_bytes": storage_bytes,
        "estimated_retained_storage_gb": storage_bytes / 1_000_000_000.0,
        "warning_over_100_gb": storage_bytes > 100_000_000_000,
    }

    eval_args = argparse.Namespace(
        checkpoint=str(checkpoint),
        crop_size=config.crop_size,
        train_mode=config.train_mode,
        mask_mode=config.mask_mode,
        diffusion_only=False,
        num_steps=config.num_inference_steps,
        seed=config.seed,
        variant="preflight_same_seed_a",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    first = evaluate(config, eval_args, device, max_clips=args.eval_clips)
    eval_args.variant = "preflight_same_seed_b"
    second = evaluate(config, eval_args, device, max_clips=args.eval_clips)
    if first["aggregate"] != second["aggregate"]:
        raise AssertionError("Same-seed real-weight evaluation metrics are not identical.")
    eval_args.seed = config.seed + 1
    eval_args.variant = "preflight_different_seed"
    different = evaluate(config, eval_args, device, max_clips=args.eval_clips)
    if first["aggregate"] == different["aggregate"]:
        raise AssertionError("Different-seed real-weight evaluation metrics did not change.")

    weights = loss_weights(config)
    active_losses = {
        name: {"weight": weight, "status": "ACTIVE" if weight != 0.0 else "INACTIVE"}
        for name, weight in weights.items()
    }
    environment = runtime_environment(config.mixed_precision)
    if torch.cuda.is_available():
        environment["gpu_total_memory_bytes"] = int(
            torch.cuda.get_device_properties(0).total_memory
        )

    report = {
        "status": "PASS",
        "resolved_modes": {"train_mode": config.train_mode, "mask_mode": resolved_mask},
        "config": config.to_dict(),
        "dataset_sizes": sizes,
        "sequence_overlaps": overlaps,
        "reference_integrity": integrity,
        "active_losses": active_losses,
        "trainable_parameters": timing_runtime["parameter_counts"],
        "environment": environment,
        "timing_and_cost": timing,
        "checkpoint_storage": storage,
        "determinism": {
            "clips": args.eval_clips,
            "same_seed": config.seed,
            "different_seed": config.seed + 1,
            "same_seed_metrics_identical": True,
            "different_seed_metrics_differ": True,
            "same_seed_aggregate": first["aggregate"],
            "different_seed_aggregate": different["aggregate"],
        },
    }
    _banner("COST ESTIMATE")
    print(json.dumps({"timing_and_cost": timing, "checkpoint_storage": storage}, indent=2))
    if storage["warning_over_100_gb"]:
        print("WARNING: estimated retained checkpoint storage exceeds 100 GB.")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Real Accelerate checkpoint for timing/eval.")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--project-root", default="/content/drive/MyDrive/TRDN_REVIDE")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--allow-resume", action="store_true")
    parser.add_argument("--train-mode", default="dehaze", choices=["dehaze", "reconstruct_synthetic"])
    parser.add_argument("--mask-mode", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mixed-precision", default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--integrity-batches", type=int, default=3)
    parser.add_argument("--eval-clips", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_path = Path(args.project_root) / "preflight_report.json"
    try:
        _banner("START")
        report = run_preflight(args)
        write_json(report_path, report)
        _banner("PASS")
        print(f"Wrote {report_path}")
    except Exception as exc:
        failure = {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}
        try:
            write_json(report_path, failure)
        except Exception:
            pass
        _banner("FAIL")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
