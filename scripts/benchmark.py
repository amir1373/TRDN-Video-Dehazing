"""Benchmark TRDN execution options on a real CUDA GPU and real REVIDE data.

This script never estimates performance. Each report row is measured from an
independently constructed runtime using the baseline plus the named overrides.
"""

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preflight import _prepare_timing_runtime, _require_real_dataset
from src.config import TRDNConfig
from src.provenance import write_json
from src.progress import ProgressReporter
from src.train import compute_training_loss, make_datasets, nonfinite_loss_terms


def _make_loader(config: TRDNConfig) -> DataLoader:
    train_dataset, _val_dataset = make_datasets(config)
    _require_real_dataset(train_dataset, "train")
    worker_options = (
        {
            "persistent_workers": config.persistent_workers,
            "prefetch_factor": config.prefetch_factor,
        }
        if config.num_workers > 0
        else {}
    )
    return DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        **worker_options,
    )


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _loss_signature(parts: Dict[str, torch.Tensor], total: torch.Tensor) -> Dict[str, float]:
    return {
        **{name: float(value.detach().float().cpu()) for name, value in parts.items()},
        "total": float(total.detach().float().cpu()),
    }


def _numerics_changed(
    baseline: Dict[str, float] | None,
    current: Dict[str, float],
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> bool | None:
    if baseline is None:
        return None
    shared = sorted(set(baseline) & set(current))
    return any(
        not torch.isclose(
            torch.tensor(baseline[key]),
            torch.tensor(current[key]),
            rtol=rtol,
            atol=atol,
        ).item()
        for key in shared
    )


def _measure_variant(
    name: str,
    category: str,
    config: TRDNConfig,
    checkpoint: str,
    warmup_steps: int,
    timed_steps: int,
    baseline_signature: Dict[str, float] | None,
) -> Dict[str, Any]:
    set_seed(config.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loader = _make_loader(config)
    runtime = _prepare_timing_runtime(config, loader, checkpoint)
    accelerator = runtime["accelerator"]
    optimizer = runtime["optimizer"]
    prepared_loader = runtime["train_loader"]
    iterator = iter(prepared_loader)
    total_durations = []
    gpu_durations = []
    data_wait_durations = []
    raft_seconds = 0.0
    overflow_skips = 0
    signature = None
    warmup_started = time.perf_counter()
    progress = ProgressReporter(
        warmup_steps + timed_steps,
        f"Benchmark {name}",
        leave=False,
        position=1,
    )

    for step_index in range(warmup_steps + timed_steps):
        step_started = time.perf_counter()
        data_started = time.perf_counter()
        batch, iterator = _next_batch(iterator, prepared_loader)
        data_wait = time.perf_counter() - data_started
        torch.cuda.synchronize()
        gpu_started = time.perf_counter()
        timing: Dict[str, float] = {}
        with accelerator.accumulate(runtime["diffusion"]["unet"]):
            total_loss, parts = compute_training_loss(
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
                timing=timing,
            )
            bad_terms = nonfinite_loss_terms(parts, total_loss)
            if bad_terms:
                raise FloatingPointError(
                    f"Benchmark variant {name!r} produced non-finite losses: {bad_terms}"
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
            if accelerator.sync_gradients and accelerator.optimizer_step_was_skipped:
                overflow_skips += 1
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        gpu_elapsed = time.perf_counter() - gpu_started
        total_elapsed = time.perf_counter() - step_started
        if step_index == warmup_steps:
            signature = _loss_signature(parts, total_loss)
        if step_index >= warmup_steps:
            total_durations.append(total_elapsed)
            gpu_durations.append(gpu_elapsed)
            data_wait_durations.append(data_wait)
            raft_seconds += timing.get("raft_seconds", 0.0)
        progress.set_postfix(
            {"phase": "warmup" if step_index < warmup_steps else "steady"}
        )
        progress.update(1)
    progress.close()

    warmup_seconds = time.perf_counter() - warmup_started - sum(total_durations)
    median_step = statistics.median(total_durations)
    median_gpu = statistics.median(gpu_durations)
    median_data_wait = statistics.median(data_wait_durations)
    peak_bytes = int(torch.cuda.max_memory_allocated())
    result = {
        "name": name,
        "category": category,
        "status": "measured",
        "overrides": {
            "mixed_precision": config.mixed_precision,
            "allow_tf32": config.allow_tf32,
            "cudnn_benchmark": config.cudnn_benchmark,
            "attention_backend": config.attention_backend,
            "batch_size": config.batch_size,
            "gradient_checkpointing": config.enable_unet_gradient_checkpointing,
            "torch_compile": config.enable_torch_compile,
            "channels_last": config.channels_last,
            "num_workers": config.num_workers,
        },
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "warmup_seconds": warmup_seconds,
        "median_seconds_per_step": median_step,
        "median_gpu_seconds_per_step": median_gpu,
        "median_data_wait_seconds": median_data_wait,
        "data_wait_fraction": median_data_wait / max(median_step, 1e-12),
        "dataloader_bound": median_data_wait / max(median_step, 1e-12) >= 0.20,
        "samples_per_second": config.batch_size / median_step,
        "peak_memory_bytes": peak_bytes,
        "peak_memory_gb": peak_bytes / 1_000_000_000.0,
        "raft_seconds_total": raft_seconds,
        "raft_fraction_of_gpu_step": raft_seconds / max(sum(gpu_durations), 1e-12),
        "optimizer_steps_skipped": overflow_skips,
        "loss_signature": signature,
        "numerics_changed": _numerics_changed(baseline_signature, signature),
    }
    accelerator.end_training()
    del runtime, loader, prepared_loader, iterator
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _variant_matrix(baseline: TRDNConfig, max_batch_size: int, worker_values: Iterable[int]):
    yield "baseline", "baseline", {}
    yield "gradient_checkpointing_off", "gradient_checkpointing", {
        "enable_unet_gradient_checkpointing": False
    }
    batch_sizes = []
    size = 1
    while size <= max_batch_size:
        batch_sizes.append(size)
        size *= 2
    for batch_size in batch_sizes:
        if batch_size != baseline.batch_size:
            yield f"batch_size_{batch_size}", "batch_size", {"batch_size": batch_size}
    for checkpointing in (True, False):
        for batch_size in batch_sizes:
            if checkpointing == baseline.enable_unet_gradient_checkpointing and batch_size == baseline.batch_size:
                continue
            yield (
                f"combined_gc_{'on' if checkpointing else 'off'}_batch_{batch_size}",
                "batch_checkpointing_combined",
                {
                    "enable_unet_gradient_checkpointing": checkpointing,
                    "batch_size": batch_size,
                },
            )
    yield "precision_bf16", "precision", {"mixed_precision": "bf16"}
    yield "tf32_on", "tf32", {"allow_tf32": True}
    yield "cudnn_benchmark_on", "cudnn_benchmark", {"cudnn_benchmark": True}
    yield "channels_last_on", "channels_last", {"channels_last": True}
    yield "attention_sdpa", "attention_backend", {
        "attention_backend": "sdpa",
        "enable_xformers_if_available": False,
    }
    yield "torch_compile_on", "torch_compile", {"enable_torch_compile": True}
    for workers in worker_values:
        if workers != baseline.num_workers:
            yield f"num_workers_{workers}", "num_workers", {
                "num_workers": workers,
            }


def _print_table(rows: list[Dict[str, Any]]) -> None:
    columns = (
        ("Option", 40),
        ("sec/step", 12),
        ("peak GB", 10),
        ("samples/s", 12),
        ("speedup", 10),
        ("numerics", 10),
        ("status", 16),
    )
    print(" ".join(title.ljust(width) for title, width in columns))
    print("-" * sum(width for _title, width in columns))
    for row in rows:
        if row["status"] == "measured":
            values = (
                row["name"],
                f"{row['median_seconds_per_step']:.6f}",
                f"{row['peak_memory_gb']:.3f}",
                f"{row['samples_per_second']:.6f}",
                f"{row['speedup_vs_baseline']:.4f}x",
                str(row["numerics_changed"]),
                row["status"],
            )
        else:
            values = (row["name"], "-", "-", "-", "-", "-", row["status"])
        print(" ".join(str(value).ljust(width) for value, (_title, width) in zip(values, columns)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/workspace/datasets/REVIDE")
    parser.add_argument("--project-root", default="/workspace/TRDN_runs")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", type=Path, default=Path("benchmark_report.json"))
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Sweep powers of two through this limit; variants that do not fit are reported as OOM.",
    )
    parser.add_argument("--num-workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "BENCHMARK FAIL: CUDA is unavailable. Run this script on the rented A40; "
            "no report was produced.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    baseline = TRDNConfig(project_root=args.project_root, seed=args.seed)
    baseline.override_dataset_root(args.dataset_root)
    rows = []
    baseline_signature = None
    baseline_seconds = None
    seen = set()
    variants = list(
        _variant_matrix(
            baseline,
            args.max_batch_size,
            args.num_workers,
        )
    )
    progress = ProgressReporter(len(variants), "A40 benchmark variants", leave=True)
    for name, category, overrides in variants:
        signature_key = tuple(sorted(overrides.items()))
        if signature_key in seen:
            progress.update(1)
            continue
        seen.add(signature_key)
        config = replace(baseline, **overrides)
        try:
            row = _measure_variant(
                name,
                category,
                config,
                args.checkpoint,
                args.warmup_steps,
                args.timed_steps,
                baseline_signature,
            )
            if name == "baseline":
                baseline_signature = row["loss_signature"]
                baseline_seconds = row["median_seconds_per_step"]
                row["numerics_changed"] = False
            row["speedup_vs_baseline"] = baseline_seconds / row["median_seconds_per_step"]
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            row = {
                "name": name,
                "category": category,
                "status": "out_of_memory",
                "error": str(exc),
                "overrides": overrides,
            }
        except Exception as exc:
            if name == "baseline":
                raise
            row = {
                "name": name,
                "category": category,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "overrides": overrides,
            }
        rows.append(row)
        progress.set_postfix({"variant": name, "status": row["status"]})
        progress.update(1)
    progress.close()

    combined = [
        row
        for row in rows
        if row["category"] == "batch_checkpointing_combined" and row["status"] == "measured"
    ]
    best_combined = max(combined, key=lambda row: row["samples_per_second"]) if combined else None
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "baseline": baseline.__dict__,
        "rows": rows,
        "best_batch_checkpointing_combination": best_combined,
    }
    write_json(args.output, report)
    _print_table(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
