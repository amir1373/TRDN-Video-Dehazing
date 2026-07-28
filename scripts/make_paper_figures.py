"""Generate deterministic paper figures from full-test evaluation JSON."""

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_full_test import (
    build_test_dataset,
    group_index_by_clip,
    load_runtime_for_eval,
)
from src.config import TRDNConfig
from src.progress import ProgressReporter
from src.provenance import git_state, write_json
from src.validate import infer_dehazed_batch, infer_diffusion_only_batch


METRICS = (
    ("psnr", "PSNR"),
    ("ssim", "SSIM"),
    ("lpips", "LPIPS"),
    ("temporal_consistency_l1", "Temporal error"),
    ("runtime_seconds", "Runtime (s)"),
)
SELECTION_RULE = (
    "For the primary eval JSON, sort random.Random(seed).sample("
    "range(N_frames), min(num_samples, N_frames)); the population is its complete "
    "unfiltered test-window index."
)


def select_sample_indices(population_size: int, num_samples: int, seed: int) -> List[int]:
    if population_size <= 0:
        raise ValueError("N_frames must be positive for deterministic sample selection.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    count = min(num_samples, population_size)
    return sorted(random.Random(seed).sample(range(population_size), count))


def load_eval_reports(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    reports = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("is_full_test") is False:
            raise ValueError(f"Refusing paper figures from non-full evaluation: {path}")
        if report.get("clips_available") is not None and (
            report.get("clips_evaluated") != report.get("clips_available")
        ):
            raise ValueError(f"Refusing incomplete-coverage paper figures: {path}")
        if "aggregate" not in report or "N_frames" not in report:
            raise ValueError(f"Evaluation JSON lacks required aggregate/N_frames fields: {path}")
        reports.append(report)
    if not reports:
        raise ValueError("At least one eval JSON is required.")
    return reports


def _metric(report: Dict[str, Any], key: str) -> float | None:
    if key == "runtime_seconds":
        value = report.get(key)
    else:
        value = report["aggregate"].get(key, {}).get("mean")
    return float(value) if value is not None else None


def _variant(report: Dict[str, Any], index: int) -> str:
    return str(report.get("variant") or f"variant_{index + 1}")


def _sidecar_payload(
    reports: List[Dict[str, Any]],
    eval_paths: List[Path],
    checkpoint: Path,
    seed: int,
    sample_indices: List[int],
) -> Dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint.resolve()),
        "variant_checkpoint_paths": [
            str(Path(report.get("checkpoint_path") or checkpoint).resolve()) for report in reports
        ],
        "git_sha": git_state()["commit_sha"],
        "evaluation_git_shas": [report.get("git_commit", "unknown") for report in reports],
        "seed": seed,
        "sample_indices": sample_indices,
        "sample_selection_rule": SELECTION_RULE,
        "eval_json_path": [str(path.resolve()) for path in eval_paths],
    }


def _save_figure(
    figure: plt.Figure,
    path: Path,
    sidecar: Dict[str, Any],
) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    write_json(path.with_suffix(".sidecar.json"), sidecar)


def generate_metric_figures(
    reports: List[Dict[str, Any]],
    eval_paths: List[Path],
    checkpoint: Path,
    output_dir: Path,
    seed: int,
    sample_indices: List[int],
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_payload(reports, eval_paths, checkpoint, seed, sample_indices)
    labels = [_variant(report, index) for index, report in enumerate(reports)]
    colors = plt.get_cmap("tab10")(np.arange(len(reports)) % 10)

    figure, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.2))
    for axis, (metric, title) in zip(np.atleast_1d(axes), METRICS):
        values = [_metric(report, metric) for report in reports]
        positions = [index for index, value in enumerate(values) if value is not None]
        axis.bar(
            positions,
            [values[index] for index in positions],
            color=[colors[index] for index in positions],
        )
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Full-test metrics by variant")
    figure.tight_layout()
    metrics_path = output_dir / "variant_metrics.png"
    _save_figure(figure, metrics_path, sidecar)

    comparison_metrics = METRICS[:4]
    figure, axes = plt.subplots(1, len(comparison_metrics), figsize=(4.5 * len(comparison_metrics), 4.2))
    for axis, (metric, title) in zip(np.atleast_1d(axes), comparison_metrics):
        points = [
            (int(report.get("seq_len", 0)), _metric(report, metric), _variant(report, index))
            for index, report in enumerate(reports)
        ]
        points = [point for point in points if point[0] > 0 and point[1] is not None]
        points.sort(key=lambda point: (point[0], point[2]))
        for position, (window, value, label) in enumerate(points):
            axis.scatter(window, value, color=colors[position % len(colors)], label=label)
        axis.set_title(title)
        axis.set_xlabel("Temporal window length")
        axis.grid(alpha=0.25)
        if points:
            axis.legend(fontsize=8)
    figure.suptitle("Temporal-window comparison")
    figure.tight_layout()
    temporal_path = output_dir / "temporal_window_comparison.png"
    _save_figure(figure, temporal_path, sidecar)
    return [metrics_path, temporal_path]


def generate_reference_weight_artifacts(
    reports: List[Dict[str, Any]],
    eval_paths: List[Path],
    checkpoint: Path,
    output_dir: Path,
    seed: int,
    sample_indices: List[int],
) -> List[Path]:
    usable = [
        (index, report)
        for index, report in enumerate(reports)
        if any(item.get("mean") is not None for item in report.get("reference_weights_by_offset", []))
    ]
    if not usable:
        return []
    raw = {
        "source": "reference_weights_by_offset fields from eval JSON; no recomputation",
        "variants": [
            {
                "variant": _variant(report, index),
                "checkpoint_path": report.get("checkpoint_path"),
                "weights": report["reference_weights_by_offset"],
            }
            for index, report in usable
        ],
    }
    raw_path = output_dir / "reference_weights.json"
    write_json(raw_path, raw)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for index, report in usable:
        rows = report["reference_weights_by_offset"]
        offsets = [row["offset"] for row in rows]
        means = [row["mean"] for row in rows]
        stds = [row["std"] for row in rows]
        axis.errorbar(offsets, means, yerr=stds, marker="o", capsize=3, label=_variant(report, index))
    axis.set_xlabel("Reference offset from current frame")
    axis.set_ylabel("Mean reference weight")
    axis.set_title("Reference selection weights over the full test set")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure_path = output_dir / "reference_weights.png"
    sidecar = _sidecar_payload(reports, eval_paths, checkpoint, seed, sample_indices)
    sidecar["raw_reference_weights_path"] = raw_path.name
    _save_figure(figure, figure_path, sidecar)
    return [figure_path, raw_path]


def _tensor_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().clamp(0, 1).numpy()
    if array.ndim == 4:
        array = array[0]
    return np.transpose(array, (1, 2, 0))


def _sample_ids(dataset: Any) -> Dict[int, str]:
    by_clip = group_index_by_clip(dataset)
    result = {}
    for seq_idx, indices in by_clip.items():
        clip_name = dataset.sequences[seq_idx]["name"]
        for position, dataset_index in enumerate(indices):
            result[dataset_index] = f"{clip_name}:{position}"
    return result


@torch.no_grad()
def _predict_selected(
    report: Dict[str, Any],
    checkpoint: Path,
    dataset_root: str,
    sample_indices: List[int],
    seed: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    config = TRDNConfig(
        project_root=str(checkpoint.parent),
        seq_len=int(report.get("seq_len", 10)),
        crop_size=int(report.get("crop_size", 256)),
        train_mode=str(report.get("train_mode", "dehaze")),
        mask_mode=str(report.get("mask_mode", "full")),
        mixed_precision=str(
            report.get("numerics", {}).get(
                "mixed_precision",
                "fp16" if torch.cuda.is_available() else "no",
            )
        ),
        guidance_scale=float(report.get("guidance_scale", 1.0)),
        text_prompt=str(
            report.get("text_prompt", "a clear clean dehazed video frame")
        ),
    )
    for key in (
        "allow_tf32",
        "cudnn_benchmark",
        "attention_backend",
        "batch_size",
        "enable_unet_gradient_checkpointing",
        "enable_torch_compile",
        "channels_last",
    ):
        if key in report.get("numerics", {}):
            setattr(config, key, report["numerics"][key])
    config.enable_xformers_if_available = config.attention_backend == "xformers"
    if dataset_root:
        config.override_dataset_root(dataset_root)
    else:
        config.dataset_root = str(report.get("dataset_root", config.dataset_root))
        config.test_root = str(report.get("test_root", config.test_root))
    args = SimpleNamespace(
        train_mode=config.train_mode,
        mask_mode=config.mask_mode,
        crop_size=config.crop_size,
    )
    dataset = build_test_dataset(config, args)
    if not dataset.index:
        raise RuntimeError(f"No real test data found for qualitative figures at {config.test_root}")
    if max(sample_indices) >= len(dataset):
        raise ValueError(
            f"Selected index {max(sample_indices)} exceeds variant dataset length {len(dataset)}."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion_only = bool(report.get("diffusion_only_baseline", False))
    runtime = load_runtime_for_eval(
        config,
        str(checkpoint),
        device,
        diffusion_only=diffusion_only,
        use_ema=report.get("ema_weights") is not None,
    )
    identifiers = _sample_ids(dataset)
    outputs = {}
    progress = ProgressReporter(
        len(sample_indices),
        f"Qualitative inference ({report.get('variant', 'variant')})",
        leave=False,
        position=1,
    )
    for dataset_index in sample_indices:
        sample = dataset[dataset_index]
        batch = {
            key: (value.unsqueeze(0).to(device) if torch.is_tensor(value) else [value])
            for key, value in sample.items()
        }
        sample_id = [identifiers[dataset_index]]
        if report.get("diffusion_only_baseline", False):
            prediction = infer_diffusion_only_batch(
                batch["mask"],
                batch["corrupted_frame"],
                runtime["diffusion"],
                device,
                num_steps=int(report["num_inference_steps"]),
                seed=seed,
                sample_ids=sample_id,
                show_progress=False,
                text_prompt=config.text_prompt,
                guidance_scale=config.guidance_scale,
            )["prediction"]
        else:
            prediction = infer_dehazed_batch(
                batch["frames"],
                batch["mask"],
                batch["corrupted_frame"],
                runtime["diffusion"],
                runtime["temporal_memory"],
                runtime["temporal_transformer"],
                runtime["reference_selector"],
                runtime["conditioning_adapter"],
                device,
                raft_model=runtime["model_raft"],
                num_steps=int(report["num_inference_steps"]),
                seed=seed,
                sample_ids=sample_id,
                show_progress=False,
                text_prompt=config.text_prompt,
                guidance_scale=config.guidance_scale,
            )["prediction"]
        outputs[dataset_index] = {
            "hazy": batch["corrupted_frame"].cpu(),
            "prediction": prediction.cpu(),
            "target": batch["target_frame"].cpu(),
        }
        progress.update(1)
    progress.close()
    del runtime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def generate_qualitative_figures(
    reports: List[Dict[str, Any]],
    eval_paths: List[Path],
    checkpoint: Path,
    dataset_root: str,
    output_dir: Path,
    seed: int,
    sample_indices: List[int],
) -> List[Path]:
    population_sizes = {int(report["N_frames"]) for report in reports}
    if len(population_sizes) != 1:
        raise ValueError(
            "Qualitative variant comparison requires matching N_frames. "
            "Generate temporal-window metric charts with --metrics-only when window sizes differ."
        )
    variant_outputs = []
    checkpoints = []
    for index, report in enumerate(reports):
        variant_checkpoint = checkpoint if index == 0 else Path(report.get("checkpoint_path", ""))
        if not variant_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Checkpoint for variant {_variant(report, index)!r} does not exist: {variant_checkpoint}"
            )
        checkpoints.append(variant_checkpoint)
        variant_outputs.append(
            _predict_selected(report, variant_checkpoint, dataset_root, sample_indices, seed)
        )

    sidecar = _sidecar_payload(reports, eval_paths, checkpoint, seed, sample_indices)
    primary = variant_outputs[0]
    figure, axes = plt.subplots(len(sample_indices), 3, figsize=(12, 4 * len(sample_indices)), squeeze=False)
    for row, sample_index in enumerate(sample_indices):
        for column, key in enumerate(("hazy", "prediction", "target")):
            axes[row, column].imshow(_tensor_image(primary[sample_index][key]))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(("Hazy input", "Prediction", "Ground truth")[column])
        axes[row, 0].set_ylabel(f"Index {sample_index}")
    figure.tight_layout()
    qualitative_path = output_dir / "qualitative_grid.png"
    _save_figure(figure, qualitative_path, sidecar)

    columns = 2 + len(reports)
    figure, axes = plt.subplots(
        len(sample_indices),
        columns,
        figsize=(4 * columns, 4 * len(sample_indices)),
        squeeze=False,
    )
    titles = ["Hazy input"] + [_variant(report, index) for index, report in enumerate(reports)] + ["Ground truth"]
    for row, sample_index in enumerate(sample_indices):
        images = (
            [variant_outputs[0][sample_index]["hazy"]]
            + [outputs[sample_index]["prediction"] for outputs in variant_outputs]
            + [variant_outputs[0][sample_index]["target"]]
        )
        for column, image in enumerate(images):
            axes[row, column].imshow(_tensor_image(image))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column])
        axes[row, 0].set_ylabel(f"Index {sample_index}")
    figure.tight_layout()
    comparison_path = output_dir / "variant_qualitative_comparison.png"
    _save_figure(figure, comparison_path, sidecar)
    return [qualitative_path, comparison_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Generate JSON-sourced metric/reference figures without loading image/model data.",
    )
    return parser


def run(args: argparse.Namespace) -> List[Path]:
    reports = load_eval_reports(args.eval_json)
    seed = int(reports[0].get("seed", 1234) if args.seed is None else args.seed)
    sample_indices = select_sample_indices(int(reports[0]["N_frames"]), args.num_samples, seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "seed": seed,
        "sample_indices": sample_indices,
        "population": "complete unfiltered test-window index of the primary eval JSON",
        "selection_rule": SELECTION_RULE,
    }
    write_json(args.output_dir / "sample_selection.json", selection)

    progress = ProgressReporter(
        2 if args.metrics_only else 3,
        "Paper figure generation",
        leave=True,
    )
    generated = generate_metric_figures(
        reports,
        args.eval_json,
        args.checkpoint,
        args.output_dir,
        seed,
        sample_indices,
    )
    progress.update(1)
    generated.extend(
        generate_reference_weight_artifacts(
            reports,
            args.eval_json,
            args.checkpoint,
            args.output_dir,
            seed,
            sample_indices,
        )
    )
    progress.update(1)
    if not args.metrics_only:
        generated.extend(
            generate_qualitative_figures(
                reports,
                args.eval_json,
                args.checkpoint,
                args.dataset_root,
                args.output_dir,
                seed,
                sample_indices,
            )
        )
        progress.update(1)
    progress.close()
    print(json.dumps(selection, indent=2))
    for path in generated:
        progress.write(f"Wrote {path}")
    return generated


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
