"""Full REVIDE test-set evaluation for TRDN.

Task A (default): evaluate the full temporal TRDN pipeline (RAFT + ConvLSTM +
temporal transformer + reference selector + SD inpainting) on every clip in
the test set. No sample filtering, scoring, or "good candidate" selection is
performed -- every window in every test sequence is scored and included.

Task B (--diffusion-only): evaluate a baseline that runs only the SD
inpainting backbone frame-by-frame, with no RAFT/ConvLSTM/transformer/
reference selector, to isolate the temporal stack's contribution. Reuses this
same script and JSON schema so results are directly comparable.

Both paths are fully seeded and deterministic (see src/seeding.py,
src/validate.py): DDIM sampling uses eta=0, and each frame's initial latent
noise is derived from (seed, clip name, frame index).

Test data (config.test_root) must never be used for checkpoint selection --
that is what src/dataset.py's val split (a held-out subset of TRAINING
sequences) is for. This script is the only place test data should be read.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TRDNConfig
from src.dataset import REVIDESequenceDataset
from src.diffusion_adapter import load_diffusion_backbone
from src.ema import load_ema_weights
from src.flow import flow_warped_temporal_consistency_error, load_raft
from src.losses import LossBundle
from src.metrics import psnr_metric, ssim_metric
from src.presets import apply_numerics_preset
from src.progress import ProgressReporter
from src.provenance import (
    append_evaluation_to_manifest,
    load_checkpoint_metadata,
    peak_gpu_memory_bytes,
    numerics_settings,
    validate_checkpoint_modes,
    write_json,
)
from src.train import build_optimizer, build_temporal_modules
from src.validate import infer_dehazed_batch, infer_diffusion_only_batch

def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _load_unet_only_checkpoint(unet: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    from accelerate.utils import load

    checkpoint = Path(checkpoint_path)
    candidates = (
        checkpoint / "model.safetensors",
        checkpoint / "pytorch_model.bin",
        checkpoint / "model.bin",
    )
    model_path = next((path for path in candidates if path.is_file()), None)
    if model_path is None:
        raise FileNotFoundError(
            "Diffusion-only evaluation could not find the first Accelerate model state "
            f"in {checkpoint}. Expected one of: {', '.join(path.name for path in candidates)}"
        )
    state = load(str(model_path), map_location=device)
    unet.load_state_dict(state, strict=True)


def load_runtime_for_eval(
    config: TRDNConfig,
    checkpoint_path: str,
    device: str,
    *,
    diffusion_only: bool = False,
    use_ema: bool = False,
) -> Dict[str, Any]:
    if use_ema and not checkpoint_path:
        raise ValueError("EMA evaluation requires a checkpoint path.")
    if checkpoint_path:
        validate_checkpoint_modes(checkpoint_path, config)
    diffusion = load_diffusion_backbone(config, device=device)
    temporal_memory = temporal_transformer = reference_selector = conditioning_adapter = None
    if diffusion_only:
        if checkpoint_path:
            _load_unet_only_checkpoint(diffusion["unet"], checkpoint_path, device)
    else:
        temporal_memory, temporal_transformer, reference_selector, conditioning_adapter = build_temporal_modules(
            config, diffusion["unet"].config.cross_attention_dim, device
        )
    if checkpoint_path and not diffusion_only:
        from accelerate import Accelerator

        accelerator = Accelerator(mixed_precision=config.mixed_precision)
        optimizer = build_optimizer(config, diffusion["unet"], temporal_memory, temporal_transformer, reference_selector, conditioning_adapter)
        if temporal_transformer is not None:
            diffusion["unet"], temporal_memory, temporal_transformer, reference_selector, conditioning_adapter, optimizer = accelerator.prepare(
                diffusion["unet"], temporal_memory, temporal_transformer, reference_selector, conditioning_adapter, optimizer
            )
        else:
            diffusion["unet"], temporal_memory, reference_selector, conditioning_adapter, optimizer = accelerator.prepare(
                diffusion["unet"], temporal_memory, reference_selector, conditioning_adapter, optimizer
            )
        accelerator.load_state(checkpoint_path)
    ema_report = None
    if use_ema:
        ema_modules = {
            "unet": diffusion["unet"],
            "temporal_memory": temporal_memory,
            "temporal_transformer": temporal_transformer,
            "reference_selector": reference_selector,
            "conditioning_adapter": conditioning_adapter,
        }
        if checkpoint_path and not diffusion_only:
            ema_modules = {
                name: (
                    accelerator.unwrap_model(module)
                    if module is not None
                    else None
                )
                for name, module in ema_modules.items()
            }
        ema_report = load_ema_weights(
            ema_modules,
            Path(checkpoint_path) / "ema_weights.pt",
            allow_module_subset=diffusion_only,
        )
    model_raft = (
        load_raft(
            device,
            config.freeze_raft,
            config.validate_raft_flow,
            config.raft_max_flow_factor,
        )
        if torch.cuda.is_available() and not diffusion_only
        else None
    )
    return {
        "diffusion": diffusion,
        "temporal_memory": temporal_memory,
        "temporal_transformer": temporal_transformer,
        "reference_selector": reference_selector,
        "conditioning_adapter": conditioning_adapter,
        "model_raft": model_raft,
        "ema": ema_report,
    }


def build_test_dataset(config: TRDNConfig, args: argparse.Namespace) -> REVIDESequenceDataset:
    return REVIDESequenceDataset(
        config.root_for_split("test"),
        split="test",
        seq_len=config.seq_len,
        crop_size=args.crop_size,
        random_crop=False,
        extensions=config.image_extensions,
        synthetic_if_empty=False,
        train_mode=args.train_mode,
        mask_mode=args.mask_mode,
        include_prev_frame=False,
    )


def group_index_by_clip(dataset: REVIDESequenceDataset) -> Dict[int, List[int]]:
    """Map sequence index -> list of dataset indices, in ascending frame order.

    Iterates the WHOLE dataset index with no filtering of any kind, exactly as
    required for a "clean full-test evaluation": every window in every test
    clip is included.
    """
    by_clip: Dict[int, List[int]] = defaultdict(list)
    for dataset_idx, (seq_idx, _end_idx) in enumerate(dataset.index):
        by_clip[seq_idx].append(dataset_idx)
    return by_clip


@torch.no_grad()
def evaluate(
    config: TRDNConfig,
    args: argparse.Namespace,
    device: str,
    max_clips: int | None = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dataset = build_test_dataset(config, args)
    if not dataset.sequences:
        raise RuntimeError(
            f"No real REVIDE test sequences found under {config.root_for_split('test')!r}. "
            "Full-test evaluation requires real test data; refusing to silently fall back to "
            "synthetic placeholder clips (that would not be a real evaluation)."
        )
    if not dataset.index:
        raise RuntimeError(
            "Every discovered test sequence is ineligible for evaluation: "
            + json.dumps(dataset.skipped_sequences, sort_keys=True)
        )
    runtime = load_runtime_for_eval(
        config,
        args.checkpoint,
        device,
        diffusion_only=bool(args.diffusion_only),
        use_ema=bool(getattr(args, "use_ema", False)),
    )
    loss_bundle = LossBundle(device)

    by_clip = group_index_by_clip(dataset)
    clips_total_found = len(dataset.sequences)
    clips_available = len(by_clip)
    skipped_clips: List[Dict[str, Any]] = [dict(item) for item in dataset.skipped_sequences]
    # Metric RAFT is evaluator-owned and deliberately independent of the model
    # path. Even the frame-by-frame diffusion baseline needs the same
    # motion-compensated temporal metric as every temporal model/ablation.
    consistency_raft = load_raft(
        device,
        True,
        config.validate_raft_flow,
        config.raft_max_flow_factor,
    )

    per_clip_results: List[Dict[str, Any]] = []
    all_psnr: List[float] = []
    all_ssim: List[float] = []
    all_lpips: List[float] = []
    all_temporal_error: List[float] = []
    all_temporal_coverage: List[float] = []
    total_frames = 0

    evaluated_seq_indices = sorted(by_clip)
    if max_clips is not None:
        omitted = evaluated_seq_indices[max_clips:]
        skipped_clips.extend(
            {
                "sequence_name": dataset.sequences[seq_idx]["name"],
                "reason": "debug_clip_limit",
            }
            for seq_idx in omitted
        )
        evaluated_seq_indices = evaluated_seq_indices[:max_clips]

    reference_weight_sum = np.zeros(config.seq_len - 1, dtype=np.float64)
    reference_weight_sum_squares = np.zeros(config.seq_len - 1, dtype=np.float64)
    reference_weight_count = np.zeros(config.seq_len - 1, dtype=np.int64)

    progress = ProgressReporter(
        len(evaluated_seq_indices),
        "Full-test evaluation" if max_clips is None else "DEBUG partial evaluation",
        leave=True,
        position=0,
    )
    for seq_idx in evaluated_seq_indices:
        clip_name = dataset.sequences[seq_idx]["name"]
        dataset_indices = by_clip[seq_idx]  # already ascending by construction of dataset.index
        clip_psnr, clip_ssim, clip_lpips, clip_temporal_error, clip_temporal_coverage = [], [], [], [], []
        prev_prediction: Optional[torch.Tensor] = None
        clip_weight_rows = []
        try:
            for frame_position, dataset_idx in enumerate(dataset_indices):
                sample = dataset[dataset_idx]
                batch = {
                    key: (value.unsqueeze(0).to(device) if torch.is_tensor(value) else [value])
                    for key, value in sample.items()
                }
                sample_ids = [f"{clip_name}:{frame_position}"]

                if args.diffusion_only:
                    output = infer_diffusion_only_batch(
                        batch["mask"],
                        batch["corrupted_frame"],
                        runtime["diffusion"],
                        device,
                        num_steps=args.num_steps,
                        seed=args.seed,
                        sample_ids=sample_ids,
                        show_progress=False,
                        text_prompt=config.text_prompt,
                        guidance_scale=config.guidance_scale,
                    )
                else:
                    output = infer_dehazed_batch(
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
                        num_steps=args.num_steps,
                        seed=args.seed,
                        sample_ids=sample_ids,
                        show_progress=False,
                        text_prompt=config.text_prompt,
                        guidance_scale=config.guidance_scale,
                    )

                prediction = output["prediction"]
                target = batch["target_frame"]
                clip_psnr.append(psnr_metric(prediction[0], target[0]))
                clip_ssim.append(ssim_metric(prediction[0], target[0]))
                clip_lpips.append(float(loss_bundle.lpips_loss(prediction, target).detach().cpu()))

                if prev_prediction is not None and consistency_raft is not None:
                    temporal_error, coverage = flow_warped_temporal_consistency_error(
                        prev_prediction,
                        prediction,
                        consistency_raft,
                    )
                    clip_temporal_error.append(temporal_error)
                    clip_temporal_coverage.append(coverage)
                prev_prediction = prediction

                weights = output.get("reference_weights")
                if weights is not None:
                    clip_weight_rows.append(weights.detach().float().cpu().numpy())
        except Exception as exc:
            skipped_clips.append(
                {
                    "sequence_name": clip_name,
                    "reason": "decode_or_evaluation_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            progress.write(
                f"SKIPPED clip={clip_name} reason={type(exc).__name__}: {exc}"
            )
            progress.update(1)
            continue

        clip_result = {
            "sequence_name": clip_name,
            "num_frames": len(dataset_indices),
            "psnr_mean": float(np.mean(clip_psnr)),
            "ssim_mean": float(np.mean(clip_ssim)),
            "lpips_mean": float(np.mean(clip_lpips)),
            "temporal_consistency_l1_mean": float(np.mean(clip_temporal_error)) if clip_temporal_error else None,
            "temporal_consistency_coverage_mean": float(np.mean(clip_temporal_coverage)) if clip_temporal_coverage else None,
        }
        per_clip_results.append(clip_result)
        all_psnr.extend(clip_psnr)
        all_ssim.extend(clip_ssim)
        all_lpips.extend(clip_lpips)
        all_temporal_error.extend(clip_temporal_error)
        all_temporal_coverage.extend(clip_temporal_coverage)
        total_frames += len(dataset_indices)
        for weights_np in clip_weight_rows:
            reference_weight_sum += weights_np.sum(axis=(0, 2, 3))
            reference_weight_sum_squares += np.square(weights_np).sum(axis=(0, 2, 3))
            reference_weight_count += weights_np.shape[0] * weights_np.shape[2] * weights_np.shape[3]
        progress.set_postfix({"mean_psnr": f"{np.mean(all_psnr):.3f}"})
        progress.update(1)
    progress.close()

    clips_evaluated = len(per_clip_results)
    clips_skipped = len(skipped_clips)
    if clips_evaluated + clips_skipped != clips_total_found:
        raise AssertionError(
            "Evaluation clip accounting mismatch: "
            f"found={clips_total_found} evaluated={clips_evaluated} skipped={clips_skipped}"
        )
    if max_clips is None and clips_evaluated != clips_available:
        raise AssertionError(
            "Full evaluation did not evaluate every eligible clip: "
            f"available={clips_available} evaluated={clips_evaluated}. "
            f"Skipped details: {json.dumps(skipped_clips, sort_keys=True)}"
        )

    def _mean_std(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"mean": None, "std": None}
        arr = np.asarray(values, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    reference_weights = []
    for index in range(config.seq_len - 1):
        count = int(reference_weight_count[index])
        if count:
            mean = reference_weight_sum[index] / count
            variance = max(reference_weight_sum_squares[index] / count - mean * mean, 0.0)
            std = float(np.sqrt(variance))
            mean_value: Optional[float] = float(mean)
        else:
            mean_value, std = None, None
        reference_weights.append(
            {
                "offset": index - (config.seq_len - 1),
                "mean": mean_value,
                "std": std,
                "count": count,
            }
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    try:
        checkpoint_metadata = load_checkpoint_metadata(args.checkpoint) if args.checkpoint else {}
    except FileNotFoundError:
        checkpoint_metadata = {}
    return {
        "schema_version": 2,
        "variant": getattr(args, "variant", "") or (
            "diffusion_only" if args.diffusion_only else f"trdn_t{config.seq_len}"
        ),
        "checkpoint_path": args.checkpoint,
        "checkpoint_git_sha": checkpoint_metadata.get("git_commit_sha", "unknown"),
        "train_mode": dataset.train_mode,
        "mask_mode": dataset._resolve_mask_mode(),
        "diffusion_only_baseline": args.diffusion_only,
        "temporal_metric": {
            "name": "raft_flow_warped_prediction_l1",
            "metric_raft_is_independent_of_model": True,
        },
        "ema_weights": runtime["ema"],
        "seed": args.seed,
        "num_inference_steps": args.num_steps,
        "guidance_scale": config.guidance_scale,
        "text_prompt": config.text_prompt,
        "crop_size": args.crop_size,
        "seq_len": config.seq_len,
        "dataset_root": config.dataset_root,
        "test_root": config.root_for_split("test"),
        "git_commit": git_commit_hash(),
        "evaluation_scope": "full_test" if max_clips is None else "DEBUG_PARTIAL",
        "is_full_test": max_clips is None,
        "debug_partial": max_clips is not None,
        "clips_total_found": clips_total_found,
        "clips_available": clips_available,
        "clips_evaluated": clips_evaluated,
        "clips_skipped": clips_skipped,
        "skipped_clips": skipped_clips,
        "N_clips": clips_evaluated,
        "N_frames": total_frames,
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes(),
        "numerics": numerics_settings(config),
        "aggregate": {
            "psnr": _mean_std(all_psnr),
            "ssim": _mean_std(all_ssim),
            "lpips": _mean_std(all_lpips),
            "temporal_consistency_l1": _mean_std(all_temporal_error),
            "temporal_consistency_coverage": _mean_std(all_temporal_coverage),
        },
        "reference_weights_by_offset": reference_weights,
        "per_clip": per_clip_results,
    }


def record_evaluation_manifest(
    checkpoint_path: str,
    output_path: Path,
    results: Dict[str, Any],
) -> Path:
    try:
        metadata = load_checkpoint_metadata(checkpoint_path)
    except FileNotFoundError:
        metadata = {}
    configured_path = metadata.get("run_manifest_path")
    manifest_path = Path(configured_path) if configured_path else output_path.parent / "run_manifest.json"
    append_evaluation_to_manifest(
        manifest_path,
        {
            "output_path": str(output_path.resolve()),
            "variant": results["variant"],
            "seed": results["seed"],
            "num_inference_steps": results["num_inference_steps"],
            "N_clips": results["N_clips"],
            "N_frames": results["N_frames"],
            "wall_clock_seconds": results["runtime_seconds"],
            "peak_gpu_memory_bytes": results["peak_gpu_memory_bytes"],
            "git_commit": results["git_commit"],
        },
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full, unfiltered TRDN test-set evaluation.")
    parser.add_argument("--checkpoint", required=True, help="Path to an accelerate checkpoint directory.")
    parser.add_argument("--project-root", default="/content/drive/MyDrive/TRDN_REVIDE")
    parser.add_argument("--dataset-root", default="", help="Optional override for config.test_root's parent.")
    parser.add_argument("--num-steps", type=int, default=None, help="DDIM inference steps.")
    parser.add_argument(
        "--step-sweep",
        type=int,
        nargs="+",
        default=None,
        help="Optional measured DDIM-step sweep; writes one full report per value plus an index JSON.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--train-mode", default="dehaze", choices=["dehaze", "reconstruct_synthetic"])
    parser.add_argument("--mask-mode", default="auto")
    parser.add_argument("--allow-mode-mismatch", action="store_true")
    parser.add_argument("--variant", default="", help="Stable label used by figures and tables.")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--diffusion-only",
        action="store_true",
        help="Task B baseline: SD inpainting only, no RAFT/ConvLSTM/transformer/reference selector.",
    )
    parser.add_argument("--output", default="", help="Output JSON path. Defaults next to the checkpoint.")
    parser.add_argument("--preset", default="", help="Filled numerics YAML preset.")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument(
        "--text-prompt",
        default=None,
    )
    parser.add_argument(
        "--debug-max-clips",
        type=int,
        default=0,
        help="Explicit debug-only clip cap. Output is stamped DEBUG_PARTIAL.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_steps is None and not args.step_sweep:
        parser.error("one of --num-steps or --step-sweep is required")
    if args.num_steps is not None and args.step_sweep:
        parser.error("--num-steps and --step-sweep are mutually exclusive")
    if args.guidance_scale is not None and args.guidance_scale <= 0:
        parser.error("--guidance-scale must be positive")
    try:
        saved_metadata = load_checkpoint_metadata(args.checkpoint)
    except FileNotFoundError:
        saved_metadata = {}
    saved_quality = saved_metadata.get("quality_settings", {})
    guidance_scale = float(
        args.guidance_scale
        if args.guidance_scale is not None
        else saved_quality.get("guidance_scale", 1.0)
    )
    text_prompt = str(
        args.text_prompt
        if args.text_prompt is not None
        else saved_quality.get(
            "text_prompt",
            "a clear clean dehazed video frame",
        )
    )

    config = TRDNConfig(
        project_root=args.project_root,
        resume_from_checkpoint=args.checkpoint,
        allow_mode_mismatch=args.allow_mode_mismatch,
        seq_len=args.seq_len,
        train_mode=args.train_mode,
        mask_mode=args.mask_mode,
        guidance_scale=guidance_scale,
        text_prompt=text_prompt,
        enable_ema=args.use_ema,
    )
    if args.preset:
        apply_numerics_preset(config, args.preset)
    if args.dataset_root:
        config.override_dataset_root(args.dataset_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_output = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).parent / "evaluate_full_test.json"
    )
    base_output.parent.mkdir(parents=True, exist_ok=True)
    steps = args.step_sweep or [args.num_steps]
    sweep_rows = []
    for step_count in steps:
        run_args = argparse.Namespace(**vars(args))
        run_args.num_steps = int(step_count)
        if len(steps) > 1:
            run_args.variant = (
                f"{args.variant}_steps_{step_count}"
                if args.variant
                else f"steps_{step_count}"
            )
            output_path = base_output.with_name(
                f"{base_output.stem}_steps_{step_count}{base_output.suffix}"
            )
        else:
            output_path = base_output
        results = evaluate(
            config,
            run_args,
            device,
            max_clips=args.debug_max_clips or None,
        )
        write_json(output_path, results)
        manifest_path = record_evaluation_manifest(args.checkpoint, output_path, results)
        sweep_rows.append(
            {
                "num_inference_steps": step_count,
                "output_path": str(output_path.resolve()),
                "aggregate": results["aggregate"],
            }
        )
        print(json.dumps({k: v for k, v in results.items() if k != "per_clip"}, indent=2))
        print(f"Wrote {output_path}")
        print(f"Updated {manifest_path}")
    if len(steps) > 1:
        write_json(
            base_output,
            {
                "schema_version": 1,
                "report_type": "ddim_step_sweep",
                "selection_policy": "No automatic best-step selection; review measured results.",
                "results": sweep_rows,
            },
        )
        print(f"Wrote step-sweep index {base_output}")


if __name__ == "__main__":
    main()
