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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.config import TRDNConfig
from src.dataset import REVIDESequenceDataset
from src.diffusion_adapter import load_diffusion_backbone
from src.flow import flow_warped_temporal_consistency_error, load_raft
from src.losses import LossBundle
from src.metrics import psnr_metric, ssim_metric
from src.train import build_optimizer, build_temporal_modules
from src.validate import infer_dehazed_batch, infer_diffusion_only_batch

REPO_ROOT = Path(__file__).resolve().parent.parent


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def load_runtime_for_eval(config: TRDNConfig, checkpoint_path: str, device: str) -> Dict[str, Any]:
    diffusion = load_diffusion_backbone(config, device=device)
    temporal_memory, temporal_transformer, reference_selector, conditioning_adapter = build_temporal_modules(
        config, diffusion["unet"].config.cross_attention_dim, device
    )
    if checkpoint_path:
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
    raft_model = load_raft(device, config.freeze_raft) if torch.cuda.is_available() else None
    return {
        "diffusion": diffusion,
        "temporal_memory": temporal_memory,
        "temporal_transformer": temporal_transformer,
        "reference_selector": reference_selector,
        "conditioning_adapter": conditioning_adapter,
        "raft_model": raft_model,
    }


def build_test_dataset(config: TRDNConfig, args: argparse.Namespace) -> REVIDESequenceDataset:
    return REVIDESequenceDataset(
        config.root_for_split("test"),
        split="test",
        seq_len=config.seq_len,
        crop_size=args.crop_size,
        random_crop=False,
        extensions=config.image_extensions,
        synthetic_if_empty=True,
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
def evaluate(config: TRDNConfig, args: argparse.Namespace, device: str) -> Dict[str, Any]:
    dataset = build_test_dataset(config, args)
    if not dataset.index:
        raise RuntimeError(
            f"No real REVIDE test sequences found under {config.root_for_split('test')!r}. "
            "Full-test evaluation requires real test data; refusing to silently fall back to "
            "synthetic placeholder clips (that would not be a real evaluation)."
        )
    runtime = load_runtime_for_eval(config, args.checkpoint, device)
    loss_bundle = LossBundle(device)

    by_clip = group_index_by_clip(dataset)
    clip_names = {
        seq_idx: (dataset.sequences[seq_idx]["name"] if dataset.sequences else "synthetic") for seq_idx in by_clip
    }
    consistency_raft = runtime["raft_model"] or load_raft(device, config.freeze_raft)

    per_clip_results: List[Dict[str, Any]] = []
    all_psnr: List[float] = []
    all_ssim: List[float] = []
    all_lpips: List[float] = []
    all_temporal_error: List[float] = []
    all_temporal_coverage: List[float] = []
    total_frames = 0

    for seq_idx in sorted(by_clip):
        clip_name = clip_names[seq_idx]
        dataset_indices = by_clip[seq_idx]  # already ascending by construction of dataset.index
        clip_psnr, clip_ssim, clip_lpips, clip_temporal_error, clip_temporal_coverage = [], [], [], [], []
        prev_prediction: Optional[torch.Tensor] = None

        for frame_position, dataset_idx in enumerate(dataset_indices):
            sample = dataset[dataset_idx]
            batch = {key: (value.unsqueeze(0).to(device) if torch.is_tensor(value) else [value]) for key, value in sample.items()}
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
                    raft_model=runtime["raft_model"],
                    num_steps=args.num_steps,
                    seed=args.seed,
                    sample_ids=sample_ids,
                )

            prediction = output["prediction"]
            target = batch["target_frame"]
            clip_psnr.append(psnr_metric(prediction[0], target[0]))
            clip_ssim.append(ssim_metric(prediction[0], target[0]))
            clip_lpips.append(float(loss_bundle.lpips_loss(prediction, target).detach().cpu()))

            if prev_prediction is not None:
                temporal_error, coverage = flow_warped_temporal_consistency_error(prev_prediction, prediction, consistency_raft)
                clip_temporal_error.append(temporal_error)
                clip_temporal_coverage.append(coverage)
            prev_prediction = prediction
            total_frames += 1

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

    def _mean_std(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"mean": None, "std": None}
        arr = np.asarray(values, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    return {
        "checkpoint_path": args.checkpoint,
        "train_mode": dataset.train_mode,
        "mask_mode": dataset._resolve_mask_mode(),
        "diffusion_only_baseline": args.diffusion_only,
        "seed": args.seed,
        "num_inference_steps": args.num_steps,
        "crop_size": args.crop_size,
        "git_commit": git_commit_hash(),
        "N_clips": len(per_clip_results),
        "N_frames": total_frames,
        "aggregate": {
            "psnr": _mean_std(all_psnr),
            "ssim": _mean_std(all_ssim),
            "lpips": _mean_std(all_lpips),
            "temporal_consistency_l1": _mean_std(all_temporal_error),
            "temporal_consistency_coverage": _mean_std(all_temporal_coverage),
        },
        "per_clip": per_clip_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full, unfiltered TRDN test-set evaluation.")
    parser.add_argument("--checkpoint", required=True, help="Path to an accelerate checkpoint directory.")
    parser.add_argument("--project-root", default="/content/drive/MyDrive/TRDN_REVIDE")
    parser.add_argument("--dataset-root", default="", help="Optional override for config.test_root's parent.")
    parser.add_argument("--num-steps", type=int, required=True, help="DDIM inference steps (no hardcoded default).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-mode", default="dehaze", choices=["dehaze", "reconstruct_synthetic"])
    parser.add_argument("--mask-mode", default="auto")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--diffusion-only",
        action="store_true",
        help="Task B baseline: SD inpainting only, no RAFT/ConvLSTM/transformer/reference selector.",
    )
    parser.add_argument("--output", default="", help="Output JSON path. Defaults next to the checkpoint.")
    args = parser.parse_args()

    config = TRDNConfig(project_root=args.project_root, resume_from_checkpoint=args.checkpoint)
    if args.dataset_root:
        config.dataset_root = args.dataset_root
        config.test_root = args.dataset_root

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = evaluate(config, args, device)

    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent / "evaluate_full_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "per_clip"}, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
