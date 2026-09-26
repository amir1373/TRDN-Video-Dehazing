"""Non-TRDN baselines for the partial-occlusion experiment, on the same occluded test windows.

(a) sd_then_single: the pretrained Stable Diffusion inpainting model (no fine-tuning) fills the
    occluded region of the current frame zero-shot; VideoEENet's single-frame model then dehazes
    the filled frame. Evaluated for both scopes.
(b) raftfill_then_videoeenet ("current" scope only): the hidden pixels of the current frame are
    filled from the previous frame warped forward under constant velocity (RAFT flow t-2 -> t-1
    applied to frame t-1); the VideoEENet reference model then dehazes the window. With a lens
    occluder the previous frames are hidden at the same place, so this baseline does not apply.

Metrics and occluders are identical to evaluate_full_test.py (--train-mode occlude).
usage: occlusion_baselines.py --dataset-root DIR --veenet-code DIR --veenet-single CKPT
       --veenet-ref CKPT --coverage 0.35 --scope current --output OUT.json [--save-predictions P]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import TRDNConfig  # noqa: E402
from src.dataset import REVIDESequenceDataset  # noqa: E402
from src.diffusion_adapter import load_diffusion_backbone  # noqa: E402
from src.flow import compute_raft_flow, load_raft  # noqa: E402
from src.losses import LossBundle  # noqa: E402
from src.metrics import psnr_metric, ssim_metric  # noqa: E402
from src.occlusion import occluder_key  # noqa: E402
from src.presets import apply_numerics_preset  # noqa: E402
from src.validate import infer_diffusion_only_batch  # noqa: E402
from src.warp import warp_with_flow  # noqa: E402


def region_psnr(pred: torch.Tensor, target: torch.Tensor, region: torch.Tensor):
    weight = region.float().expand_as(target)
    count = float(weight.sum())
    if count < 1:
        return None
    mse = float((((pred.float().clamp(0, 1) - target.float().clamp(0, 1)) ** 2) * weight).sum()) / count
    return 99.0 if mse <= 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def load_veenet(code_dir: str, checkpoint: str, device: str):
    sys.path.insert(0, code_dir)
    from model import VideoEENet, load_model_state  # noqa: WPS433

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) or {}
    model = VideoEENet(cfg.get("base_channels", 32), cfg.get("hidden_dim", 64), cfg.get("temporal_mode", "convlstm"),
                       cfg.get("attention_heads", 4), cfg.get("attention_pool_size", 8), cfg.get("seq_len", 10),
                       output_mode=cfg.get("output_mode", "sigmoid_sum"),
                       in_channels=4 if cfg.get("occlusion") else 3).to(device)
    load_model_state(model, ck["model"])
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--veenet-code", required=True)
    ap.add_argument("--veenet-single", required=True, help="VideoEENet single-frame checkpoint (for baseline a)")
    ap.add_argument("--veenet-ref", required=True, help="VideoEENet reference checkpoint (for baseline b)")
    ap.add_argument("--coverage", type=float, required=True)
    ap.add_argument("--scope", choices=["lens", "current"], required=True)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--prompt", default="a photo of an indoor room")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", required=True)
    ap.add_argument("--save-predictions", default="")
    ap.add_argument("--preset", default="", help="Numerics preset (the same one every TRDN run uses).")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()

    config = TRDNConfig(train_mode="occlude", model_variant="diffusion_only", train_unet=False)
    config.override_dataset_root(args.dataset_root)
    if args.preset:
        apply_numerics_preset(config, args.preset)
    dataset = REVIDESequenceDataset(
        config.root_for_split("test"), split="test", seq_len=10, crop_size=256, random_crop=False,
        extensions=config.image_extensions, synthetic_if_empty=False, train_mode="occlude",
        include_prev_frame=False, include_reference_frames=True,
        occlusion_eval_coverage=args.coverage, occlusion_scope=args.scope,
    )
    diffusion = load_diffusion_backbone(config, device)          # pretrained SD inpainting, not fine-tuned
    raft = load_raft(device, True, False)
    single = load_veenet(args.veenet_code, args.veenet_single, device)
    reference = load_veenet(args.veenet_code, args.veenet_ref, device)
    lpips = LossBundle(device)

    methods = ["sd_then_single"] + (["raftfill_then_videoeenet"] if args.scope == "current" else [])
    rows: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
    saved: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in methods}
    for index in range(len(dataset)):
        sample = dataset[index]
        key = occluder_key(sample["frame_paths"][-1])
        frames = sample["frames"].unsqueeze(0).to(device)          # occluded window
        hazy = sample["hazy_frames"].unsqueeze(0).to(device)       # the same window, unoccluded
        mask = sample["mask"].unsqueeze(0).to(device)
        target = sample["target_frame"].unsqueeze(0).to(device)
        outputs = {}
        with torch.no_grad():
            filled = infer_diffusion_only_batch(
                mask, frames[:, -1], diffusion, device, num_steps=args.num_steps, seed=args.seed,
                sample_ids=[key], show_progress=False, text_prompt=args.prompt, guidance_scale=args.guidance_scale,
            )["prediction"].float().clamp(0, 1)
            current = frames[:, -1] * (1 - mask) + filled * mask   # keep the visible pixels as captured
            outputs["sd_then_single"] = single(current.unsqueeze(1)).clamp(0, 1)
            if "raftfill_then_videoeenet" in methods:
                flow = compute_raft_flow(raft, hazy[:, -3], hazy[:, -2])
                predicted = warp_with_flow(hazy[:, -2], flow)
                window = frames.clone()
                window[:, -1] = frames[:, -1] * (1 - mask) + predicted * mask
                outputs["raftfill_then_videoeenet"] = reference(window).clamp(0, 1)
        for method, pred in outputs.items():
            rows[method].append({
                "key": key,
                "psnr": psnr_metric(pred[0], target[0]),
                "ssim": ssim_metric(pred[0], target[0]),
                "lpips": float(lpips.lpips_loss(pred, target).detach().cpu()),
                "psnr_occluded": region_psnr(pred[0], target[0], mask[0]),
                "psnr_visible": region_psnr(pred[0], target[0], 1 - mask[0]),
                "coverage": float(mask.mean()),
            })
            if args.save_predictions:
                saved[method][key.replace("/", "|")] = pred[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        if index % 20 == 0:
            print(f"{index + 1}/{len(dataset)} " + " ".join(
                f"{m} {np.mean([r['psnr'] for r in rows[m]]):.3f}" for m in methods), flush=True)

    result = {"coverage": args.coverage, "scope": args.scope, "n_windows": len(dataset),
              "runtime_seconds": time.perf_counter() - started, "settings": vars(args), "methods": {}}
    for method in methods:
        agg = {k: float(np.mean([r[k] for r in rows[method] if r[k] is not None]))
               for k in ("psnr", "ssim", "lpips", "psnr_occluded", "psnr_visible", "coverage")}
        result["methods"][method] = {"aggregate": agg, "per_window": rows[method]}
        print(f"AGGREGATE {method} " + " ".join(f"{k} {v:.4f}" for k, v in agg.items()), flush=True)
        if args.save_predictions:
            out = Path(args.save_predictions.replace(".npz", f"_{method}.npz"))
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **saved[method])
    Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
