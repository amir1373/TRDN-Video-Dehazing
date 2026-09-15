"""Measure the Stable Diffusion VAE clean-frame round-trip quality ceiling."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TRDNConfig
from src.dataset import discover_revide_sequences, image_to_tensor
from src.diffusion_adapter import (
    decode_latents_to_images,
    normalize_to_neg_one_to_one,
    resolve_frozen_dtype,
)
from src.losses import LossBundle
from src.metrics import psnr_metric, ssim_metric
from src.progress import ProgressReporter
from src.provenance import git_state, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug cap; 0 uses every clean frame.")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _load_vae(args: argparse.Namespace, device: str):
    from diffusers import AutoencoderKL

    dtype = resolve_frozen_dtype(args.precision, device)
    return AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device).eval()


def _round_trip(vae: Any, image: torch.Tensor) -> torch.Tensor:
    dtype = next(vae.parameters()).dtype
    normalized = normalize_to_neg_one_to_one(image).to(device=vae.device, dtype=dtype)
    posterior = vae.encode(normalized).latent_dist
    latents = posterior.mode() * vae.config.scaling_factor
    return decode_latents_to_images(vae, latents)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    config = TRDNConfig()
    config.override_dataset_root(str(dataset_root))
    test_root = Path(config.root_for_split("test"))
    sequences = discover_revide_sequences(test_root, "test", config.image_extensions)
    clean_frames = sorted(
        {path.resolve() for sequence in sequences for path in sequence["clean_files"]}
    )
    if not clean_frames:
        raise RuntimeError(
            f"No paired REVIDE clean frames were discovered under {test_root}."
        )
    if args.max_frames > 0:
        clean_frames = clean_frames[: args.max_frames]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = _load_vae(args, device)
    loss_bundle = LossBundle(device)
    psnrs, ssims, lpips_values = [], [], []
    progress = ProgressReporter(len(clean_frames), "VAE ceiling", leave=True)
    for path in clean_frames:
        target = image_to_tensor(path).unsqueeze(0)
        target = F.interpolate(
            target,
            size=(args.resolution, args.resolution),
            mode="bilinear",
            align_corners=False,
        ).to(device)
        with torch.no_grad():
            reconstruction = _round_trip(vae, target)
        psnrs.append(psnr_metric(reconstruction[0], target[0]))
        ssims.append(ssim_metric(reconstruction[0], target[0]))
        lpips_values.append(
            float(loss_bundle.lpips_loss(reconstruction, target).detach().cpu())
        )
        progress.set_postfix({"mean_psnr": f"{np.mean(psnrs):.3f}"})
        progress.update(1)
    progress.close()

    def summary(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {"mean": float(array.mean()), "std": float(array.std())}

    report = {
        "schema_version": 1,
        "measurement": "deterministic_vae_posterior_mode_round_trip",
        "model_id": args.model_id,
        "resolution": args.resolution,
        "precision": args.precision,
        "device": device,
        "N_frames": len(clean_frames),
        "debug_capped": args.max_frames > 0,
        "dataset_root": str(dataset_root.resolve()),
        "test_root": str(test_root.resolve()),
        "git": git_state(),
        "aggregate": {
            "psnr": summary(psnrs),
            "ssim": summary(ssims),
            "lpips": summary(lpips_values),
        },
    }
    write_json(args.output, report)
    return report


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(f"VAE CEILING FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(report, indent=2, default=str))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
