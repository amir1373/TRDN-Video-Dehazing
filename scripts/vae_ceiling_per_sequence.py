"""Per-sequence VAE round-trip ceiling.

Why this exists: the aggregate ceiling in vae_ceiling.json (PSNR 24.698 / SSIM 0.7719)
is a MEAN over all Test clean frames from six very different scenes. Trained models have
already EXCEEDED it per-sequence (W002 hit 26.89 PSNR) and in aggregate SSIM (0.7751),
so it is not a valid upper bound. This reports the ceiling per sequence so headroom can
be judged scene by scene.

Reuses vae_ceiling.py's own helpers so the measurement is identical, only grouped.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TRDNConfig
from src.dataset import discover_revide_sequences, image_to_tensor
from src.losses import LossBundle
from src.metrics import psnr_metric, ssim_metric
from src.progress import ProgressReporter
from src.provenance import git_state, write_json

from importlib import import_module
_vc = import_module("scripts.vae_ceiling") if "scripts" in sys.modules else None
if _vc is None:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    _vc = import_module("vae_ceiling")
_load_vae = _vc._load_vae
_round_trip = _vc._round_trip


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--model-id", default="runwayml/stable-diffusion-inpainting")
    p.add_argument("--precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--split", default="test")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()

    config = TRDNConfig()
    config.override_dataset_root(str(Path(args.dataset_root)))
    root = Path(config.root_for_split(args.split))
    sequences = discover_revide_sequences(root, args.split, config.image_extensions)
    if not sequences:
        raise RuntimeError(f"No sequences discovered under {root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = _load_vae(args, device)
    loss_bundle = LossBundle(device)

    total = sum(len(s["clean_files"]) for s in sequences)
    progress = ProgressReporter(total, f"VAE ceiling/seq ({args.precision})", leave=True)
    per_seq, all_p, all_s, all_l = [], [], [], []

    for seq in sequences:
        name = seq.get("sequence_name") or seq.get("name") or "unknown"
        ps, ss, ls = [], [], []
        for path in sorted({q.resolve() for q in seq["clean_files"]}):
            target = image_to_tensor(path).unsqueeze(0)
            target = F.interpolate(target, size=(args.resolution, args.resolution),
                                   mode="bilinear", align_corners=False).to(device)
            with torch.no_grad():
                recon = _round_trip(vae, target)
            ps.append(psnr_metric(recon[0], target[0]))
            ss.append(ssim_metric(recon[0], target[0]))
            ls.append(float(loss_bundle.lpips_loss(recon, target).detach().cpu()))
            progress.update(1)
        per_seq.append({
            "sequence_name": name,
            "num_frames": len(ps),
            "psnr": summary(ps), "ssim": summary(ss), "lpips": summary(ls),
        })
        all_p += ps; all_s += ss; all_l += ls
        progress.set_postfix({"seq": name[:10], "psnr": f"{np.mean(ps):.2f}"})
    progress.close()

    per_seq.sort(key=lambda r: r["psnr"]["mean"])
    report = {
        "schema_version": 1,
        "measurement": "deterministic_vae_posterior_mode_round_trip_per_sequence",
        "model_id": args.model_id,
        "resolution": args.resolution,
        "precision": args.precision,
        "device": device,
        "split": args.split,
        "N_frames": len(all_p),
        "aggregate": {"psnr": summary(all_p), "ssim": summary(all_s), "lpips": summary(all_l)},
        "per_sequence": per_seq,
        "git": git_state(),
        "note": ("Per-sequence ceilings. The aggregate is a frame-mean across heterogeneous "
                 "scenes and must NOT be used as a per-sequence upper bound."),
    }
    write_json(args.output, report)
    print(f"\nwrote {args.output}")
    print(f"{'sequence':<18}{'frames':>7}{'psnr':>9}{'ssim':>9}{'lpips':>9}")
    for r in per_seq:
        print(f"{r['sequence_name'][:16]:<18}{r['num_frames']:>7}"
              f"{r['psnr']['mean']:>9.3f}{r['ssim']['mean']:>9.4f}{r['lpips']['mean']:>9.4f}")
    a = report["aggregate"]
    print(f"{'AGGREGATE':<18}{len(all_p):>7}{a['psnr']['mean']:>9.3f}"
          f"{a['ssim']['mean']:>9.4f}{a['lpips']['mean']:>9.4f}")


if __name__ == "__main__":
    main()
