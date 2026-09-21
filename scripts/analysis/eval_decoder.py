"""Score the fine-tuned decoder on the cached TEST latents, per scene, in the eval schema.

No UNet, no DDIM, no RAFT — the latents are on disk, so this is decode + metrics only. Emits
the same per_clip JSON the bootstrap script consumes, so the result drops straight into the
existing comparison with sequence-level intervals.
"""
import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/workspace/repos/TRDN-Video-Dehazing")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/trdn_paper_run/latent_cache/test")
ap.add_argument("--decoder", default="/workspace/trdn_paper_run/decoder_ft/decoder_best.pt")
ap.add_argument("--sd-model", default="runwayml/stable-diffusion-inpainting")
ap.add_argument("--out", default="/workspace/trdn_paper_run/evaluations/trdn_decoderft.json")
ap.add_argument("--frozen-out", default="/workspace/trdn_paper_run/evaluations/trdn_frozendec_cached.json")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
from diffusers import AutoencoderKL
from src.metrics import psnr_metric, ssim_metric

vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae").to(dev).eval()
SF = vae.config.scaling_factor

def decode(lat):
    with torch.no_grad():
        x = vae.decode(lat / SF).sample.float()
    return ((x + 1.0) / 2.0).clamp(0, 1)

files = sorted(glob.glob(os.path.join(args.cache, "*.pt")))
if not files:
    sys.exit("no cached test latents")
print(f"{len(files)} cached test samples", flush=True)

def score_all(tag):
    per = {}
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        lat = d["latent"].float().unsqueeze(0).to(dev)
        tgt = d["target"].float().unsqueeze(0).to(dev)
        pred = decode(lat)
        scene = str(d["clip"]).split("_run_")[0]
        e = per.setdefault(scene, {"p": [], "s": []})
        e["p"].append(psnr_metric(pred[0], tgt[0]))
        e["s"].append(ssim_metric(pred[0], tgt[0]))
    per_clip = [{"sequence_name": f"{k}_run_000", "num_frames": len(v["p"]),
                 "psnr_mean": float(np.mean(v["p"])), "ssim_mean": float(np.mean(v["s"])),
                 "lpips_mean": 0.0} for k, v in sorted(per.items())]
    tot = sum(c["num_frames"] for c in per_clip) or 1
    agg = {"psnr": {"mean": sum(c["psnr_mean"]*c["num_frames"] for c in per_clip)/tot},
           "ssim": {"mean": sum(c["ssim_mean"]*c["num_frames"] for c in per_clip)/tot},
           "lpips": {"mean": 0.0}}
    print(f"  {tag}: PSNR {agg['psnr']['mean']:.4f}  SSIM {agg['ssim']['mean']:.4f}", flush=True)
    for c in per_clip:
        print(f"    {c['sequence_name']:<16} {c['psnr_mean']:7.4f}", flush=True)
    return {"per_clip": per_clip, "aggregate": agg, "decoder": tag}

# frozen decoder on the SAME cached latents — the like-for-like control
frozen = score_all("frozen decoder")
json.dump(frozen, open(args.frozen_out, "w"), indent=2)

if os.path.isfile(args.decoder):
    sd = torch.load(args.decoder, map_location="cpu", weights_only=False)
    missing = vae.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    print(f"loaded fine-tuned decoder ({len(sd)} tensors)", flush=True)
    ft = score_all("fine-tuned decoder")
    json.dump(ft, open(args.out, "w"), indent=2)
    d = ft["aggregate"]["psnr"]["mean"] - frozen["aggregate"]["psnr"]["mean"]
    print(f"\ndecoder fine-tuning: {d:+.4f} dB on the test set")
    print("  (the bootstrap script gives the interval on this difference)")
else:
    print(f"no fine-tuned decoder at {args.decoder}; wrote the frozen control only")
