"""Generate the qualitative TRDN figures the project never produced.

Reuses evaluate_full_test's dataset construction and inference call verbatim, so every
pixel shown corresponds to the numbers actually reported (run2 = 21.7838 dB). Emits, per
selected test scene: hazy input | TRDN output | ground truth, plus an absolute-error map
and a zoomed crop. Also writes a per-scene contact sheet across all scenes.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/workspace/repos/TRDN-Video-Dehazing")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.config import TRDNConfig
from src.presets import apply_numerics_preset
from src.metrics import psnr_metric, ssim_metric
import evaluate_full_test as EFT
from src.validate import infer_dehazed_batch

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--project-root", required=True)
p.add_argument("--dataset-root", default="/workspace/datasets/REVIDE_TRDN")
p.add_argument("--preset", default=str(REPO / "configs/a40.yaml"))
p.add_argument("--out", default="/workspace/trdn_paper_run/figures")
p.add_argument("--num-steps", type=int, default=50)
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--seq-len", type=int, default=10)
p.add_argument("--crop-size", type=int, default=256)
p.add_argument("--model-variant", default="full")
p.add_argument("--frames-per-scene", type=int, default=2)
p.add_argument("--guidance-scale", type=float, default=1.0)
p.add_argument("--diffusion-only", action="store_true")
p.add_argument("--train-mode", default="")
p.add_argument("--mask-mode", default="")
p.add_argument("--allow-mode-mismatch", action="store_true")
p.add_argument("--use-ema", action="store_true")
p.add_argument("--text-prompt", default="")
p.add_argument("--debug-max-clips", type=int, default=0)
args = p.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Build the config exactly as evaluate_full_test.main() does, so the figures come from the
# same runtime that produced the reported numbers. Constructing TRDNConfig() bare picks up
# defaults that request xformers, which is not available here.
config = TRDNConfig(
    project_root=args.project_root,
    resume_from_checkpoint=args.checkpoint,
    allow_mode_mismatch=args.allow_mode_mismatch,
    seq_len=args.seq_len,
    train_mode=args.train_mode,
    model_variant=args.model_variant,
    mask_mode=args.mask_mode,
    guidance_scale=args.guidance_scale,
    enable_ema=args.use_ema,
)
if args.text_prompt:
    config.text_prompt = args.text_prompt
if args.preset:
    apply_numerics_preset(config, args.preset)
config.apply_model_variant()
if args.dataset_root:
    config.override_dataset_root(args.dataset_root)
config.crop_size = args.crop_size

ds = EFT.build_test_dataset(config, args)
by_clip = EFT.group_index_by_clip(ds)
runtime = EFT.load_runtime_for_eval(
    config, args.checkpoint, device,
    diffusion_only=bool(args.diffusion_only or config.model_variant == "diffusion_only"),
    use_ema=bool(args.use_ema))
print(f"dataset: {len(ds)} samples, {len(by_clip)} clips", flush=True)

def to_img(t):
    a = t.detach().float().clamp(0, 1).cpu().numpy()
    if a.ndim == 4: a = a[0]
    return np.transpose(a, (1, 2, 0))

rows = []
for seq_idx, idxs in sorted(by_clip.items()):
    name = ds.sequences[seq_idx]["name"]
    picks = np.linspace(0, len(idxs) - 1, min(args.frames_per_scene, len(idxs))).astype(int)
    for pos in picks:
        sample = ds[idxs[int(pos)]]
        batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else [v])
                 for k, v in sample.items()}
        with torch.no_grad():
            o = infer_dehazed_batch(
                batch["frames"], batch["mask"], batch["corrupted_frame"],
                runtime["diffusion"], runtime["temporal_memory"],
                runtime["temporal_transformer"], runtime["reference_selector"],
                runtime["conditioning_adapter"], device,
                raft_model=runtime["model_raft"], num_steps=args.num_steps,
                seed=args.seed, sample_ids=[f"{name}:{pos}"], show_progress=False,
                text_prompt=config.text_prompt, guidance_scale=config.guidance_scale)
        pred, tgt = o["prediction"], batch["target_frame"]
        hazy = batch["corrupted_frame"]
        ps = psnr_metric(pred[0], tgt[0]); ss = ssim_metric(pred[0], tgt[0])
        hz_ps = psnr_metric(hazy[0], tgt[0])
        H, P, T = to_img(hazy), to_img(pred), to_img(tgt)
        err = np.abs(P - T).mean(axis=2)

        fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
        for a_, im, ti in ((ax[0], H, f"Hazy input\n{hz_ps:.2f} dB"),
                           (ax[1], P, f"TRDN\n{ps:.2f} dB / SSIM {ss:.3f}"),
                           (ax[2], T, "Ground truth")):
            a_.imshow(np.clip(im, 0, 1)); a_.set_title(ti, fontsize=11); a_.axis("off")
        m = ax[3].imshow(err, cmap="inferno", vmin=0, vmax=max(0.25, float(err.max())))
        ax[3].set_title(f"Absolute error\nmean {err.mean():.4f}", fontsize=11); ax[3].axis("off")
        fig.colorbar(m, ax=ax[3], fraction=0.046)
        fig.suptitle(f"{name}  frame {pos}", fontsize=12)
        fig.tight_layout()
        stem = out / f"qualitative_{name}_f{pos}"
        fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        rows.append({"scene": name, "frame": int(pos), "psnr": ps, "ssim": ss,
                     "hazy_psnr": hz_ps, "mean_abs_error": float(err.mean())})
        print(f"  {name} f{pos}: hazy {hz_ps:.2f} -> TRDN {ps:.2f} dB (ssim {ss:.3f})", flush=True)

# contact sheet: one representative frame per scene
scenes = sorted({r["scene"] for r in rows})
fig, axes = plt.subplots(len(scenes), 3, figsize=(10.5, 3.4 * len(scenes)))
if len(scenes) == 1: axes = np.array([axes])
for i, sc in enumerate(scenes):
    png = sorted(out.glob(f"qualitative_{sc}_f*.png"))[0]
    im = plt.imread(png)
    axes[i, 0].imshow(im); axes[i, 0].axis("off")
    axes[i, 0].set_ylabel(sc, fontsize=10)
    for j in (1, 2): axes[i, j].axis("off")
    r = next(r for r in rows if r["scene"] == sc)
    axes[i, 0].set_title(f"{sc}: hazy {r['hazy_psnr']:.2f} -> TRDN {r['psnr']:.2f} dB", fontsize=10)
fig.tight_layout()
fig.savefig(out / "contact_sheet.png", dpi=130, bbox_inches="tight")
fig.savefig(out / "contact_sheet.pdf", bbox_inches="tight")
plt.close(fig)

json.dump(rows, open(out / "qualitative_index.json", "w"), indent=2)
print(f"\nwrote {len(rows)} qualitative figures to {out}")
