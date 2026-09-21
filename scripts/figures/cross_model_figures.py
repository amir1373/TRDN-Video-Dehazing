"""Side-by-side qualitative comparison of every model on IDENTICAL 256x256 crops.

Both architectures are driven from the same source root (REVIDE_TRDN) and the same centre-crop
geometry, so the panels differ only by model. Chapter 5 claims the two are not statistically
distinguishable; this is the figure a reader will check that claim against.

Columns: hazy input | VideoEENet (crop-trained) | TRDN run 2 | ground truth
Plus a second row of absolute-error maps for the two models.
"""
import argparse, json, os, sys, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path("/workspace/repos/TRDN-Video-Dehazing")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, "/workspace/repos/videoeennet-agri-dehazing-final/scripts")

from src.config import TRDNConfig
from src.presets import apply_numerics_preset
from src.metrics import psnr_metric, ssim_metric
import evaluate_full_test as EFT
from src.validate import infer_dehazed_batch
from model import VideoEENet

ap = argparse.ArgumentParser()
ap.add_argument("--trdn-ckpt", default="/workspace/trdn_paper_run/runs/full_cos/checkpoints/last")
ap.add_argument("--veenet-ckpt", default="/workspace/videoeennet_runs/aug_crop_seed1234/checkpoints/best_psnr.pt")
ap.add_argument("--project-root", default="/workspace/trdn_paper_run/runs/full_cos")
ap.add_argument("--dataset-root", default="/workspace/datasets/REVIDE_TRDN")
ap.add_argument("--preset", default=str(REPO / "configs/a40.yaml"))
ap.add_argument("--out", default="/workspace/trdn_paper_run/figures")
ap.add_argument("--num-steps", type=int, default=50)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--seq-len", type=int, default=10)
ap.add_argument("--crop-size", type=int, default=256)
ap.add_argument("--model-variant", default="full")
ap.add_argument("--train-mode", default="dehaze")
ap.add_argument("--mask-mode", default="full")
ap.add_argument("--guidance-scale", type=float, default=1.0)
ap.add_argument("--allow-mode-mismatch", action="store_true")
ap.add_argument("--use-ema", action="store_true")
ap.add_argument("--diffusion-only", action="store_true")
ap.add_argument("--text-prompt", default="")
ap.add_argument("--debug-max-clips", type=int, default=0)
ap.add_argument("--per-scene", type=int, default=1)
args = ap.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- TRDN: build exactly as evaluate_full_test does ----
cfg = TRDNConfig(project_root=args.project_root, resume_from_checkpoint=args.trdn_ckpt,
                 allow_mode_mismatch=args.allow_mode_mismatch, seq_len=args.seq_len,
                 train_mode=args.train_mode, model_variant=args.model_variant,
                 mask_mode=args.mask_mode, guidance_scale=args.guidance_scale,
                 enable_ema=args.use_ema)
if args.preset: apply_numerics_preset(cfg, args.preset)
cfg.apply_model_variant()
if args.dataset_root: cfg.override_dataset_root(args.dataset_root)
cfg.crop_size = args.crop_size

ds = EFT.build_test_dataset(cfg, args)
by_clip = EFT.group_index_by_clip(ds)
runtime = EFT.load_runtime_for_eval(cfg, args.trdn_ckpt, dev,
                                    diffusion_only=False, use_ema=bool(args.use_ema))
print(f"TRDN ready: {len(ds)} samples, {len(by_clip)} clips", flush=True)

# ---- VideoEENet ----
ck = torch.load(args.veenet_ckpt, map_location="cpu", weights_only=False)
vc = ck.get("config", {}) or {}
ve = VideoEENet(vc.get("base_channels",32), vc.get("hidden_dim",64),
                vc.get("temporal_mode","convlstm"), vc.get("attention_heads",4),
                vc.get("attention_pool_size",8), vc.get("seq_len",10)).to(dev).eval()
ve.load_state_dict(ck["model"])
print(f"VideoEENet ready ({vc.get('temporal_mode')}, preprocess={vc.get('preprocess')})", flush=True)

def img(t):
    a = t.detach().float().clamp(0,1).cpu().numpy()
    if a.ndim == 4: a = a[0]
    return np.transpose(a, (1,2,0))

rows_meta = []
for seq_idx, idxs in sorted(by_clip.items()):
    name = ds.sequences[seq_idx]["name"].split("_run_")[0]
    picks = np.linspace(0, len(idxs)-1, args.per_scene+2).astype(int)[1:-1]
    for pos in picks:
        sample = ds[idxs[int(pos)]]
        batch = {k:(v.unsqueeze(0).to(dev) if torch.is_tensor(v) else [v]) for k,v in sample.items()}
        with torch.no_grad():
            o = infer_dehazed_batch(batch["frames"], batch["mask"], batch["corrupted_frame"],
                runtime["diffusion"], runtime["temporal_memory"], runtime["temporal_transformer"],
                runtime["reference_selector"], runtime["conditioning_adapter"], dev,
                raft_model=runtime["model_raft"], num_steps=args.num_steps, seed=args.seed,
                sample_ids=[f"{name}:{pos}"], show_progress=False,
                text_prompt=cfg.text_prompt, guidance_scale=cfg.guidance_scale)
        trdn_pred = o["prediction"]; tgt = batch["target_frame"]; hazy = batch["corrupted_frame"]

        # VideoEENet consumes the SAME frame stack TRDN was given
        with torch.no_grad():
            ve_pred = ve(batch["frames"]).clamp(0,1)
        if ve_pred.shape[-2:] != tgt.shape[-2:]:
            ve_pred = F.interpolate(ve_pred, size=tgt.shape[-2:], mode="bilinear", align_corners=False)

        H, Tr, Ve, G = img(hazy), img(trdn_pred), img(ve_pred), img(tgt)
        p_h  = psnr_metric(hazy[0], tgt[0])
        p_tr = psnr_metric(trdn_pred[0], tgt[0]); s_tr = ssim_metric(trdn_pred[0], tgt[0])
        p_ve = psnr_metric(ve_pred[0], tgt[0]);   s_ve = ssim_metric(ve_pred[0], tgt[0])
        e_tr = np.abs(Tr-G).mean(axis=2); e_ve = np.abs(Ve-G).mean(axis=2)
        vmax = max(0.25, float(max(e_tr.max(), e_ve.max())))

        fig, ax = plt.subplots(2, 4, figsize=(16.5, 8.4))
        for a_, im, ti in ((ax[0,0], H,  f"Hazy input\n{p_h:.2f} dB"),
                           (ax[0,1], Ve, f"VideoEENet (1.93 M)\n{p_ve:.2f} dB / SSIM {s_ve:.3f}"),
                           (ax[0,2], Tr, f"TRDN (864 M)\n{p_tr:.2f} dB / SSIM {s_tr:.3f}"),
                           (ax[0,3], G,  "Ground truth")):
            a_.imshow(np.clip(im,0,1)); a_.set_title(ti, fontsize=11); a_.axis("off")
        ax[1,0].axis("off")
        ax[1,0].text(0.5,0.5, f"{name}\nframe {pos}\n\nidentical 256x256\ncentre crop at\nnative resolution",
                     ha="center", va="center", fontsize=11, color="#2d3748")
        m1 = ax[1,1].imshow(e_ve, cmap="inferno", vmin=0, vmax=vmax)
        ax[1,1].set_title(f"VideoEENet error\nmean {e_ve.mean():.4f}", fontsize=10); ax[1,1].axis("off")
        m2 = ax[1,2].imshow(e_tr, cmap="inferno", vmin=0, vmax=vmax)
        ax[1,2].set_title(f"TRDN error\nmean {e_tr.mean():.4f}", fontsize=10); ax[1,2].axis("off")
        fig.colorbar(m2, ax=ax[1,2], fraction=0.046)
        ax[1,3].axis("off")
        d = p_tr - p_ve
        ax[1,3].text(0.5,0.5, f"TRDN − VideoEENet\n{d:+.2f} dB\non this frame",
                     ha="center", va="center", fontsize=12, weight="bold",
                     color="#2b6cb0" if d>0 else "#c53030")
        fig.suptitle(f"Cross-model comparison — {name}, frame {pos}", fontsize=13)
        fig.tight_layout()
        stem = out / f"crossmodel_{name}_f{pos}"
        fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        rows_meta.append({"scene":name,"frame":int(pos),"hazy":p_h,"trdn":p_tr,"veenet":p_ve,
                          "trdn_ssim":s_tr,"veenet_ssim":s_ve,"delta":d})
        print(f"  {name} f{pos}: hazy {p_h:.2f} | VEENet {p_ve:.2f} | TRDN {p_tr:.2f} | Δ {d:+.2f}", flush=True)

json.dump(rows_meta, open(out/"crossmodel_index.json","w"), indent=2)
print(f"\nwrote {len(rows_meta)} cross-model figures to {out}")
