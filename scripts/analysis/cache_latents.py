"""Cache TRDN's PREDICTED latents alongside their ground-truth frames.

WHY: the pipeline scores 34.566 dB when the frozen VAE decodes GROUND-TRUTH latents, but only
21.78 dB when it decodes the latents TRDN actually predicts. That 12.8 dB gap is either poor
latents or a decoder that was trained on clean SD latents and mishandles restoration-specific
ones. Caching the predicted latents lets us fine-tune the decoder on them and find out which.

Inference only — no backward pass. This is the expensive stage; the decoder training that
follows is cheap because it skips the 859M UNet and the 50-step DDIM chain entirely.
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path("/workspace/repos/TRDN-Video-Dehazing")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from src.config import TRDNConfig
from src.presets import apply_numerics_preset
import evaluate_full_test as EFT
from src.validate import infer_dehazed_batch

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="/workspace/trdn_paper_run/runs/full_cos/checkpoints/last")
ap.add_argument("--project-root", default="/workspace/trdn_paper_run/runs/full_cos")
ap.add_argument("--dataset-root", default="/workspace/datasets/REVIDE_TRDN")
ap.add_argument("--preset", default=str(REPO / "configs/a40.yaml"))
ap.add_argument("--out", default="/workspace/trdn_paper_run/latent_cache")
ap.add_argument("--split", default="test", choices=["test", "train"])
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
ap.add_argument("--limit", type=int, default=0, help="cap samples, for a smoke test")
args = ap.parse_args()

out = Path(args.out) / args.split
out.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = TRDNConfig(project_root=args.project_root, resume_from_checkpoint=args.checkpoint,
                 allow_mode_mismatch=args.allow_mode_mismatch, seq_len=args.seq_len,
                 train_mode=args.train_mode, model_variant=args.model_variant,
                 mask_mode=args.mask_mode, guidance_scale=args.guidance_scale,
                 enable_ema=args.use_ema)
if args.preset: apply_numerics_preset(cfg, args.preset)
cfg.apply_model_variant()
if args.dataset_root: cfg.override_dataset_root(args.dataset_root)
cfg.crop_size = args.crop_size

# The training split is reached by pointing the test-set builder at the train root.
if args.split == "train":
    for attr in ("test_root",):
        cur = getattr(cfg, attr, "")
        if cur and "Test" in str(cur):
            setattr(cfg, attr, str(cur).replace("Test", "Train"))
            print(f"  redirected {attr} -> {getattr(cfg, attr)}")

ds = EFT.build_test_dataset(cfg, args)
by_clip = EFT.group_index_by_clip(ds)
runtime = EFT.load_runtime_for_eval(cfg, args.checkpoint, dev,
                                    diffusion_only=False, use_ema=bool(args.use_ema))
total = sum(len(v) for v in by_clip.values())
if args.limit: total = min(total, args.limit)
print(f"caching {args.split}: {total} samples across {len(by_clip)} clips", flush=True)

n, t0 = 0, time.time()
for seq_idx, idxs in sorted(by_clip.items()):
    name = ds.sequences[seq_idx]["name"]
    for pos, di in enumerate(idxs):
        if args.limit and n >= args.limit: break
        sample = ds[di]
        batch = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else [v])
                 for k, v in sample.items()}
        with torch.no_grad():
            o = infer_dehazed_batch(
                batch["frames"], batch["mask"], batch["corrupted_frame"],
                runtime["diffusion"], runtime["temporal_memory"],
                runtime["temporal_transformer"], runtime["reference_selector"],
                runtime["conditioning_adapter"], dev,
                raft_model=runtime["model_raft"], num_steps=args.num_steps,
                seed=args.seed, sample_ids=[f"{name}:{pos}"], show_progress=False,
                text_prompt=cfg.text_prompt, guidance_scale=cfg.guidance_scale)
        torch.save({
            "latent": o["latents"][0].detach().half().cpu(),      # predicted latent
            "target": batch["target_frame"][0].detach().half().cpu(),  # GT frame, [0,1]
            "clip": name, "pos": int(pos),
        }, out / f"{name}_{pos:04d}.pt")
        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f"  {n}/{total}  {el/n:.2f}s/sample  eta {(total-n)*el/n/60:.1f} min", flush=True)
    if args.limit and n >= args.limit: break

print(f"\ncached {n} samples to {out}  ({(time.time()-t0)/60:.1f} min)")
