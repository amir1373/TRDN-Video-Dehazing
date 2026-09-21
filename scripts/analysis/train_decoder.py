"""Fine-tune the frozen VAE decoder on TRDN's OWN predicted latents.

THE QUESTION: the pipeline scores 34.566 dB when the decoder is given GROUND-TRUTH latents but
only 21.78 dB on the latents TRDN actually predicts. That 12.8 dB gap is either (a) the latents
are poor, or (b) the decoder — trained only on clean SD latents — mishandles restoration
latents and amplifies their defects. Nobody has separated the two.

This trains ONLY the decoder, on cached (predicted latent -> ground-truth frame) pairs. The
UNet, the DDIM chain, RAFT and the temporal modules are never run: the latents are already on
disk, so an epoch costs seconds rather than hours.

Both outcomes are reportable. A large recovery means the decoder is a fixable bottleneck. A
small one proves the gap is latent quality, which sharpens the thesis's claim that the failure
is upstream — where the reference-selector collapse and the inert flow loss already point.
"""
import argparse, glob, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO = Path("/workspace/repos/TRDN-Video-Dehazing")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/trdn_paper_run/latent_cache")
ap.add_argument("--out", default="/workspace/trdn_paper_run/decoder_ft")
ap.add_argument("--sd-model", default="runwayml/stable-diffusion-inpainting")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--w-lpips", type=float, default=0.25)
ap.add_argument("--val-frac", type=float, default=0.12)
ap.add_argument("--seed", type=int, default=1234)
args = ap.parse_args()

torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

class LatentPairs(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu", weights_only=False)
        return d["latent"].float(), d["target"].float()

train_files = sorted(glob.glob(os.path.join(args.cache, "train", "*.pt")))
if not train_files:
    sys.exit("no cached training latents found")
random.Random(args.seed).shuffle(train_files)
n_val = max(8, int(len(train_files) * args.val_frac))
val_files, train_files = train_files[:n_val], train_files[n_val:]
print(f"train pairs {len(train_files)} | val pairs {len(val_files)}", flush=True)

tl = DataLoader(LatentPairs(train_files), batch_size=args.batch_size, shuffle=True,
                num_workers=4, drop_last=True)
vl = DataLoader(LatentPairs(val_files), batch_size=args.batch_size, shuffle=False, num_workers=2)

# ---- the VAE: freeze everything, then unfreeze the decoder only ----
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae").to(dev)
for p in vae.parameters(): p.requires_grad_(False)
dec_params = []
for n, p in vae.named_parameters():
    if n.startswith("decoder") or n.startswith("post_quant_conv"):
        p.requires_grad_(True); dec_params.append(p)
n_dec = sum(p.numel() for p in dec_params)
print(f"decoder parameters being trained: {n_dec:,} of {sum(p.numel() for p in vae.parameters()):,}", flush=True)
assert n_dec > 0, "no decoder parameters selected"

import lpips
lp = lpips.LPIPS(net="alex").to(dev).eval()
for p in lp.parameters(): p.requires_grad_(False)

SF = vae.config.scaling_factor

def decode(latents):
    """Mirrors src/diffusion_adapter.decode_latents_to_images so the numbers stay comparable."""
    x = vae.decode(latents / SF).sample.float()
    return ((x + 1.0) / 2.0).clamp(0, 1)

def psnr(a, b):
    mse = F.mse_loss(a.clamp(0,1), b.clamp(0,1)).item()
    return 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))

@torch.no_grad()
def evaluate(loader):
    vae.eval(); ps = []
    for lat, tgt in loader:
        lat, tgt = lat.to(dev), tgt.to(dev)
        ps.append(psnr(decode(lat), tgt))
    return float(np.mean(ps))

base = evaluate(vl)
print(f"\nBASELINE (frozen decoder, predicted latents, val): {base:.4f} dB", flush=True)

opt = torch.optim.AdamW(dec_params, lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
hist = [{"epoch": 0, "val_psnr": base, "note": "frozen decoder baseline"}]
best = base

for ep in range(1, args.epochs + 1):
    vae.train(); t0 = time.time(); losses = []
    for lat, tgt in tl:
        lat, tgt = lat.to(dev), tgt.to(dev)
        pred = decode(lat)
        l1 = F.l1_loss(pred, tgt)
        # LPIPS wants [-1,1]
        pl = lp(pred * 2 - 1, tgt * 2 - 1).mean()
        loss = l1 + args.w_lpips * pl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec_params, 1.0)
        opt.step()
        losses.append(float(loss))
    sched.step()
    v = evaluate(vl)
    hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_psnr": v,
                 "gain_vs_frozen": v - base})
    print(f"epoch {ep}/{args.epochs}  loss {np.mean(losses):.4f}  val {v:.4f} dB "
          f"({v-base:+.4f} vs frozen)  {time.time()-t0:.0f}s", flush=True)
    if v > best:
        best = v
        torch.save({n: p.detach().half().cpu() for n, p in vae.named_parameters()
                    if n.startswith("decoder") or n.startswith("post_quant_conv")},
                   out / "decoder_best.pt")
    json.dump(hist, open(out / "history.json", "w"), indent=2)

print(f"\nfrozen decoder {base:.4f} dB -> fine-tuned {best:.4f} dB  ({best-base:+.4f} dB on val)")
json.dump({"baseline_val_psnr": base, "best_val_psnr": best, "gain": best - base,
           "n_train": len(train_files), "n_val": len(val_files),
           "epochs": args.epochs, "lr": args.lr, "w_lpips": args.w_lpips,
           "decoder_params": n_dec}, open(out / "summary.json", "w"), indent=2)
print("wrote", out / "summary.json")
