"""VAE ceiling measured through the EXACT eval harness.

Why: vae_ceiling.py RESIZES 284 unique clean frames to 256x256, while
evaluate_full_test.py CENTER-CROPS 256x256 and scores 230 window targets.
Different frames AND different preprocessing -> the old ceiling was never an
upper bound for the evaluation, which is why run2 SSIM 0.7751 exceeded the
"ceiling" 0.7719. This reuses build_test_dataset()/group_index_by_clip() and
batch["target_frame"] so the ceiling is directly comparable, per-sequence.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import TRDNConfig
from src.losses import LossBundle
from src.metrics import psnr_metric, ssim_metric
from src.progress import ProgressReporter
from src.provenance import git_state, write_json
import evaluate_full_test as EFT
import vae_ceiling as VC


def summary(v):
    a = np.asarray(v, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--model-id", default="runwayml/stable-diffusion-inpainting")
    ap.add_argument("--precision", choices=["no", "fp16", "bf16"], default="fp16")
    ap.add_argument("--train-mode", default="dehaze")
    ap.add_argument("--mask-mode", default="auto")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    config = TRDNConfig()
    config.override_dataset_root(args.dataset_root)
    config.seq_len = args.seq_len
    ds = EFT.build_test_dataset(config, args)
    groups = EFT.group_index_by_clip(ds)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VC._load_vae(args, device)
    lb = LossBundle(device)

    names = getattr(ds, "sequence_names", None)
    total = sum(len(v) for v in groups.values())
    prog = ProgressReporter(total, "matched ceiling", leave=True)
    per_seq, ap_, as_, al_ = [], [], [], []

    for sidx, idxs in groups.items():
        ps, ss, ls = [], [], []
        for i in idxs:
            target = ds[i]["target_frame"].unsqueeze(0).to(device)
            with torch.no_grad():
                recon = VC._round_trip(vae, target)
            ps.append(psnr_metric(recon[0], target[0]))
            ss.append(ssim_metric(recon[0], target[0]))
            ls.append(float(lb.lpips_loss(recon, target).detach().cpu()))
            prog.update(1)
        nm = None
        if isinstance(names, (list, tuple)) and sidx < len(names):
            nm = names[sidx]
        if nm is None:
            seqs = getattr(ds, "sequences", None)
            if seqs and sidx < len(seqs):
                nm = seqs[sidx].get("sequence_name") if isinstance(seqs[sidx], dict) else None
        per_seq.append({"sequence_index": int(sidx), "sequence_name": nm or f"seq_{sidx}",
                        "num_frames": len(ps),
                        "psnr": summary(ps), "ssim": summary(ss), "lpips": summary(ls)})
        ap_ += ps; as_ += ss; al_ += ls
    prog.close()

    per_seq.sort(key=lambda r: r["psnr"]["mean"])
    rep = {"schema_version": 1,
           "measurement": "vae_round_trip_through_evaluate_full_test_harness",
           "preprocessing": f"center crop {args.crop_size} (matches evaluate_full_test.py)",
           "model_id": args.model_id, "precision": args.precision, "device": device,
           "N_frames": len(ap_),
           "aggregate": {"psnr": summary(ap_), "ssim": summary(as_), "lpips": summary(al_)},
           "per_sequence": per_seq, "git": git_state(),
           "note": ("Directly comparable to evaluations/*.json: same dataset object, same "
                    "center crop, same target_frame, same metric functions. The older "
                    "vae_ceiling.json used 284 RESIZED frames and is NOT comparable.")}
    write_json(args.output, rep)
    print(f"\nwrote {args.output}")
    print(f"{'sequence':<18}{'frames':>7}{'psnr':>9}{'ssim':>9}{'lpips':>9}")
    for r in per_seq:
        print(f"{r['sequence_name'][:16]:<18}{r['num_frames']:>7}{r['psnr']['mean']:>9.3f}"
              f"{r['ssim']['mean']:>9.4f}{r['lpips']['mean']:>9.4f}")
    a = rep["aggregate"]
    print(f"{'AGGREGATE':<18}{len(ap_):>7}{a['psnr']['mean']:>9.3f}{a['ssim']['mean']:>9.4f}{a['lpips']['mean']:>9.4f}")
    print("\nold (resized, 284 frames): psnr 24.698 ssim 0.7719 lpips 0.0662  <- NOT comparable")


if __name__ == "__main__":
    main()
