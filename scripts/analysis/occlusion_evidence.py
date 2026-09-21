"""Evidence for TRDN's occlusion / temporal-fusion design rationale.

Tests two predictions of the "dehazing is partly inpainting" argument, and reports what the
reference selector actually did. n = 6 scenes: EXPLORATORY and correlational. No p-values.

P1  Denser haze (lower transmission t) => single-frame inversion J=(I-A(1-t))/t is more
    ill-conditioned => more of the signal must come from a learned prior. If TRDN's inpainting
    prior is doing work, its RELATIVE gain over the hazy input should be larger where t is low.
    Relative gain is used because ABSOLUTE PSNR is confounded by scene difficulty.

P2  Haze that fluctuates more over time (higher std of t) makes successive frames more
    decorrelated observations of the same scene point, which is what temporal fusion can
    exploit. So gain should rise with std_t.
"""
import json, os, glob, sys
import numpy as np
from PIL import Image

D = "/workspace/trdn_paper_run"
ROOT = "/workspace/datasets/REVIDE_sequences"
haze = json.load(open(f"{D}/haze_analysis.json"))

def scene_of(name):
    return name.split("_run_")[0].split("_seq_")[0]

def load_eval(fn):
    d = json.load(open(f"{D}/evaluations/{fn}"))
    return d, {scene_of(c["sequence_name"]): c for c in d["per_clip"]}

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))

# --- per-scene PSNR of the HAZY INPUT itself (the floor each model starts from) ---
cache = f"{D}/hazy_baseline_per_scene.json"
if os.path.isfile(cache):
    hazy_psnr = json.load(open(cache))
else:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    hazy_psnr = {}
    for scene in sorted(haze):
        vals = []
        for hd in sorted(glob.glob(f"{ROOT}/Test/hazy/{scene}_seq_*")):
            gd = hd.replace("/hazy/", "/gt/")
            if not os.path.isdir(gd):
                continue
            for f in sorted(os.listdir(hd)):
                if not f.lower().endswith(exts):
                    continue
                g = os.path.join(gd, f)
                if not os.path.isfile(g):
                    continue
                h = np.asarray(Image.open(os.path.join(hd, f)).convert("RGB"), np.float64) / 255.0
                t = np.asarray(Image.open(g).convert("RGB"), np.float64) / 255.0
                if h.shape != t.shape:
                    continue
                vals.append(psnr(h, t))
        if vals:
            hazy_psnr[scene] = float(np.mean(vals))
            print(f"  hazy baseline {scene}: {hazy_psnr[scene]:.3f} dB over {len(vals)} frames", flush=True)
    json.dump(hazy_psnr, open(cache, "w"), indent=2)

def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)

RUNS = [("run1", "full_run1.json"), ("run2 BEST", "full_cos_last.json"),
        ("run3 lpips", "full_lpips.json"), ("run3+SWA", "full_lpips_swa.json")]

print("\n" + "=" * 104)
print("PER-SCENE TABLE  (t from dark channel prior; LOWER t = DENSER haze)")
print("=" * 104)
hdr = f"{'scene':<8}{'mean_t':>8}{'std_t':>8}{'hazy dB':>9}"
for lbl, _ in RUNS:
    hdr += f"{lbl:>12}"
hdr += f"{'gain(run2)':>12}"
print(hdr)

evals = {lbl: load_eval(fn)[1] for lbl, fn in RUNS}
scenes = sorted(haze)
rows = {}
for s in scenes:
    hz = hazy_psnr.get(s, float("nan"))
    line = f"{s:<8}{haze[s]['mean_transmission']:>8.4f}{haze[s]['std_transmission']:>8.4f}{hz:>9.2f}"
    per = {}
    for lbl, _ in RUNS:
        v = evals[lbl].get(s, {}).get("psnr_mean", float("nan"))
        per[lbl] = v
        line += f"{v:>12.2f}"
    gain = per["run2 BEST"] - hz
    rows[s] = {"t": haze[s]["mean_transmission"], "std_t": haze[s]["std_transmission"],
               "hazy": hz, "gain": gain, **per}
    line += f"{gain:>12.2f}"
    print(line)

t   = [rows[s]["t"] for s in scenes]
sd  = [rows[s]["std_t"] for s in scenes]
gn  = [rows[s]["gain"] for s in scenes]
ab  = [rows[s]["run2 BEST"] for s in scenes]

print("\n" + "=" * 104)
print("PREDICTION TESTS  (n=6 scenes -> EXPLORATORY; report as correlational, not confirmatory)")
print("=" * 104)
print(f"P1  mean_t  vs RELATIVE gain over hazy : pearson {pearson(t, gn):+.3f}   spearman {spearman(t, gn):+.3f}")
print(f"    prediction: NEGATIVE (denser haze = lower t = larger gain) if the inpainting prior does work")
print(f"P2  std_t   vs RELATIVE gain over hazy : pearson {pearson(sd, gn):+.3f}   spearman {spearman(sd, gn):+.3f}")
print(f"    prediction: POSITIVE (more temporal haze fluctuation = more decorrelated views = larger gain)")
print(f"\n    (context) mean_t vs ABSOLUTE PSNR  : pearson {pearson(t, ab):+.3f}   spearman {spearman(t, ab):+.3f}")
print( "    absolute PSNR is confounded by scene difficulty - this is why P1 uses relative gain")

# --- what the reference selector actually did ---
print("\n" + "=" * 104)
print("REFERENCE SELECTOR - weight by temporal offset (behavioural evidence for temporal fusion)")
print("=" * 104)
for lbl, fn in RUNS:
    d = json.load(open(f"{D}/evaluations/{fn}"))
    rw = d.get("reference_weights_by_offset")
    if not rw:
        print(f"  {lbl:<12} (not recorded)")
        continue
    if isinstance(rw, dict):
        items = sorted(((int(k), float(v)) for k, v in rw.items()), key=lambda kv: kv[0])
    else:
        # list of records: {"offset": int, "mean": float, "std": float, "count": int}
        items = sorted(((int(r["offset"]), float(r["mean"])) for r in rw), key=lambda kv: kv[0])
    tot = sum(v for _, v in items) or 1.0
    raw_max = max(abs(v) for _, v in items) if items else 0.0
    print(f"  {lbl:<12} raw max |mean| = {raw_max:.3e}")
    print(f"  {'':<12} normalised: " + "  ".join(f"{k:+d}:{v/tot:.3f}" for k, v in items))
print("\n  Read the RAW magnitude first. If max |mean| is ~1e-9, the reference selector is")
print("  contributing essentially NOTHING regardless of how the normalised weights look -")
print("  normalising near-zero numbers manufactures a shape that is not there.")
print("  Only if the raw magnitude is non-trivial does the offset distribution mean anything:")
print("  mass on NEAR offsets = smoothing; mass on DISTANT offsets = genuine reference selection.")

json.dump({"per_scene": rows,
           "P1_mean_t_vs_relative_gain": {"pearson": pearson(t, gn), "spearman": spearman(t, gn)},
           "P2_std_t_vs_relative_gain": {"pearson": pearson(sd, gn), "spearman": spearman(sd, gn)},
           "n_scenes": len(scenes),
           "caveat": "n=6 scenes. Exploratory and correlational. Not significance testing."},
          open(f"{D}/occlusion_evidence.json", "w"), indent=2)
print(f"\nwrote {D}/occlusion_evidence.json")
