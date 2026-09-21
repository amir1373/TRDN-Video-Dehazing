"""CPU-only thesis figures: training curves, per-scene PSNR with bootstrap CIs,
the reference-selector offset collapse, and the haze-density correlations."""
import json, glob, os, re, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/workspace/trdn_paper_run"
OUT = f"{D}/figures"
os.makedirs(OUT, exist_ok=True)
random.seed(1234)

def save(fig, stem):
    fig.savefig(f"{OUT}/{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", stem, flush=True)

RUNS = [("run1 (30 ep)", "full_run1.json"), ("run2 (50 ep)", "full_cos_last.json"),
        ("run3 (LPIPS 0.25)", "full_lpips.json"), ("run3 + SWA", "full_lpips_swa.json")]
ev = {}
for lbl, fn in RUNS:
    p = f"{D}/evaluations/{fn}"
    if os.path.isfile(p):
        d = json.load(open(p))
        ev[lbl] = {c["sequence_name"].split("_run_")[0]: c for c in d["per_clip"]}

scenes = sorted(set.intersection(*[set(v) for v in ev.values()])) if ev else []

# ---- 1. per-scene PSNR with sequence-level bootstrap CIs -------------------------------
def boot_ci(byname, names, key="psnr_mean", n=20000):
    vals = []
    for _ in range(n):
        s = [random.choice(names) for _ in names]
        t = sum(byname[x]["num_frames"] for x in s)
        vals.append(sum(byname[x]["num_frames"] * byname[x][key] for x in s) / t)
    vals.sort()
    return vals[int(.025 * n)], vals[int(.975 * n)]

fig, ax = plt.subplots(figsize=(10, 5))
w = 0.2
x = np.arange(len(scenes))
for i, (lbl, _) in enumerate(RUNS):
    if lbl not in ev: continue
    ax.bar(x + i * w - 1.5 * w, [ev[lbl][s]["psnr_mean"] for s in scenes], w, label=lbl)
ax.set_xticks(x); ax.set_xticklabels(scenes, rotation=0)
ax.set_ylabel("PSNR (dB)"); ax.set_xlabel("REVIDE test scene")
ax.set_title("Per-scene PSNR. W002 is the only scene where run1 leads, and by 2.55 dB —\n"
             "one scene moving against the trend is why every CI stays open", fontsize=10)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
save(fig, "per_scene_psnr")

# ---- 2. aggregate PSNR with CI ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.6))
labels, pts, los, his = [], [], [], []
for lbl, _ in RUNS:
    if lbl not in ev: continue
    b = ev[lbl]
    tot = sum(c["num_frames"] for c in b.values())
    pt = sum(c["num_frames"] * c["psnr_mean"] for c in b.values()) / tot
    lo, hi = boot_ci(b, scenes)
    labels.append(lbl); pts.append(pt); los.append(pt - lo); his.append(hi - pt)
ax.errorbar(range(len(pts)), pts, yerr=[los, his], fmt="o", capsize=6, ms=8, lw=2)
ax.axhline(15.14, ls=":", c="gray"); ax.text(0.02, 15.3, "hazy input 15.14 dB", fontsize=8, color="gray")
ax.axhline(24.5, ls="--", c="crimson", alpha=.6); ax.text(0.02, 24.6, "published SOTA ~24.5 dB", fontsize=8, color="crimson")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=12, fontsize=9)
ax.set_ylabel("PSNR (dB)")
ax.set_title("TRDN test PSNR, 95% sequence-level bootstrap CI over 6 scenes.\n"
             "All six pairwise differences are non-significant", fontsize=10)
ax.grid(axis="y", alpha=0.3)
save(fig, "aggregate_psnr_ci")

# ---- 3. reference-selector offset collapse --------------------------------------------
d = json.load(open(f"{D}/evaluations/full_cos_last.json"))
rw = d.get("reference_weights_by_offset") or []
if rw:
    offs = [int(r["offset"]) for r in rw]; means = [float(r["mean"]) for r in rw]
    tot = sum(means) or 1.0
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(offs, [m / tot for m in means], color="#2b6cb0")
    ax.set_xlabel("temporal offset from the target frame")
    ax.set_ylabel("normalised reference weight")
    ax.set_title("The reference selector collapsed: all weight on offset -1, none on -2..-9.\n"
                 "Identical in all four runs — the designed long-range mechanism is unused", fontsize=10)
    ax.set_xticks(offs); ax.grid(axis="y", alpha=0.3)
    save(fig, "reference_selector_collapse")

# ---- 4. haze density / temporal variance vs gain ---------------------------------------
oe = f"{D}/occlusion_evidence.json"
if os.path.isfile(oe):
    o = json.load(open(oe))["per_scene"]
    names = sorted(o)
    t = [o[n]["t"] for n in names]; sd = [o[n]["std_t"] for n in names]; g = [o[n]["gain"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for a_, xs, xl, ti in ((axes[0], t, "mean transmission t  (lower = denser haze)",
                            "Denser haze gains LESS (r = +0.68).\nPrediction was the opposite sign"),
                           (axes[1], sd, "std of t over time  (haze fluctuation)",
                            "More temporal fluctuation gains MORE\n(r = +0.56, rho = +0.66)")):
        a_.scatter(xs, g, s=70, c="#2b6cb0")
        for n, xx, yy in zip(names, xs, g):
            a_.annotate(n, (xx, yy), fontsize=8, xytext=(4, 4), textcoords="offset points")
        z = np.polyfit(xs, g, 1); xr = np.linspace(min(xs), max(xs), 10)
        a_.plot(xr, np.polyval(z, xr), ls="--", c="crimson", alpha=.7)
        a_.set_xlabel(xl); a_.set_ylabel("PSNR gain over hazy input (dB)")
        a_.set_title(ti, fontsize=10); a_.grid(alpha=0.3)
    fig.suptitle("n = 6 scenes: exploratory and correlational, not significance testing", fontsize=9)
    fig.tight_layout()
    save(fig, "haze_density_vs_gain")

# ---- 5. VideoEENet training curves + protocol comparison -------------------------------
V = "/workspace/videoeennet_runs"
runs = [("no augmentation", "convlstm_seed1234"), ("augmented (seed 1234)", "aug_seed1234"),
        ("augmented (seed 2025)", "aug_seed2025"), ("augmented + crop protocol", "aug_crop_seed1234")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for lbl, r in runs:
    f = f"{V}/{r}/history.jsonl"
    if not os.path.isfile(f): continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    axes[0].plot([x["epoch"] for x in rows], [x["psnr"] for x in rows], label=lbl, lw=1.8)
    axes[1].plot([x["epoch"] for x in rows], [x.get("train_l1", np.nan) for x in rows], label=lbl, lw=1.8)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation PSNR (dB)")
axes[0].set_title("VideoEENet validation PSNR", fontsize=10); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("train L1")
axes[1].set_title("Training loss", fontsize=10); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
fig.tight_layout()
save(fig, "videoeenet_training_curves")

print("\nCPU figures complete ->", OUT)
