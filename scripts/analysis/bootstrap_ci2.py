"""Sequence-level bootstrap CIs over the REVIDE test scenes, for ANY number of runs.

You have 6 test SCENES, not 230 independent windows. Windows inside a sequence are
highly correlated, so a pooled mean over 230 windows badly overstates precision.
Resample whole sequences with replacement instead.

Generalises /root/bootstrap_ci.py (which was hardcoded to run1 vs run2) to auto-discover
every evaluation JSON, and shares ONE set of scene resamples across all models so the
pairwise differences are paired on identical scenes.
"""
import glob, json, os, random, sys

E = "/workspace/trdn_paper_run/evaluations/"
N = 20000
SEED = 1234
BASELINE = os.environ.get("CI_BASELINE", "")   # substring of the filename to compare against

LABELS = {                      # filename -> human label; anything else falls back to the stem
    "full_run1.json":      "run1 30ep",
    "full_cos_last.json":  "run2 50ep",
}

def label_for(path):
    b = os.path.basename(path)
    if b in LABELS:
        return LABELS[b]
    stem = b[:-5]
    for noisy in ("full_", "_last"):
        stem = stem.replace(noisy, "")
    return stem[:22]

def load():
    out = []
    for f in sorted(glob.glob(E + "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        clips = d.get("per_clip")
        if not clips or not isinstance(clips, list):
            continue
        if not all("sequence_name" in c and "num_frames" in c for c in clips):
            continue
        out.append((label_for(f), os.path.basename(f), {c["sequence_name"]: c for c in clips}))
    return out

def wmean(byname, names, key):
    t = sum(byname[n]["num_frames"] for n in names)
    return sum(byname[n]["num_frames"] * byname[n][key] for n in names) / t

ORDER = ["run1", "run2", "run3", "lpips", "valsel", "swa"]

def sort_key(r):
    for i, tag in enumerate(ORDER):
        if tag in r[1].lower() or tag in r[0].lower():
            return (i, r[1])
    return (len(ORDER), r[1])

runs = sorted(load(), key=sort_key)
if not runs:
    sys.exit("no evaluation JSONs with per_clip found in " + E)

common = set.intersection(*[set(r[2]) for r in runs])
if not common:
    sys.exit("no sequences common to all runs")
names = sorted(common)

# ONE shared set of resamples -> every model is scored on identical scene draws,
# which is what makes the pairwise differences properly paired.
random.seed(SEED)
resamples = [[random.choice(names) for _ in names] for _ in range(N)]

print(f"{len(runs)} evaluation(s), {len(names)} common test scenes, {N} bootstrap resamples\n")
print(f"{'model':<24}{'metric':<7}{'point':>9}{'95% CI':>24}{'width':>8}")
curves = {}
for label, fname, byname in runs:
    for key, name in (("psnr_mean", "PSNR"), ("ssim_mean", "SSIM"), ("lpips_mean", "LPIPS")):
        if key not in next(iter(byname.values())):
            continue
        point = wmean(byname, names, key)
        ms = sorted(wmean(byname, s, key) for s in resamples)
        lo, hi = ms[int(.025 * N)], ms[int(.975 * N)]
        curves[(label, key)] = ms
        print(f"{label:<24}{name:<7}{point:>9.4f}   [{lo:8.4f},{hi:8.4f}]{hi-lo:>8.3f}")

# Pairwise paired-bootstrap differences on PSNR.
print("\npairwise PSNR differences (paired on identical scene resamples):")
pairs = []
for i in range(len(runs)):
    for j in range(len(runs)):
        if i >= j:
            continue
        if BASELINE and BASELINE not in runs[i][1] and BASELINE not in runs[j][1]:
            continue
        pairs.append((runs[i], runs[j]))
for (la, _fa, A), (lb, _fb, B) in pairs:
    diffs = sorted(wmean(B, s, "psnr_mean") - wmean(A, s, "psnr_mean") for s in resamples)
    lo, hi = diffs[int(.025 * N)], diffs[int(.975 * N)]
    point = wmean(B, names, "psnr_mean") - wmean(A, names, "psnr_mean")
    verdict = "SIGNIFICANT (CI excludes 0)" if (lo > 0 or hi < 0) else "NOT SIGNIFICANT (CI includes 0)"
    print(f"  {lb} - {la}: {point:+.3f} dB  95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")

if len(pairs) > 1:
    print(f"\n  NOTE: {len(pairs)} pairwise tests. With 6 scenes and multiple comparisons the")
    print("  family-wise false-positive rate is well above 5%. Treat a single barely-excluding")
    print("  CI as suggestive, not conclusive, and say in the thesis how many were tested.")

print("\nper-sequence PSNR (why the CIs are wide):")
hdr = "  " + f"{'sequence':<16}" + "".join(f"{r[0][:11]:>13}" for r in runs)
print(hdr)
for n in sorted(names, key=lambda x: runs[0][2][x]["psnr_mean"]):
    print("  " + f"{n[:14]:<16}" + "".join(f"{r[2][n]['psnr_mean']:>13.2f}" for r in runs))
