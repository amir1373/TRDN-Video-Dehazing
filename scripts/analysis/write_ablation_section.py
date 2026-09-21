"""Generate Chapter 3's temporal-ablation section from the measured results and splice it in.

Runs after the watchdog downloads the variant evaluations. Reads the per-clip JSONs, computes
paired sequence-level bootstrap CIs against the ConvLSTM configuration, and writes prose that
states what the intervals actually support — not what the point estimates suggest.
"""
import json, os, random, sys, re

EX = "/home/amir-fx507/thesis_export"
CH3 = "/home/amir-fx507/thesis_work/thesis/CH3_VIDEOEENET.md"
random.seed(1234)

RUNS = [
    ("ConvLSTM + attention", "veenet_aug1234_RESIZE.json"),
    ("Spatial transformer",  "veenet_spatial_transformer_RESIZE.json"),
    ("Hybrid",               "veenet_hybrid_RESIZE.json"),
]

def load(fn):
    p = os.path.join(EX, "trdn", "evaluations", fn)
    if not os.path.isfile(p):
        p = os.path.join(EX, "evaluations", fn)
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    return {c["sequence_name"].split("_run_")[0]: c for c in d["per_clip"]}

data = [(lbl, load(fn)) for lbl, fn in RUNS]
have = [(l, d) for l, d in data if d]
if len(have) < 2:
    sys.exit("not enough variant evaluations yet; nothing written")

names = sorted(set.intersection(*[set(d) for _, d in have]))
N = 20000
resamples = [[random.choice(names) for _ in names] for _ in range(N)]

def wmean(d, ns):
    t = sum(d[n]["num_frames"] for n in ns)
    return sum(d[n]["num_frames"] * d[n]["psnr_mean"] for n in ns) / t

def ssim_mean(d, ns):
    t = sum(d[n]["num_frames"] for n in ns)
    return sum(d[n]["num_frames"] * d[n]["ssim_mean"] for n in ns) / t

rows = []
for lbl, d in have:
    pt = wmean(d, names)
    ms = sorted(wmean(d, s) for s in resamples)
    rows.append((lbl, pt, ms[int(.025*N)], ms[int(.975*N)], ssim_mean(d, names)))

base_lbl, base_d = have[0]
diffs = []
for lbl, d in have[1:]:
    ds = sorted(wmean(d, s) - wmean(base_d, s) for s in resamples)
    pt = wmean(d, names) - wmean(base_d, names)
    lo, hi = ds[int(.025*N)], ds[int(.975*N)]
    diffs.append((lbl, pt, lo, hi, (lo > 0 or hi < 0)))

best = max(rows, key=lambda r: r[1])
any_sig = any(d[4] for d in diffs)

L = []
L.append("## 3.9 Temporal-stage ablation")
L.append("")
L.append("The backbone, augmentation bundle, schedule, split, seed and protocol are held fixed;")
L.append("only the temporal stage changes. This isolates the contribution of the recurrent,")
L.append("attention-based and hybrid designs to the one component that differs between them.")
L.append("")
L.append("| temporal stage | test PSNR | 95 % CI | test SSIM |")
L.append("|---|---|---|---|")
for lbl, pt, lo, hi, ss in rows:
    star = "**" if lbl == best[0] else ""
    L.append(f"| {star}{lbl}{star} | {star}{pt:.4f}{star} | [{lo:.2f}, {hi:.2f}] | {ss:.4f} |")
L.append("")
L.append(f"Paired sequence-level bootstrap against {base_lbl}, identical scene resamples within")
L.append("each pair:")
L.append("")
L.append("| comparison | Δ PSNR | 95 % CI | verdict |")
L.append("|---|---|---|---|")
for lbl, pt, lo, hi, sig in diffs:
    L.append(f"| {lbl} − {base_lbl} | {pt:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
             f"{'**significant**' if sig else 'not significant'} |")
L.append("")
if any_sig:
    L.append(f"At least one temporal stage is separable from {base_lbl} at six test scenes, which")
    L.append("is unusual in this thesis and worth treating as a genuine finding rather than a")
    L.append("point estimate. The per-scene table in Appendix B shows where the difference arises.")
else:
    L.append(f"No temporal stage is distinguishable from {base_lbl}. The point estimates differ by")
    L.append(f"up to {max(abs(d[1]) for d in diffs):.2f} dB, but every interval spans zero, so the data do not")
    L.append("support a preference among the three designs. This is an absence of evidence rather")
    L.append("than evidence that the designs perform alike; with six test scenes the comparison is")
    L.append("underpowered, and no equivalence test was performed.")
L.append("")
L.append("Two variants from the originally planned study are absent. Flow-guided attention, in")
L.append("both pre- and post-alignment forms, requires integrating optical-flow warping into the")
L.append("temporal stage; that integration was attempted earlier in the project and abandoned")
L.append("after it could not be made to train stably within the time available. The ablation")
L.append("reported here therefore covers the three temporal designs the codebase supports, and")
L.append("flow-guided temporal fusion for this backbone remains open.")
L.append("")

section = "\n".join(L)
s = open(CH3).read()
s = re.sub(r"\n## 3\.9 Temporal-stage ablation.*?(?=\n## |\Z)", "\n", s, flags=re.S)
anchor = "## 3.8 Limitations"
assert anchor in s, "anchor not found in CH3"
s = s.replace(anchor, section + anchor, 1)
open(CH3, "w").write(s)

print("=== section written into CH3 ===")
print(section)
