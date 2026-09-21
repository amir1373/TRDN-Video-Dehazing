"""Per-scene haze density + temporal haze variance on the REVIDE TEST scenes.

WHY: TRDN is motivated as reconstructing scene content hidden behind occluding haze.
That rationale predicts two things, and this script tests both correlationally:
  (1) scenes with LOWER transmission t (denser haze) should show a LARGER relative gain
      over the hazy input, because that is where single-frame inversion is ill-conditioned
      and a learned prior has something to contribute;
  (2) scenes with HIGHER frame-to-frame variance in t should gain more, because fluctuating
      haze makes successive frames more decorrelated observations of the same scene point,
      which is exactly what temporal fusion can exploit.

t is estimated with the dark channel prior (He et al. 2009):
  t(x) = 1 - omega * min_c( min_{y in patch(x)} I^c(y) / A^c )
n = 6 scenes, so this is EXPLORATORY. Report correlations, never p-values as if confirmatory.
"""
import json, os, sys, glob
import numpy as np
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/datasets/REVIDE_sequences"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/trdn_paper_run/haze_analysis.json"
MAX_FRAMES = 40          # per scene; enough for a stable mean and a variance estimate
PATCH = 15
OMEGA = 0.95

def dark_channel(img, patch=PATCH):
    m = img.min(axis=2)
    pad = patch // 2
    padded = np.pad(m, pad, mode="edge")
    out = np.empty_like(m)
    for i in range(m.shape[0]):
        out[i] = padded[i:i+patch, :].min(axis=0)[:m.shape[1]]
    padded2 = np.pad(out, ((0, 0), (pad, pad)), mode="edge")
    for j in range(m.shape[1]):
        out[:, j] = padded2[:, j:j+patch].min(axis=1)
    return out

def atmospheric_light(img, dc):
    flat = dc.ravel()
    n = max(1, int(flat.size * 0.001))
    idx = np.argpartition(flat, -n)[-n:]
    return img.reshape(-1, 3)[idx].max(axis=0)

def transmission(img):
    dc = dark_channel(img)
    A = np.maximum(atmospheric_light(img, dc), 1e-3)
    norm = img / A[None, None, :]
    t = 1.0 - OMEGA * dark_channel(norm)
    return float(np.clip(t, 0.0, 1.0).mean())

def find_test_scenes(root):
    """REVIDE test is laid out Test/hazy/<SCENE>_seq_<NNN>/. The evaluation reports ONE row
    per SCENE (e.g. C005_run_000), so group the per-sequence dirs by their scene prefix."""
    for split in ("Test", "test", "TEST"):
        p = os.path.join(root, split)
        if not os.path.isdir(p):
            continue
        hz = os.path.join(p, "hazy")
        base = hz if os.path.isdir(hz) else p
        dirs = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
        groups = {}
        for d in dirs:
            scene = os.path.basename(d).split("_seq_")[0].split("_run_")[0]
            groups.setdefault(scene, []).append(d)
        return groups
    return {}

groups = find_test_scenes(ROOT)
if not groups:
    sys.exit("no test scenes found under " + ROOT)

results = {}
for name, dirs in sorted(groups.items()):
    # REVIDE ships uppercase .JPG; match extensions case-insensitively.
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    frames = []
    for d in dirs:
        for f in os.listdir(d):
            if f.lower().endswith(exts):
                frames.append(os.path.join(d, f))
    frames = sorted(frames)
    if not frames:
        continue
    step = max(1, len(frames) // MAX_FRAMES)
    picked = frames[::step][:MAX_FRAMES]
    ts = []
    for f in picked:
        im = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64) / 255.0
        if max(im.shape[:2]) > 640:                    # downscale for speed; t is a low-freq statistic
            s = 640.0 / max(im.shape[:2])
            im = np.asarray(Image.fromarray((im*255).astype(np.uint8)).resize(
                (int(im.shape[1]*s), int(im.shape[0]*s))), dtype=np.float64) / 255.0
        ts.append(transmission(im))
    results[name] = {
        "n_frames_sampled": len(ts),
        "mean_transmission": float(np.mean(ts)),   # LOWER = denser haze
        "std_transmission": float(np.std(ts)),     # temporal fluctuation of haze
        "min_transmission": float(np.min(ts)),
        "max_transmission": float(np.max(ts)),
    }
    print(f"{name:<18} mean_t={np.mean(ts):.4f}  std_t={np.std(ts):.4f}  "
          f"range=[{np.min(ts):.3f},{np.max(ts):.3f}]  n={len(ts)}", flush=True)

json.dump(results, open(OUT, "w"), indent=2)
print("\nwrote " + OUT)
