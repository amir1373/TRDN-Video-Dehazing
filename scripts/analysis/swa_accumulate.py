"""CPU-only running-average (SWA) over run3's cosine-tail checkpoints.

WHY THIS EXISTS: enable_ema could not be turned on for a warm-started run
(accelerate raises "Number of custom checkpoints ... Found 0, Registered 1").
Weight averaging over the cosine tail is the standard substitute.

WHY IT MUST RUN DURING TRAINING: checkpoint retention is keep_last_n_checkpoints=2,
so every step_* directory is deleted ~2 checkpoints after it appears. There is no
way to do this after the fact.

WHY A RUNNING MEAN AND NOT A POOL: each checkpoint's weights are 3.46 GB. Pooling
10 of them is 35 GB against ~60 GB of remaining quota, and a full volume has already
killed a trainer on this project once. A running mean holds one accumulator and one
incoming checkpoint, and writes 3.46 GB exactly once at the end.

Averages ONLY the model shards (model*.safetensors), never optimizer.bin.
"""
import json, os, re, sys, time
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

CKDIR = Path("/workspace/trdn_paper_run/runs/full_lpips/checkpoints")
OUT = Path("/workspace/trdn_paper_run/swa")
STATE = OUT / "swa_state.json"
MIN_AGE = 90          # seconds; do not read a checkpoint still being written
START_STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 63888
SHARDS = ["model.safetensors", "model_1.safetensors", "model_2.safetensors",
          "model_3.safetensors", "model_4.safetensors"]

OUT.mkdir(parents=True, exist_ok=True)
sums, count, done = {}, 0, []

def log(m):
    print(f"{time.strftime('%F %T')} {m}", flush=True)

def steps_available():
    out = []
    for d in CKDIR.glob("step_*"):
        m = re.match(r"step_(\d+)$", d.name)
        if not m:
            continue
        s = int(m.group(1))
        if s >= START_STEP and s not in done:
            out.append((s, d))
    return sorted(out)

log(f"SWA accumulator start: from step {START_STEP}, target {TARGET}")
stall = 0
while True:
    progressed = False
    for step, d in steps_available():
        try:
            if time.time() - d.stat().st_mtime < MIN_AGE:
                continue
            present = [s for s in SHARDS if (d / s).is_file()]
            if "model.safetensors" not in present:
                continue
            for shard in present:
                t = load_file(str(d / shard), device="cpu")
                acc = sums.setdefault(shard, {})
                for k, v in t.items():
                    fv = v.to(torch.float32)
                    acc[k] = fv if k not in acc else acc[k] + fv
                del t
            count += 1
            done.append(step)
            progressed = True
            log(f"accumulated step_{step:06d} (n={count}, shards={len(present)})")
        except Exception as e:
            log(f"skip step_{step}: {type(e).__name__}: {e}")
    if max(done, default=0) >= TARGET or (not progressed and stall > 40):
        break
    stall = 0 if progressed else stall + 1
    time.sleep(30)

if count == 0:
    log("no checkpoints accumulated; nothing written")
    sys.exit(1)

log(f"averaging {count} checkpoints -> {OUT}")
for shard, acc in sums.items():
    save_file({k: (v / count) for k, v in acc.items()}, str(OUT / shard))
    log(f"wrote {shard}")
STATE.write_text(json.dumps(
    {"num_averaged": count, "steps": sorted(done), "start_step": START_STEP,
     "target": TARGET, "source_run": "full_lpips",
     "note": "SWA over cosine-tail checkpoints; model shards only, optimizer excluded."},
    indent=2))
log(f"SWA COMPLETE: {count} checkpoints averaged, steps {sorted(done)}")
