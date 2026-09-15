"""Run controlled TRDN seeds and report mean, standard deviation, and 95% CIs."""
from __future__ import annotations
import argparse, json, subprocess, sys
from math import sqrt
from pathlib import Path
import numpy as np

def summarise(values: list[float]) -> dict[str, float]:
    mean = float(np.mean(values)); std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    return {"mean": mean, "std": std, "ci95": critical * std / sqrt(len(values)) if values else 0.0}

def main() -> None:
    parser = argparse.ArgumentParser(description="TRDN multi-seed controlled experiment.")
    parser.add_argument("--dataset-root", required=True); parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 31]); parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=5); parser.add_argument("--temporal-hidden-dim", type=int, default=64)
    parser.add_argument("--disable-transformer", action="store_true"); parser.add_argument("--disable-flow", action="store_true")
    args = parser.parse_args(); root = Path(args.output_root); results = []
    for seed in args.seeds:
        project = root / f"seed_{seed}"; command = [sys.executable, "scripts/train_colab.py", "--dataset-root", args.dataset_root, "--project-root", str(project), "--num-epochs", str(args.num_epochs), "--seed", str(seed), "--seq-len", str(args.seq_len), "--temporal-hidden-dim", str(args.temporal_hidden_dim)]
        if args.disable_transformer: command.append("--no-transformer")
        if args.disable_flow: command.append("--no-raft")
        subprocess.run(command, check=True)
        metadata = json.loads((project / "checkpoints" / "last" / "metadata.json").read_text(encoding="utf-8")); results.append({"seed": seed, "psnr": metadata["best_psnr"], "ssim": metadata["best_ssim"]})
    summary = {"seeds": args.seeds, "seq_len": args.seq_len, "temporal_hidden_dim": args.temporal_hidden_dim, "flow": not args.disable_flow, "spatial_transformer": not args.disable_transformer, "psnr": summarise([x["psnr"] for x in results]), "ssim": summarise([x["ssim"] for x in results]), "per_seed": results}
    root.mkdir(parents=True, exist_ok=True); (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
