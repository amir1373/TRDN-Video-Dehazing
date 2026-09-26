"""Build the R1 retrieval index for the REVIDE Train and Test videos (see src/retrieval.py).

usage: build_retrieval_index.py --dataset-root /workspace/datasets/REVIDE_TRDN --output INDEX.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import TRDNConfig  # noqa: E402
from src.dataset import discover_revide_sequences  # noqa: E402
from src.retrieval import build_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    config = TRDNConfig()
    config.override_dataset_root(args.dataset_root)
    videos = {}
    for split in ("train", "test"):
        for sequence in discover_revide_sequences(Path(config.root_for_split(split)), split, config.image_extensions):
            videos[f"{split}/{sequence['name']}"] = [str(p) for p in sequence["hazy_files"]]
    stats = build_index(videos, args.output)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
