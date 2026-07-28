import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TRDNConfig
from src.inference import run_inference_on_index


def main():
    parser = argparse.ArgumentParser(description="Run TRDN inference on one REVIDE sequence.")
    parser.add_argument("--dataset-root", default="", help="Optional override for config train/test roots.")
    parser.add_argument("--project-root", default="/content/drive/MyDrive/TRDN_REVIDE")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--allow-mode-mismatch", action="store_true")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    config = TRDNConfig(
        project_root=args.project_root,
        allow_mode_mismatch=args.allow_mode_mismatch,
    )
    if args.dataset_root:
        config.override_dataset_root(args.dataset_root)
    output = run_inference_on_index(config, index=args.index, checkpoint_path=args.checkpoint)
    print("Saved prediction:", output["save_path"])


if __name__ == "__main__":
    main()
