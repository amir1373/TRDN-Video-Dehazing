import argparse

from src.config import TRDNConfig
from src.inference import run_inference_on_index


def main():
    parser = argparse.ArgumentParser(description="Run TRDN inference on one REVIDE sequence.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--project-root", default="/content/drive/MyDrive/TRDN_REVIDE")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    config = TRDNConfig(dataset_root=args.dataset_root, project_root=args.project_root)
    output = run_inference_on_index(config, index=args.index, checkpoint_path=args.checkpoint)
    print("Saved prediction:", output["save_path"])


if __name__ == "__main__":
    main()
