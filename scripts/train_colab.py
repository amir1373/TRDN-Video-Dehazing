import argparse

from src.config import TRDNConfig
from src.train import train_trdn


def main():
    parser = argparse.ArgumentParser(description="Train TRDN on REVIDE from a Colab runtime.")
    parser.add_argument("--dataset-root", required=True, help="Google Drive path to REVIDE.")
    parser.add_argument("--project-root", default="/content/drive/MyDrive/TRDN_REVIDE")
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--resume-from-checkpoint", default="")
    args = parser.parse_args()

    config = TRDNConfig(
        dataset_root=args.dataset_root,
        project_root=args.project_root,
        max_train_steps=args.max_train_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(train_trdn(config))


if __name__ == "__main__":
    main()
