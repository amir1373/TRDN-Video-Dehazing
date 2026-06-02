import argparse

from src.config import TRDNConfig
from src.train import train_trdn


def main():
    parser = argparse.ArgumentParser(description="Train TRDN on REVIDE from a Colab runtime.")
    parser.add_argument("--dataset-root", default="", help="Optional override for config.train_root/test_root parent.")
    parser.add_argument("--project-root", default="/content/drive/MyDrive/TRDN_REVIDE")
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--resume-from-checkpoint", default="")
    args = parser.parse_args()

    config = TRDNConfig(
        project_root=args.project_root,
        max_train_steps=args.max_train_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    if args.dataset_root:
        config.dataset_root = args.dataset_root
        config.train_root = args.dataset_root
        config.test_root = args.dataset_root
    print(train_trdn(config))


if __name__ == "__main__":
    main()
