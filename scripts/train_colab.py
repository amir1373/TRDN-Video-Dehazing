import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TRDNConfig
from src.presets import apply_numerics_preset
from src.train import train_trdn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TRDN on REVIDE from a Colab runtime.")
    parser.add_argument("--dataset-root", default="", help="Optional override for config.train_root/test_root parent.")
    parser.add_argument("--project-root", default="/content/drive/MyDrive/TRDN_REVIDE")
    parser.add_argument("--max-train-steps", type=int, default=0, help="0 means train for --num-epochs without a step cap.")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--allow-mode-mismatch", action="store_true")
    parser.add_argument("--allow-output-collision", action="store_true")
    parser.add_argument("--train-mode", default="dehaze", choices=["dehaze", "reconstruct_synthetic"])
    parser.add_argument("--mask-mode", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mixed-precision", default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=3)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--preset", default="", help="Filled numerics YAML preset.")
    parser.add_argument("--enable-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "warmup_cosine"],
        default="constant",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--enable-linear-lr-scaling", action="store_true")
    parser.add_argument("--lr-reference-batch-size", type=int, default=1)
    parser.add_argument("--validation-num-samples", type=int, default=32)
    parser.add_argument("--validation-num-steps", type=int, default=30)
    parser.add_argument("--validation-seed", type=int, default=1234)
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=["psnr", "ssim"],
        default="psnr",
    )
    parser.add_argument("--enable-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--text-prompt",
        default="a clear clean dehazed video frame",
    )
    return parser


def main():
    args = build_parser().parse_args()

    config = TRDNConfig(
        project_root=args.project_root,
        max_train_steps=args.max_train_steps,
        num_epochs=args.num_epochs,
        resume_from_checkpoint=args.resume_from_checkpoint,
        allow_mode_mismatch=args.allow_mode_mismatch,
        allow_output_collision=args.allow_output_collision,
        train_mode=args.train_mode,
        mask_mode=args.mask_mode,
        seed=args.seed,
        seq_len=args.seq_len,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        run_name=args.run_name,
        enable_ema=args.enable_ema,
        ema_decay=args.ema_decay,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        enable_linear_lr_scaling=args.enable_linear_lr_scaling,
        lr_reference_batch_size=args.lr_reference_batch_size,
        validation_num_samples=args.validation_num_samples,
        validation_num_inference_steps=args.validation_num_steps,
        validation_seed=args.validation_seed,
        checkpoint_selection_metric=args.checkpoint_selection_metric,
        enable_early_stopping=args.enable_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        guidance_scale=args.guidance_scale,
        text_prompt=args.text_prompt,
    )
    if args.preset:
        apply_numerics_preset(config, args.preset)
    if args.dataset_root:
        config.override_dataset_root(args.dataset_root)
    print(train_trdn(config))


if __name__ == "__main__":
    main()
