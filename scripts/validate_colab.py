import argparse

import torch
from torch.utils.data import DataLoader

from src.config import TRDNConfig
from src.dataset import REVIDESequenceDataset
from src.diffusion_adapter import load_diffusion_backbone
from src.flow import load_raft
from src.losses import LossBundle
from src.train import build_optimizer, build_temporal_modules
from src.validate import validate_trdn


def main():
    parser = argparse.ArgumentParser(description="Validate TRDN on REVIDE.")
    parser.add_argument("--dataset-root", default="", help="Optional override for config train/test roots.")
    parser.add_argument("--project-root", default="/content/drive/MyDrive/TRDN_REVIDE")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max-batches", type=int, default=8)
    args = parser.parse_args()

    config = TRDNConfig(project_root=args.project_root, resume_from_checkpoint=args.checkpoint)
    if args.dataset_root:
        config.dataset_root = args.dataset_root
        config.train_root = args.dataset_root
        config.test_root = args.dataset_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = load_diffusion_backbone(config, device)
    temporal_memory, temporal_transformer, reference_selector, conditioning_adapter = build_temporal_modules(
        config, diffusion["unet"].config.cross_attention_dim, device
    )
    if args.checkpoint:
        from accelerate import Accelerator

        accelerator = Accelerator(mixed_precision=config.mixed_precision)
        optimizer = build_optimizer(config, diffusion["unet"], temporal_memory, temporal_transformer, reference_selector, conditioning_adapter)
        if temporal_transformer is not None:
            diffusion["unet"], temporal_memory, temporal_transformer, reference_selector, conditioning_adapter, optimizer = accelerator.prepare(
                diffusion["unet"], temporal_memory, temporal_transformer, reference_selector, conditioning_adapter, optimizer
            )
        else:
            diffusion["unet"], temporal_memory, reference_selector, conditioning_adapter, optimizer = accelerator.prepare(
                diffusion["unet"], temporal_memory, reference_selector, conditioning_adapter, optimizer
            )
        accelerator.load_state(args.checkpoint)
    raft_model = load_raft(device, config.freeze_raft) if config.use_raft_alignment and torch.cuda.is_available() else None
    dataset = REVIDESequenceDataset(
        config.root_for_split(config.val_split),
        split=config.val_split,
        seq_len=config.seq_len,
        crop_size=config.crop_size,
        random_crop=False,
        extensions=config.image_extensions,
        synthetic_if_empty=True,
        train_mode=config.train_mode,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    metrics = validate_trdn(
        loader,
        diffusion,
        temporal_memory,
        temporal_transformer,
        reference_selector,
        conditioning_adapter,
        LossBundle(device),
        device,
        raft_model=raft_model,
        max_batches=args.max_batches,
    )
    print({key: value for key, value in metrics.items() if isinstance(value, float)})


if __name__ == "__main__":
    main()
