from pathlib import Path
from typing import Any, Dict

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .config import TRDNConfig
from .dataset import REVIDESequenceDataset
from .diffusion_adapter import load_diffusion_backbone
from .flow import load_raft
from .losses import LossBundle
from .train import build_optimizer, build_temporal_modules
from .validate import infer_dehazed_batch


def load_runtime(config: TRDNConfig, checkpoint_path: str = "") -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = load_diffusion_backbone(config, device=device)
    temporal_memory, temporal_transformer, reference_selector, conditioning_adapter = build_temporal_modules(
        config, diffusion["unet"].config.cross_attention_dim, device
    )
    if checkpoint_path:
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
        accelerator.load_state(checkpoint_path)
    raft_model = load_raft(device, config.freeze_raft) if config.use_raft_alignment and torch.cuda.is_available() else None
    return {
        "device": device,
        "diffusion": diffusion,
        "temporal_memory": temporal_memory,
        "temporal_transformer": temporal_transformer,
        "reference_selector": reference_selector,
        "conditioning_adapter": conditioning_adapter,
        "raft_model": raft_model,
        "loss_bundle": LossBundle(device),
    }


@torch.no_grad()
def run_inference_on_index(config: TRDNConfig, index: int = 0, checkpoint_path: str = "") -> Dict[str, torch.Tensor]:
    paths = config.ensure_dirs()
    runtime = load_runtime(config, checkpoint_path)
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
    sample = dataset[index]
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }
    output = infer_dehazed_batch(
        batch["frames"],
        batch["mask"],
        batch["corrupted_frame"],
        runtime["diffusion"],
        runtime["temporal_memory"],
        runtime["temporal_transformer"],
        runtime["reference_selector"],
        runtime["conditioning_adapter"],
        runtime["device"],
        raft_model=runtime["raft_model"],
        num_steps=config.num_inference_steps,
    )
    save_path = Path(paths["outputs"]) / f"dehazed_index_{index:04d}.png"
    save_image(output["prediction"], save_path)
    output["save_path"] = str(save_path)
    return output
