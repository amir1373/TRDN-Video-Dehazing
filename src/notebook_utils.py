from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from .convlstm import TemporalMemoryModule
from .dataset import REVIDESequenceDataset
from .flow import compute_warped_references_batch, load_raft
from .reference_selector import ReferenceSelectionModule
from .temporal_transformer import TemporalRetrievalTransformer
from .diffusion_adapter import TemporalConditioningAdapter


def show_tensor_images(images: list[torch.Tensor], titles: list[str], figsize=(16, 4), save_path: str | Path | None = None) -> None:
    """Notebook visualization helper for tensors in [0, 1]."""
    plt.figure(figsize=figsize)
    for idx, (image, title) in enumerate(zip(images, titles), start=1):
        x = image.detach().float().cpu()
        if x.ndim == 4:
            x = x[0]
        plt.subplot(1, len(images), idx)
        if x.shape[0] == 1:
            plt.imshow(x[0].clamp(0, 1), cmap="magma")
        else:
            plt.imshow(x.clamp(0, 1).permute(1, 2, 0))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()


def make_dataset(config: Any, split: str = "train", crop_size: int | None = None, random_crop: bool = False) -> REVIDESequenceDataset:
    """Create a REVIDE dataset using centralized config paths."""
    return REVIDESequenceDataset(
        config.root_for_split(split),
        split=split,
        seq_len=config.seq_len,
        crop_size=crop_size or config.crop_size,
        random_crop=random_crop,
        extensions=config.image_extensions,
        synthetic_if_empty=True,
        train_mode=config.train_mode,
    )


def get_sample_batch(config: Any, device: str, split: str = "train", crop_size: int | None = None) -> tuple:
    """Return dataset, raw sample, and batchified tensors on device."""
    dataset = make_dataset(config, split=split, crop_size=crop_size, random_crop=False)
    sample = dataset[0]
    batch = {key: (value.unsqueeze(0).to(device) if torch.is_tensor(value) else [value]) for key, value in sample.items()}
    return dataset, sample, batch


def run_temporal_debug(config: Any, batch: dict, device: str, use_raft: bool = False) -> dict:
    """Run the repository temporal stack for notebook visualization/debugging."""
    frames = batch["frames"].to(device)
    raft_model = load_raft(device, config.freeze_raft) if use_raft and torch.cuda.is_available() else None
    warped_refs, flows = compute_warped_references_batch(frames, raft_model)
    current = frames[:, -1]
    temporal_memory = TemporalMemoryModule(hidden_dim=64).to(device)
    transformer = TemporalRetrievalTransformer(
        memory_dim=64,
        token_dim=config.transformer_token_dim,
        num_layers=config.transformer_num_layers,
        num_heads=config.transformer_num_heads,
        pool_size=config.transformer_pool_size,
        max_seq_len=config.seq_len,
    ).to(device)
    reference_selector = ReferenceSelectionModule(num_references=config.seq_len - 1).to(device)
    adapter = TemporalConditioningAdapter(cross_attention_dim=768).to(device)
    aligned = torch.cat([warped_refs, current.unsqueeze(1)], dim=1)
    with torch.no_grad():
        memory = temporal_memory(aligned)
        transformer_out = transformer(aligned, memory)
        memory = transformer_out["enhanced_memory"]
        ref = reference_selector(warped_refs, memory, prior_logits=transformer_out["reference_prior_logits"])
        tokens = adapter(memory, ref["reference_feature"])
    return {
        "frames": frames,
        "warped_refs": warped_refs,
        "flows": flows,
        "memory": memory,
        "transformer": transformer_out,
        "reference": ref,
        "conditioning_tokens": tokens,
        "raft_model": raft_model,
    }
