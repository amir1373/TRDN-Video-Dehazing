import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.assertions import assert_frames, assert_mask, assert_reference_weights, assert_temporal_memory, assert_warped_references
from src.config import TRDNConfig
from src.dataset import REVIDESequenceDataset
from src.diffusion_adapter import TemporalConditioningAdapter, load_diffusion_backbone
from src.flow import compute_warped_references_batch, flow_to_rgb, load_raft
from src.convlstm import TemporalMemoryModule
from src.reference_selector import ReferenceSelectionModule
from src.temporal_transformer import TemporalRetrievalTransformer
from src.validate import infer_dehazed_batch


def count_parameters(module: torch.nn.Module) -> dict:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_percent": 100.0 * trainable / max(total, 1)}


def memory_report(device: str) -> dict:
    if not torch.cuda.is_available() or device == "cpu":
        return {"device": device, "cuda_available": False}
    return {
        "device": device,
        "cuda_available": True,
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    parser = argparse.ArgumentParser(description="Run and save a TRDN debug forward pass.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--skip-diffusion", action="store_true", help="Skip Stable Diffusion loading for CPU/local smoke tests.")
    parser.add_argument("--no-raft", action="store_true", help="Use zero-flow alignment for fast shape debugging.")
    args = parser.parse_args()

    config = TRDNConfig()
    try:
        import google.colab  # type: ignore  # noqa: F401

        running_colab = True
    except Exception:
        running_colab = False
    if not running_colab:
        config.project_root = str(Path.cwd())
    paths = config.ensure_dirs()
    debug_dir = Path(paths["debug_outputs"])
    debug_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = config.root_for_split(args.split)
    dataset = REVIDESequenceDataset(
        root,
        split=args.split,
        seq_len=config.seq_len,
        crop_size=config.crop_size,
        random_crop=False,
        extensions=config.image_extensions,
        synthetic_if_empty=True,
        train_mode=config.train_mode,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    frames = batch["frames"].to(device)
    target = batch["target_frame"].to(device)
    mask = batch["mask"].to(device)
    corrupted = batch["corrupted_frame"].to(device)
    assert_frames(frames, seq_len=config.seq_len)
    assert_mask(mask, corrupted)

    save_image(mask, debug_dir / "mask.png")
    save_image(corrupted, debug_dir / "corrupted_input.png")
    save_image(target, debug_dir / "target_clean.png")

    raft_model = None
    if config.use_raft_alignment and torch.cuda.is_available() and not args.no_raft:
        raft_model = load_raft(device, config.freeze_raft)
    warped_refs, flows = compute_warped_references_batch(frames, raft_model)
    assert_warped_references(warped_refs, seq_len=config.seq_len)
    save_image(warped_refs[0, 0], debug_dir / "warped_reference_0.png")
    save_image(flow_to_rgb(flows[0, 0]).unsqueeze(0), debug_dir / "flow_0_rgb.png")

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

    aligned = torch.cat([warped_refs, frames[:, -1:].contiguous()], dim=1)
    memory = temporal_memory(aligned)
    transformer_out = transformer(aligned, memory)
    memory = transformer_out["enhanced_memory"]
    assert_temporal_memory(memory, batch=1)
    ref = reference_selector(warped_refs, memory, prior_logits=transformer_out["reference_prior_logits"])
    assert_reference_weights(ref["weights"], seq_len=config.seq_len)
    tokens = adapter(memory, ref["reference_feature"])

    mem_vis = memory[:, :1]
    mem_vis = (mem_vis - mem_vis.amin(dim=(2, 3), keepdim=True)) / (mem_vis.amax(dim=(2, 3), keepdim=True) - mem_vis.amin(dim=(2, 3), keepdim=True) + 1e-8)
    save_image(mem_vis, debug_dir / "temporal_memory_ch0.png")
    save_image(ref["weights"][:, 0:1], debug_dir / "reference_weight_0.png")
    save_image(ref["weighted_reference"], debug_dir / "weighted_reference.png")

    report = {
        "train_mode": config.train_mode,
        "dataset_root_used": root,
        "sequence_name": batch["sequence_name"][0] if isinstance(batch["sequence_name"], list) else str(batch["sequence_name"]),
        "shapes": {
            "frames": tuple(frames.shape),
            "target": tuple(target.shape),
            "mask": tuple(mask.shape),
            "warped_references": tuple(warped_refs.shape),
            "flows": tuple(flows.shape),
            "temporal_memory": tuple(memory.shape),
            "transformer_tokens": tuple(transformer_out["tokens"].shape),
            "reference_weights": tuple(ref["weights"].shape),
            "conditioning_tokens": tuple(tokens.shape),
        },
        "parameters": {
            "temporal_memory": count_parameters(temporal_memory),
            "temporal_transformer": count_parameters(transformer),
            "reference_selector": count_parameters(reference_selector),
            "conditioning_adapter": count_parameters(adapter),
        },
        "memory": memory_report(device),
        "diffusion_ran": False,
    }

    if not args.skip_diffusion:
        diffusion = load_diffusion_backbone(config, device)
        cross_dim = diffusion["unet"].config.cross_attention_dim
        if cross_dim != 768:
            adapter = TemporalConditioningAdapter(cross_attention_dim=cross_dim).to(device)
        output = infer_dehazed_batch(
            frames,
            mask,
            corrupted,
            diffusion,
            temporal_memory,
            transformer,
            reference_selector,
            adapter,
            device,
            raft_model=raft_model,
            num_steps=2,
        )
        save_image(output["prediction"], debug_dir / "diffusion_prediction.png")
        report["diffusion_ran"] = True
        report["parameters"]["unet"] = count_parameters(diffusion["unet"])

    (debug_dir / "debug_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Debug outputs saved to {debug_dir}")


if __name__ == "__main__":
    main()
