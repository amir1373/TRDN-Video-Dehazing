import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TRDNConfig
from .convlstm import TemporalMemoryModule
from .dataset import REVIDESequenceDataset
from .diffusion_adapter import estimate_x0_from_epsilon, get_text_embeddings, prepare_inpainting_inputs, decode_latents_to_images, encode_images_to_latents
from .flow import compute_warped_references_batch, load_raft
from .losses import LossBundle, weighted_total_loss
from .reference_selector import ReferenceSelectionModule
from .diffusion_adapter import TemporalConditioningAdapter, load_diffusion_backbone
from .validate import validate_trdn


def make_datasets(config: TRDNConfig) -> Tuple[REVIDESequenceDataset, REVIDESequenceDataset]:
    train_dataset = REVIDESequenceDataset(
        config.dataset_root,
        split=config.train_split,
        seq_len=config.seq_len,
        crop_size=config.crop_size,
        random_crop=True,
        extensions=config.image_extensions,
        synthetic_if_empty=True,
    )
    val_dataset = REVIDESequenceDataset(
        config.dataset_root,
        split=config.val_split,
        seq_len=config.seq_len,
        crop_size=config.crop_size,
        random_crop=False,
        extensions=config.image_extensions,
        synthetic_if_empty=True,
    )
    return train_dataset, val_dataset


def make_dataloaders(config: TRDNConfig) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = make_datasets(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def build_temporal_modules(config: TRDNConfig, cross_attention_dim: int, device: str) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    temporal_memory = TemporalMemoryModule(hidden_dim=64).to(device)
    reference_selector = ReferenceSelectionModule(num_references=config.seq_len - 1).to(device)
    conditioning_adapter = TemporalConditioningAdapter(cross_attention_dim=cross_attention_dim, num_tokens=16).to(device)
    return temporal_memory, reference_selector, conditioning_adapter


def build_optimizer(config: TRDNConfig, unet, temporal_memory, reference_selector, conditioning_adapter):
    groups = []
    if config.train_unet:
        groups.append({"params": [p for p in unet.parameters() if p.requires_grad], "lr": config.learning_rate})
    if config.train_temporal_modules:
        temporal_params = list(temporal_memory.parameters()) + list(reference_selector.parameters()) + list(conditioning_adapter.parameters())
        groups.append({"params": temporal_params, "lr": config.temporal_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def save_checkpoint(accelerator: Accelerator, checkpoint_dir: Path, step: int, best_psnr: float, best_ssim: float, name: str | None = None) -> None:
    out_dir = checkpoint_dir / (name or f"step_{step:06d}")
    accelerator.save_state(str(out_dir))
    if accelerator.is_main_process:
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump({"step": step, "best_psnr": best_psnr, "best_ssim": best_ssim, "time": time.time()}, handle, indent=2)


def train_trdn(config: TRDNConfig) -> Dict[str, float]:
    paths = config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=str(paths["logs"]),
    )
    accelerator.init_trackers("TRDN_REVIDE", config=config.to_dict())

    diffusion = load_diffusion_backbone(config, device=device)
    temporal_memory, reference_selector, conditioning_adapter = build_temporal_modules(
        config, diffusion["unet"].config.cross_attention_dim, device
    )
    loss_bundle = LossBundle(device=device)
    optimizer = build_optimizer(config, diffusion["unet"], temporal_memory, reference_selector, conditioning_adapter)
    train_loader, val_loader = make_dataloaders(config)
    raft_model = load_raft(device, config.freeze_raft) if config.use_raft_alignment and torch.cuda.is_available() else None

    diffusion["unet"], temporal_memory, reference_selector, conditioning_adapter, optimizer, train_loader = accelerator.prepare(
        diffusion["unet"], temporal_memory, reference_selector, conditioning_adapter, optimizer, train_loader
    )
    diffusion["vae"].to(accelerator.device)
    diffusion["text_encoder"].to(accelerator.device)
    if raft_model is not None:
        raft_model.to(accelerator.device).eval()

    global_step, best_psnr, best_ssim = 0, -1.0, -1.0
    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        metadata_path = Path(config.resume_from_checkpoint) / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            global_step = int(metadata.get("step", 0))
            best_psnr = float(metadata.get("best_psnr", -1.0))
            best_ssim = float(metadata.get("best_ssim", -1.0))

    progress = tqdm(total=config.max_train_steps, initial=global_step, disable=not accelerator.is_main_process, desc="Training TRDN")
    for _epoch in range(config.num_epochs):
        for batch in train_loader:
            if global_step >= config.max_train_steps:
                break
            with accelerator.accumulate(diffusion["unet"]):
                frames = batch["frames"].to(accelerator.device, non_blocking=True)
                target = batch["target_frame"].to(accelerator.device, non_blocking=True)
                mask = batch["mask"].to(accelerator.device, non_blocking=True)
                corrupted = batch["corrupted_frame"].to(accelerator.device, non_blocking=True)
                current = batch["current_frame"].to(accelerator.device, non_blocking=True)

                with torch.no_grad():
                    warped_refs, _flows = compute_warped_references_batch(frames, raft_model)

                memory = temporal_memory(torch.cat([warped_refs, current.unsqueeze(1)], dim=1))
                ref = reference_selector(warped_refs, memory)
                cond_tokens = conditioning_adapter(memory, ref["reference_feature"])
                text = get_text_embeddings(diffusion["tokenizer"], diffusion["text_encoder"], frames.shape[0]).to(cond_tokens.dtype)
                encoder_hidden_states = torch.cat([text, cond_tokens], dim=1)

                with torch.no_grad():
                    latents = encode_images_to_latents(diffusion["vae"], target)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, diffusion["noise_scheduler"].config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = diffusion["noise_scheduler"].add_noise(latents, noise, timesteps)
                model_input = prepare_inpainting_inputs(diffusion["vae"], noisy_latents, mask, corrupted)
                noise_pred = diffusion["unet"](model_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())
                pred_x0 = estimate_x0_from_epsilon(diffusion["noise_scheduler"], noisy_latents, timesteps, noise_pred)
                pred_img = decode_latents_to_images(diffusion["vae"], pred_x0)
                parts = {
                    "diffusion": diffusion_loss,
                    "l1": F.l1_loss(pred_img, target),
                    "lpips": loss_bundle.lpips_loss(pred_img, target),
                    "temporal": loss_bundle.temporal_consistency_loss(pred_img, warped_refs, ref["weights"]),
                    "flow": loss_bundle.flow_consistency_loss(warped_refs, current, ref["weights"]),
                    "reference": loss_bundle.reference_preservation_loss(pred_img, ref["weighted_reference"], mask),
                }
                total_loss = weighted_total_loss(config, parts)
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(diffusion["unet"].parameters())
                        + list(temporal_memory.parameters())
                        + list(reference_selector.parameters())
                        + list(conditioning_adapter.parameters()),
                        config.max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and global_step % config.log_every == 0:
                logs = {f"train/{key}_loss": float(value.detach().cpu()) for key, value in parts.items()}
                logs["train/total_loss"] = float(total_loss.detach().cpu())
                accelerator.log(logs, step=global_step)
                progress.set_postfix({"loss": logs["train/total_loss"]})

            if global_step > 0 and global_step % config.validate_every == 0:
                metrics = validate_trdn(
                    val_loader,
                    diffusion,
                    temporal_memory,
                    reference_selector,
                    conditioning_adapter,
                    loss_bundle,
                    str(accelerator.device),
                    raft_model=raft_model,
                    max_batches=4,
                    num_steps=min(10, config.num_inference_steps),
                )
                scalar_metrics = {key: value for key, value in metrics.items() if isinstance(value, float)}
                accelerator.log({f"val/{key}": value for key, value in scalar_metrics.items()}, step=global_step)
                if scalar_metrics["psnr"] > best_psnr:
                    best_psnr = scalar_metrics["psnr"]
                    save_checkpoint(accelerator, paths["checkpoints"], global_step, best_psnr, best_ssim, "best_psnr")
                if scalar_metrics["ssim"] > best_ssim:
                    best_ssim = scalar_metrics["ssim"]
                    save_checkpoint(accelerator, paths["checkpoints"], global_step, best_psnr, best_ssim, "best_ssim")

            if global_step > 0 and global_step % config.checkpoint_every == 0:
                save_checkpoint(accelerator, paths["checkpoints"], global_step, best_psnr, best_ssim)

            global_step += 1
            progress.update(1)
        if global_step >= config.max_train_steps:
            break

    save_checkpoint(accelerator, paths["checkpoints"], global_step, best_psnr, best_ssim, "last")
    accelerator.end_training()
    progress.close()
    return {"step": float(global_step), "best_psnr": best_psnr, "best_ssim": best_ssim}


def dry_run_shape_test(seq_len: int = 10, image_size: int = 64, batch_size: int = 1) -> Dict[str, tuple]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = torch.rand(batch_size, seq_len, 3, image_size, image_size, device=device)
    mask = torch.rand(batch_size, 1, image_size, image_size, device=device)
    warped_refs, flows = compute_warped_references_batch(frames, raft_model=None)
    memory_module = TemporalMemoryModule(hidden_dim=64).to(device)
    selector = ReferenceSelectionModule(num_references=seq_len - 1).to(device)
    adapter = TemporalConditioningAdapter(cross_attention_dim=768, num_tokens=16).to(device)
    memory = memory_module(torch.cat([warped_refs, frames[:, -1:].contiguous()], dim=1))
    ref = selector(warped_refs, memory)
    tokens = adapter(memory, ref["reference_feature"])
    return {
        "frames": tuple(frames.shape),
        "current_hazy": tuple(frames[:, -1].shape),
        "target_clean": tuple(frames[:, -1].shape),
        "mask": tuple(mask.shape),
        "warped_references": tuple(warped_refs.shape),
        "flows": tuple(flows.shape),
        "temporal_memory": tuple(memory.shape),
        "reference_weights": tuple(ref["weights"].shape),
        "conditioning_tokens": tuple(tokens.shape),
    }
