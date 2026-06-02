from typing import Any, Dict

import numpy as np
import torch
from tqdm.auto import tqdm

from .assertions import assert_frames, assert_mask, assert_reference_weights, assert_temporal_memory, assert_warped_references
from .diffusion_adapter import decode_latents_to_images, encode_images_to_latents, get_text_embeddings
from .flow import compute_warped_references_batch
from .metrics import psnr_metric, ssim_metric


@torch.no_grad()
def infer_dehazed_batch(
    frames: torch.Tensor,
    mask: torch.Tensor,
    corrupted: torch.Tensor,
    diffusion: Dict[str, Any],
    temporal_memory: torch.nn.Module,
    temporal_transformer: torch.nn.Module | None,
    reference_selector: torch.nn.Module,
    conditioning_adapter: torch.nn.Module,
    device: str,
    raft_model: torch.nn.Module | None = None,
    num_steps: int = 30,
) -> Dict[str, torch.Tensor]:
    frames = frames.to(device)
    mask = mask.to(device)
    corrupted = corrupted.to(device)
    assert_frames(frames)
    assert_mask(mask, corrupted)
    batch = frames.shape[0]
    scheduler = diffusion["inference_scheduler"]
    scheduler.set_timesteps(num_steps, device=device)

    warped_refs, flows = compute_warped_references_batch(frames, raft_model)
    assert_warped_references(warped_refs, seq_len=frames.shape[1])
    current = frames[:, -1]
    aligned_frames = torch.cat([warped_refs, current.unsqueeze(1)], dim=1)
    memory = temporal_memory(aligned_frames)
    prior_logits = None
    transformer_tokens = None
    if temporal_transformer is not None:
        transformer_out = temporal_transformer(aligned_frames, memory)
        memory = transformer_out["enhanced_memory"]
        prior_logits = transformer_out["reference_prior_logits"]
        transformer_tokens = transformer_out["tokens"]
    assert_temporal_memory(memory, batch=batch)
    ref = reference_selector(warped_refs, memory, prior_logits=prior_logits)
    assert_reference_weights(ref["weights"], seq_len=frames.shape[1])
    cond_tokens = conditioning_adapter(memory, ref["reference_feature"])
    text = get_text_embeddings(diffusion["tokenizer"], diffusion["text_encoder"], batch).to(cond_tokens.dtype)
    encoder_hidden_states = torch.cat([text, cond_tokens], dim=1)

    latent_shape = (batch, 4, frames.shape[-2] // 8, frames.shape[-1] // 8)
    latents = torch.randn(latent_shape, device=device, dtype=cond_tokens.dtype) * scheduler.init_noise_sigma
    mask_latent = torch.nn.functional.interpolate(mask.float(), size=latents.shape[-2:], mode="nearest").to(
        device, latents.dtype
    )
    masked_latents = encode_images_to_latents(diffusion["vae"], corrupted).to(latents.dtype)

    for timestep in tqdm(scheduler.timesteps, desc="DDIM inference", leave=False):
        model_input = torch.cat([latents, mask_latent, masked_latents], dim=1)
        noise_pred = diffusion["unet"](model_input, timestep, encoder_hidden_states=encoder_hidden_states).sample
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return {
        "prediction": decode_latents_to_images(diffusion["vae"], latents),
        "warped_refs": warped_refs,
        "flows": flows,
        "reference_weights": ref["weights"],
        "weighted_reference": ref["weighted_reference"],
        "memory": memory,
        "transformer_tokens": transformer_tokens,
    }


@torch.no_grad()
def validate_trdn(
    val_loader,
    diffusion: Dict[str, Any],
    temporal_memory: torch.nn.Module,
    temporal_transformer: torch.nn.Module | None,
    reference_selector: torch.nn.Module,
    conditioning_adapter: torch.nn.Module,
    loss_bundle: torch.nn.Module,
    device: str,
    raft_model: torch.nn.Module | None = None,
    max_batches: int = 8,
    num_steps: int = 10,
) -> Dict[str, float]:
    diffusion["unet"].eval()
    temporal_memory.eval()
    if temporal_transformer is not None:
        temporal_transformer.eval()
    reference_selector.eval()
    conditioning_adapter.eval()
    psnrs, ssims, lpips_values = [], [], []
    first_output = None
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
        if batch_idx >= max_batches:
            break
        frames = batch["frames"].to(device)
        target = batch["target_frame"].to(device)
        mask = batch["mask"].to(device)
        corrupted = batch["corrupted_frame"].to(device)
        output = infer_dehazed_batch(
            frames,
            mask,
            corrupted,
            diffusion,
            temporal_memory,
            temporal_transformer,
            reference_selector,
            conditioning_adapter,
            device,
            raft_model=raft_model,
            num_steps=num_steps,
        )
        pred = output["prediction"]
        psnrs.append(psnr_metric(pred[0], target[0]))
        ssims.append(ssim_metric(pred[0], target[0]))
        lpips_values.append(float(loss_bundle.lpips_loss(pred, target).detach().cpu()))
        if first_output is None:
            first_output = output

    diffusion["unet"].train()
    temporal_memory.train()
    if temporal_transformer is not None:
        temporal_transformer.train()
    reference_selector.train()
    conditioning_adapter.train()
    return {
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim": float(np.mean(ssims)) if ssims else 0.0,
        "lpips": float(np.mean(lpips_values)) if lpips_values else 0.0,
        "first_output": first_output,
    }
