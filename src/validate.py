import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .assertions import assert_frames, assert_mask, assert_reference_weights, assert_temporal_memory, assert_warped_references
from .diffusion_adapter import decode_latents_to_images, encode_images_to_latents, get_text_embeddings
from .flow import compute_warped_references_batch
from .metrics import psnr_metric, ssim_metric
from .progress import ProgressReporter
from .seeding import derive_generator

DEFAULT_EVAL_SEED = 1234


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
    seed: Optional[int] = DEFAULT_EVAL_SEED,
    generator: Optional[torch.Generator] = None,
    sample_ids: Optional[List[str]] = None,
    ddim_eta: float = 0.0,
    show_progress: bool = True,
    text_prompt: str = "a clear clean dehazed video frame",
    guidance_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Run the full TRDN pipeline for one batch.

    Deterministic by default: pass `seed` (and optionally `sample_ids`, e.g.
    [f"{clip_name}:{frame_index}"]) to derive a reproducible per-call noise
    generator, or pass an explicit `generator`. DDIM sampling uses eta=0 so the
    only randomness is the initial latent noise -- fixing that makes the whole
    trajectory reproducible.
    """
    if guidance_scale <= 0:
        raise ValueError(f"guidance_scale must be positive, got {guidance_scale}")
    frames = frames.to(device)
    mask = mask.to(device)
    corrupted = corrupted.to(device)
    assert_frames(frames)
    assert_mask(mask, corrupted)
    batch = frames.shape[0]
    scheduler = diffusion["inference_scheduler"]
    scheduler.set_timesteps(num_steps, device=device)
    if generator is None:
        generator = derive_generator(seed if seed is not None else 0, *(sample_ids or ["batch"]), device=device)

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
    text = get_text_embeddings(
        diffusion["tokenizer"],
        diffusion["text_encoder"],
        batch,
        prompt=text_prompt,
    ).to(cond_tokens.dtype)
    encoder_hidden_states = torch.cat([text, cond_tokens], dim=1)
    unconditional_hidden_states = None
    if guidance_scale != 1.0:
        unconditional_text = get_text_embeddings(
            diffusion["tokenizer"],
            diffusion["text_encoder"],
            batch,
            prompt="",
        ).to(cond_tokens.dtype)
        unconditional_hidden_states = torch.cat(
            [unconditional_text, torch.zeros_like(cond_tokens)],
            dim=1,
        )

    latent_shape = (batch, 4, frames.shape[-2] // 8, frames.shape[-1] // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=cond_tokens.dtype) * scheduler.init_noise_sigma
    mask_latent = torch.nn.functional.interpolate(mask.float(), size=latents.shape[-2:], mode="nearest").to(
        device, latents.dtype
    )
    masked_latents = encode_images_to_latents(diffusion["vae"], corrupted).to(latents.dtype)

    progress = ProgressReporter(
        len(scheduler.timesteps),
        "DDIM inference",
        leave=False,
        position=2,
        enabled=show_progress,
    )
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents, mask_latent, masked_latents], dim=1)
        noise_pred = diffusion["unet"](
            model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        if unconditional_hidden_states is not None:
            unconditional = diffusion["unet"](
                model_input,
                timestep,
                encoder_hidden_states=unconditional_hidden_states,
            ).sample
            noise_pred = unconditional + guidance_scale * (noise_pred - unconditional)
        latents = scheduler.step(noise_pred, timestep, latents, eta=ddim_eta, generator=generator).prev_sample
        progress.update(1)
    progress.close()

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
def infer_diffusion_only_batch(
    mask: torch.Tensor,
    corrupted: torch.Tensor,
    diffusion: Dict[str, Any],
    device: str,
    num_steps: int = 30,
    seed: Optional[int] = DEFAULT_EVAL_SEED,
    generator: Optional[torch.Generator] = None,
    sample_ids: Optional[List[str]] = None,
    ddim_eta: float = 0.0,
    show_progress: bool = True,
    text_prompt: str = "a clear clean dehazed video frame",
    guidance_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Diffusion-only baseline: SD inpainting on a single frame, no temporal modules.

    No RAFT, ConvLSTM, temporal transformer, or reference selector are invoked
    -- only the frozen/fine-tuned SD inpainting UNet conditioned on text alone.
    Used to isolate how much of TRDN's performance comes from the temporal
    stack vs. the diffusion backbone alone (see scripts/evaluate_full_test.py).
    """
    if guidance_scale <= 0:
        raise ValueError(f"guidance_scale must be positive, got {guidance_scale}")
    mask = mask.to(device)
    corrupted = corrupted.to(device)
    assert_mask(mask, corrupted)
    batch = corrupted.shape[0]
    scheduler = diffusion["inference_scheduler"]
    scheduler.set_timesteps(num_steps, device=device)
    if generator is None:
        generator = derive_generator(seed if seed is not None else 0, *(sample_ids or ["batch"]), device=device)

    text = get_text_embeddings(
        diffusion["tokenizer"],
        diffusion["text_encoder"],
        batch,
        prompt=text_prompt,
    )
    unconditional_text = (
        get_text_embeddings(
            diffusion["tokenizer"],
            diffusion["text_encoder"],
            batch,
            prompt="",
        )
        if guidance_scale != 1.0
        else None
    )

    latent_shape = (batch, 4, corrupted.shape[-2] // 8, corrupted.shape[-1] // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=text.dtype) * scheduler.init_noise_sigma
    mask_latent = torch.nn.functional.interpolate(mask.float(), size=latents.shape[-2:], mode="nearest").to(
        device, latents.dtype
    )
    masked_latents = encode_images_to_latents(diffusion["vae"], corrupted).to(latents.dtype)

    progress = ProgressReporter(
        len(scheduler.timesteps),
        "Diffusion-only inference",
        leave=False,
        position=2,
        enabled=show_progress,
    )
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents, mask_latent, masked_latents], dim=1)
        noise_pred = diffusion["unet"](
            model_input,
            timestep,
            encoder_hidden_states=text,
        ).sample
        if unconditional_text is not None:
            unconditional = diffusion["unet"](
                model_input,
                timestep,
                encoder_hidden_states=unconditional_text,
            ).sample
            noise_pred = unconditional + guidance_scale * (noise_pred - unconditional)
        latents = scheduler.step(noise_pred, timestep, latents, eta=ddim_eta, generator=generator).prev_sample
        progress.update(1)
    progress.close()

    return {"prediction": decode_latents_to_images(diffusion["vae"], latents)}


@torch.no_grad()
def validate_trdn(
    val_loader,
    diffusion: Dict[str, Any],
    temporal_memory: torch.nn.Module | None,
    temporal_transformer: torch.nn.Module | None,
    reference_selector: torch.nn.Module | None,
    conditioning_adapter: torch.nn.Module | None,
    loss_bundle: torch.nn.Module,
    device: str,
    raft_model: torch.nn.Module | None = None,
    num_samples: int = 32,
    num_steps: int = 30,
    seed: Optional[int] = DEFAULT_EVAL_SEED,
    text_prompt: str = "a clear clean dehazed video frame",
    guidance_scale: float = 1.0,
    diffusion_only: bool = False,
) -> Dict[str, Any]:
    if num_samples <= 0:
        raise ValueError(f"Validation num_samples must be positive, got {num_samples}.")
    if num_steps <= 0:
        raise ValueError(f"Validation num_steps must be positive, got {num_steps}.")
    started = time.perf_counter()
    diffusion["unet"].eval()
    if temporal_memory is not None:
        temporal_memory.eval()
    if temporal_transformer is not None:
        temporal_transformer.eval()
    if reference_selector is not None:
        reference_selector.eval()
    if conditioning_adapter is not None:
        conditioning_adapter.eval()
    psnrs, ssims, lpips_values = [], [], []
    first_output = None
    validation_total = min(len(val_loader.dataset), num_samples)
    progress = ProgressReporter(
        validation_total,
        "Validation",
        leave=False,
        position=1,
    )
    evaluated_samples = 0
    for batch_idx, batch in enumerate(val_loader):
        if evaluated_samples >= num_samples:
            break
        remaining = num_samples - evaluated_samples
        take = min(int(batch["frames"].shape[0]), remaining)
        frames = batch["frames"][:take].to(device)
        target = batch["target_frame"][:take].to(device)
        mask = batch["mask"][:take].to(device)
        corrupted = batch["corrupted_frame"][:take].to(device)
        sequence_names = batch.get("sequence_name", [f"batch{batch_idx}"] * take)
        if isinstance(sequence_names, str):
            sequence_names = [sequence_names]
        sample_ids = [
            f"{sequence_names[index]}:{evaluated_samples + index}"
            for index in range(take)
        ]
        if diffusion_only:
            output = infer_diffusion_only_batch(
                mask,
                corrupted,
                diffusion,
                device,
                num_steps=num_steps,
                seed=seed,
                sample_ids=sample_ids,
                show_progress=False,
                text_prompt=text_prompt,
                guidance_scale=guidance_scale,
            )
        else:
            if temporal_memory is None or reference_selector is None or conditioning_adapter is None:
                raise RuntimeError("Temporal validation requires temporal modules.")
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
                seed=seed,
                sample_ids=sample_ids,
                show_progress=False,
                text_prompt=text_prompt,
                guidance_scale=guidance_scale,
            )
        pred = output["prediction"]
        for sample_index in range(take):
            sample_prediction = pred[sample_index : sample_index + 1]
            sample_target = target[sample_index : sample_index + 1]
            psnrs.append(psnr_metric(sample_prediction[0], sample_target[0]))
            ssims.append(ssim_metric(sample_prediction[0], sample_target[0]))
            lpips_values.append(
                float(
                    loss_bundle.lpips_loss(
                        sample_prediction,
                        sample_target,
                    ).detach().cpu()
                )
            )
        evaluated_samples += take
        if first_output is None:
            first_output = output
        progress.set_postfix({"mean_psnr": f"{np.mean(psnrs):.3f}"})
        progress.update(take)
    progress.close()

    diffusion["unet"].train()
    if temporal_memory is not None:
        temporal_memory.train()
    if temporal_transformer is not None:
        temporal_transformer.train()
    if reference_selector is not None:
        reference_selector.train()
    if conditioning_adapter is not None:
        conditioning_adapter.train()
    return {
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim": float(np.mean(ssims)) if ssims else 0.0,
        "lpips": float(np.mean(lpips_values)) if lpips_values else 0.0,
        "num_samples": evaluated_samples,
        "num_inference_steps": num_steps,
        "seed": seed,
        "wall_clock_seconds": time.perf_counter() - started,
        "unet_forward_passes": (
            evaluated_samples * num_steps * (2 if guidance_scale != 1.0 else 1)
        ),
        "first_output": first_output,
    }
