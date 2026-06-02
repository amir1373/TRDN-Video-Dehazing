import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def unnormalize_to_zero_to_one(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


class TemporalConditioningAdapter(nn.Module):
    """Project dense temporal/reference features to Stable Diffusion cross-attention tokens."""

    def __init__(
        self,
        memory_dim: int = 64,
        reference_dim: int = 64,
        cross_attention_dim: int = 768,
        adapter_dim: int = 256,
        num_tokens: int = 16,
    ):
        super().__init__()
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError("num_tokens must be a perfect square")
        self.encoder = nn.Sequential(
            nn.Conv2d(memory_dim + reference_dim, adapter_dim, 3, padding=1),
            nn.GroupNorm(16, adapter_dim),
            nn.SiLU(),
            nn.Conv2d(adapter_dim, adapter_dim, 3, padding=1),
            nn.GroupNorm(16, adapter_dim),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((side, side))
        self.proj = nn.Sequential(nn.LayerNorm(adapter_dim), nn.Linear(adapter_dim, cross_attention_dim))
        self.token_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, temporal_memory: torch.Tensor, reference_feature: torch.Tensor) -> torch.Tensor:
        x = self.encoder(torch.cat([temporal_memory, reference_feature], dim=1))
        tokens = self.pool(x).flatten(2).transpose(1, 2)
        return self.proj(tokens) * self.token_scale


def load_diffusion_backbone(config: Any, device: str = "cuda") -> Dict[str, Any]:
    from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    dtype = torch.float16 if config.mixed_precision == "fp16" and torch.cuda.is_available() else torch.float32
    tokenizer = CLIPTokenizer.from_pretrained(config.sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.sd_model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(config.sd_model_id, subfolder="vae", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(config.sd_model_id, subfolder="unet", torch_dtype=dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(config.sd_model_id, subfolder="scheduler")
    inference_scheduler = DDIMScheduler.from_pretrained(config.sd_model_id, subfolder="scheduler")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(config.train_unet)
    if config.enable_unet_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if config.enable_xformers_if_available:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
        "inference_scheduler": inference_scheduler,
    }


def get_text_embeddings(tokenizer: Any, text_encoder: Any, batch_size: int, prompt: str = "a clear clean dehazed video frame") -> torch.Tensor:
    tokens = tokenizer(
        [prompt] * batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return text_encoder(tokens.input_ids.to(text_encoder.device))[0]


def encode_images_to_latents(vae: Any, images: torch.Tensor) -> torch.Tensor:
    dtype = next(vae.parameters()).dtype
    images = normalize_to_neg_one_to_one(images).to(device=vae.device, dtype=dtype)
    return vae.encode(images).latent_dist.sample() * vae.config.scaling_factor


def decode_latents_to_images(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    dtype = next(vae.parameters()).dtype
    latents = latents.to(device=vae.device, dtype=dtype) / vae.config.scaling_factor
    images = vae.decode(latents).sample.float()
    return unnormalize_to_zero_to_one(images).clamp(0, 1)


def prepare_inpainting_inputs(vae: Any, noisy_latents: torch.Tensor, mask: torch.Tensor, masked_image: torch.Tensor) -> torch.Tensor:
    latent_h, latent_w = noisy_latents.shape[-2:]
    mask_latent = F.interpolate(mask.float(), size=(latent_h, latent_w), mode="nearest").to(
        noisy_latents.device, noisy_latents.dtype
    )
    masked_latents = encode_images_to_latents(vae, masked_image).to(noisy_latents.dtype)
    return torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)


def estimate_x0_from_epsilon(noise_scheduler: Any, noisy_latents: torch.Tensor, timesteps: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(noisy_latents.device, noisy_latents.dtype)
    sqrt_alpha = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus = (1.0 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
    return (noisy_latents - sqrt_one_minus * noise_pred) / (sqrt_alpha + 1e-8)


def enable_lora_for_unet(unet: Any, rank: int = 8) -> Any:
    """LoRA hook for future experiments. Baseline V1 trains the UNet directly."""
    print(f"LoRA hook available. Configure PEFT/Diffusers LoRA adapters here with rank={rank}.")
    return unet
