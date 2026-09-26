from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_adapter import normalize_to_neg_one_to_one
from .warp import warp_with_flow


class LossBundle(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        try:
            import lpips

            self.lpips_model = lpips.LPIPS(net="alex").to(device).eval()
            for param in self.lpips_model.parameters():
                param.requires_grad_(False)
        except Exception as exc:
            raise RuntimeError(
                "LPIPS initialization failed while w_lpips is part of the training objective. "
                "Refusing to continue with a silently disabled perceptual loss."
            ) from exc
        self.lpips_startup_probe = self._assert_lpips_nonzero(device)

    def _assert_lpips_nonzero(self, device: str) -> float:
        first = torch.zeros(1, 3, 64, 64, device=device)
        second = torch.ones(1, 3, 64, 64, device=device)
        with torch.no_grad():
            value = self.lpips_loss(first, second).float()
        if not torch.isfinite(value):
            raise RuntimeError("LPIPS startup probe returned a non-finite value.")
        scalar = float(value.detach().cpu())
        if scalar <= 0.0:
            raise RuntimeError(
                "LPIPS startup probe returned zero for deliberately different tensors. "
                "Refusing to train with an ineffective perceptual loss."
            )
        return scalar

    def lpips_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = normalize_to_neg_one_to_one(pred.clamp(0, 1))
        target = normalize_to_neg_one_to_one(target.clamp(0, 1))
        return self.lpips_model(pred, target).mean()

    @staticmethod
    def legacy_temporal_consistency_loss(pred: torch.Tensor, warped_refs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Original (pre-fix) temporal consistency loss: L1(pred, fused warped reference).

        Only correct when the references are known-clean (train_mode="reconstruct_synthetic"),
        which is why it is kept only for that legacy path. In "dehaze" mode the
        references are real hazy frames, so this term would pull predictions back
        toward haze -- use `predictive_temporal_consistency_loss` instead.
        """
        weighted_ref = (weights.unsqueeze(2) * warped_refs).sum(dim=1)
        return F.l1_loss(pred, weighted_ref.detach())

    @staticmethod
    def predictive_temporal_consistency_loss(
        pred_current: torch.Tensor, pred_previous: torch.Tensor, flow_previous_to_current: torch.Tensor
    ) -> torch.Tensor:
        """True temporal consistency, measured on the model's own predictions.

        Warps the previous frame's PREDICTION into the current frame's
        coordinates using the existing RAFT flow + warp utility, then penalizes
        disagreement with the current prediction. Unlike the legacy loss above,
        this never references the (possibly hazy) input frames, so it cannot
        pull dehazed predictions back toward haze.
        """
        warped_previous = warp_with_flow(pred_previous, flow_previous_to_current)
        return F.l1_loss(pred_current, warped_previous.detach())

    @staticmethod
    def flow_consistency_loss(warped_refs: torch.Tensor, current: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_ref = (weights.unsqueeze(2) * warped_refs).sum(dim=1)
        return F.l1_loss(weighted_ref, current.detach())

    @staticmethod
    def reference_preservation_loss(pred: torch.Tensor, weighted_reference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Penalizes prediction drift outside the mask, relative to the fused reference.

        Only meaningful when the mask marks a real synthetic-occlusion region
        (train_mode="reconstruct_synthetic") and the reference is clean. With a
        full-frame mask (dehaze mode default), (1 - mask) is zero everywhere and
        this term is a no-op by construction; config.w_reference defaults to 0.0
        for dehaze mode regardless, so it is skipped rather than relying on that.
        """
        return (torch.abs(pred - weighted_reference.detach()) * (1.0 - mask)).mean()


def weighted_total_loss(config: Any, parts: dict) -> torch.Tensor:
    total = (
        config.w_diffusion * parts["diffusion"]
        + config.w_l1 * parts["l1"]
        + config.w_lpips * parts["lpips"]
    )
    if "temporal" in parts:
        total = total + config.w_temporal * parts["temporal"]
    if "flow" in parts:
        total = total + config.w_flow * parts["flow"]
    if "reference" in parts:
        total = total + config.w_reference * parts["reference"]
    for name in ("rgb_mse", "haze_map"):
        weight = float(getattr(config, f"w_{name}", 0.0))
        if weight > 0 and name in parts:
            total = total + weight * parts[name]
    w_latent_x0 = float(getattr(config, "w_latent_x0", 0.0))
    if w_latent_x0 > 0 and "latent_x0" in parts:
        total = total + w_latent_x0 * parts["latent_x0"]
    w_entropy = float(getattr(config, "w_selector_entropy", 0.0))
    if w_entropy > 0 and "sel_entropy" in parts:
        total = total - w_entropy * parts["sel_entropy"]
    return total


def latent_x0_min_snr_loss(
    noise_scheduler: Any,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise_pred: torch.Tensor,
    clean_latents: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Min-SNR-weighted MSE between the predicted clean latent and the clean target latent.

    Computed in float32: the x0 estimate divides by sqrt(alpha_bar), which is tiny at large t.
    Each sample's x0 MSE is weighted by min(SNR, gamma), so the effective weight on the
    epsilon error is min(SNR, gamma) / SNR <= 1 at every timestep.
    """
    alphas = noise_scheduler.alphas_cumprod.to(noisy_latents.device, torch.float32)[timesteps]
    alphas = alphas.view(-1, 1, 1, 1)
    pred_x0 = (noisy_latents.float() - (1.0 - alphas).sqrt() * noise_pred.float()) / alphas.sqrt()
    per_sample = ((pred_x0 - clean_latents.detach().float()) ** 2).mean(dim=(1, 2, 3))
    snr = (alphas / (1.0 - alphas)).view(-1)
    weight = torch.clamp(snr, max=gamma)
    return (weight * per_sample).mean()
