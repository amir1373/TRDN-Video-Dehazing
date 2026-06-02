from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_adapter import normalize_to_neg_one_to_one


class LossBundle(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        try:
            import lpips

            self.lpips_model = lpips.LPIPS(net="alex").to(device).eval()
            for param in self.lpips_model.parameters():
                param.requires_grad_(False)
        except Exception:
            self.lpips_model = None

    def lpips_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lpips_model is None:
            return pred.new_tensor(0.0)
        pred = normalize_to_neg_one_to_one(pred.clamp(0, 1))
        target = normalize_to_neg_one_to_one(target.clamp(0, 1))
        return self.lpips_model(pred, target).mean()

    @staticmethod
    def temporal_consistency_loss(pred: torch.Tensor, warped_refs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_ref = (weights.unsqueeze(2) * warped_refs).sum(dim=1)
        return F.l1_loss(pred, weighted_ref.detach())

    @staticmethod
    def flow_consistency_loss(warped_refs: torch.Tensor, current: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_ref = (weights.unsqueeze(2) * warped_refs).sum(dim=1)
        return F.l1_loss(weighted_ref, current.detach())

    @staticmethod
    def reference_preservation_loss(pred: torch.Tensor, weighted_reference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (torch.abs(pred - weighted_reference.detach()) * (1.0 - mask)).mean()


def weighted_total_loss(config: Any, parts: dict) -> torch.Tensor:
    return (
        config.w_diffusion * parts["diffusion"]
        + config.w_l1 * parts["l1"]
        + config.w_lpips * parts["lpips"]
        + config.w_temporal * parts["temporal"]
        + config.w_flow * parts["flow"]
        + config.w_reference * parts["reference"]
    )
