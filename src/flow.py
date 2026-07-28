from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .warp import warp_with_flow
from .assertions import assert_flow, assert_frames, assert_image, assert_warped_references


def load_raft(
    device: str = "cuda",
    freeze: bool = True,
    validate_flow: bool = True,
    max_flow_factor: float = 2.0,
) -> nn.Module:
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(device).eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)
    model._trdn_validate_flow = validate_flow
    model._trdn_max_flow_factor = max_flow_factor
    return model


def _raft_preprocess(x: torch.Tensor) -> torch.Tensor:
    return x.float().clamp(0, 1) * 2.0 - 1.0


@torch.no_grad()
def compute_raft_flow(raft_model: nn.Module, source: torch.Tensor, target: torch.Tensor, iters: int = 12) -> torch.Tensor:
    """Compute RAFT flow source -> target as [B, 2, H, W]."""
    assert_image(source, name="source")
    assert_image(target, name="target")
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(f"RAFT source and target shapes must match, got {tuple(source.shape)} and {tuple(target.shape)}")
    _, _, height, width = source.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    source_in = F.pad(source.float(), (0, pad_w, 0, pad_h), mode="replicate")
    target_in = F.pad(target.float(), (0, pad_w, 0, pad_h), mode="replicate")
    flow = raft_model(_raft_preprocess(source_in), _raft_preprocess(target_in), num_flow_updates=iters)[-1]
    flow = flow[..., :height, :width]
    assert_flow(flow, name="raft_flow")
    if getattr(raft_model, "_trdn_validate_flow", True):
        assert_raft_flow_sane(
            flow,
            height,
            width,
            max_flow_factor=float(getattr(raft_model, "_trdn_max_flow_factor", 2.0)),
        )
    return flow


def assert_raft_flow_sane(
    flow: torch.Tensor,
    height: int,
    width: int,
    max_flow_factor: float = 2.0,
) -> None:
    if not torch.isfinite(flow).all():
        raise FloatingPointError("RAFT produced NaN or Inf flow values.")
    if max_flow_factor <= 0:
        raise ValueError(f"max_flow_factor must be positive, got {max_flow_factor}")
    maximum_allowed = max(height, width) * max_flow_factor
    maximum_observed = torch.linalg.vector_norm(flow.float(), dim=1).amax()
    if maximum_observed > maximum_allowed:
        raise FloatingPointError(
            "RAFT flow magnitude is implausible for the frame size: "
            f"max={float(maximum_observed):.3f}, allowed={maximum_allowed:.3f} "
            f"for {height}x{width} frames."
        )


@torch.no_grad()
def compute_warped_references_batch(
    frames: torch.Tensor,
    raft_model: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align previous frames to the current frame.

    Args:
        frames: [B, T, 3, H, W]
    Returns:
        warped_references: [B, T-1, 3, H, W]
        flows: [B, T-1, 2, H, W]
    """
    assert_frames(frames, name="frames")
    batch, seq_len, _, height, width = frames.shape
    current = frames[:, -1]
    warped_refs = []
    flows = []
    for tidx in range(seq_len - 1):
        source = frames[:, tidx]
        if raft_model is None:
            flow = torch.zeros(batch, 2, height, width, device=frames.device, dtype=frames.dtype)
        else:
            flow = compute_raft_flow(raft_model, source, current)
        warped_refs.append(warp_with_flow(source, flow))
        flows.append(flow)
    warped = torch.stack(warped_refs, dim=1)
    flow_stack = torch.stack(flows, dim=1)
    assert_warped_references(warped, seq_len=seq_len)
    assert_flow(flow_stack, frames=frames, name="flow_stack")
    return warped, flow_stack


def flow_warped_temporal_consistency_error(
    prev_pred: torch.Tensor, curr_pred: torch.Tensor, raft_model: nn.Module, fb_threshold: float = 1.0
) -> Tuple[float, float]:
    """Flow-warped temporal consistency error between two consecutive predictions.

    Warps ``prev_pred`` into ``curr_pred``'s coordinates with RAFT flow (via the
    existing `warp_with_flow` utility) and reports the mean L1 error, restricted
    to pixels that pass a forward-backward flow consistency check (standard
    occlusion/invalid-flow masking: a pixel is trusted only if following the
    forward flow and then the backward flow returns close to the start).

    Returns:
        (mean_l1_error_on_valid_pixels, fraction_of_pixels_marked_valid)
    """
    flow_fwd = compute_raft_flow(raft_model, prev_pred, curr_pred)
    flow_bwd = compute_raft_flow(raft_model, curr_pred, prev_pred)
    warped_prev = warp_with_flow(prev_pred, flow_fwd)
    warped_flow_bwd = warp_with_flow(flow_bwd, flow_fwd)

    fb_diff = torch.norm(flow_fwd + warped_flow_bwd, dim=1, keepdim=True)
    flow_mag_sq = torch.sum(flow_fwd**2, dim=1, keepdim=True) + torch.sum(warped_flow_bwd**2, dim=1, keepdim=True)
    threshold = 0.01 * flow_mag_sq + fb_threshold
    valid_mask = (fb_diff < threshold).float()

    error = torch.abs(warped_prev - curr_pred) * valid_mask
    valid_pixels = valid_mask.sum().clamp_min(1.0)
    mean_l1 = (error.sum() / (valid_pixels * curr_pred.shape[1])).item()
    coverage = valid_mask.mean().item()
    return mean_l1, coverage


def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)
    flow_np = flow[0].detach().float().cpu().permute(1, 2, 0).numpy()
    magnitude, angle = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / (np.percentile(magnitude, 95) + 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)
