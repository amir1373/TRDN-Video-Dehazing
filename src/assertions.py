import torch


def _shape(x: torch.Tensor) -> tuple:
    return tuple(x.shape)


def assert_frames(frames: torch.Tensor, seq_len: int | None = None, name: str = "frames") -> None:
    if frames.ndim != 5:
        raise ValueError(f"{name} must have shape [B,T,3,H,W], got {_shape(frames)}")
    if frames.shape[2] != 3:
        raise ValueError(f"{name} channel dimension must be 3, got {_shape(frames)}")
    if seq_len is not None and frames.shape[1] != seq_len:
        raise ValueError(f"{name} sequence length must be {seq_len}, got {_shape(frames)}")


def assert_image(image: torch.Tensor, channels: int = 3, name: str = "image") -> None:
    if image.ndim != 4 or image.shape[1] != channels:
        raise ValueError(f"{name} must have shape [B,{channels},H,W], got {_shape(image)}")


def assert_mask(mask: torch.Tensor, image: torch.Tensor | None = None, name: str = "mask") -> None:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W], got {_shape(mask)}")
    if image is not None and (mask.shape[0], mask.shape[-2], mask.shape[-1]) != (image.shape[0], image.shape[-2], image.shape[-1]):
        raise ValueError(f"{name} shape {_shape(mask)} is incompatible with image shape {_shape(image)}")


def assert_warped_references(warped_refs: torch.Tensor, seq_len: int | None = None, name: str = "warped_references") -> None:
    if warped_refs.ndim != 5 or warped_refs.shape[2] != 3:
        raise ValueError(f"{name} must have shape [B,T-1,3,H,W], got {_shape(warped_refs)}")
    if seq_len is not None and warped_refs.shape[1] != seq_len - 1:
        raise ValueError(f"{name} must have {seq_len - 1} references, got {_shape(warped_refs)}")


def assert_temporal_memory(memory: torch.Tensor, batch: int | None = None, name: str = "temporal_memory") -> None:
    if memory.ndim != 4:
        raise ValueError(f"{name} must have shape [B,C,H,W], got {_shape(memory)}")
    if batch is not None and memory.shape[0] != batch:
        raise ValueError(f"{name} batch must be {batch}, got {_shape(memory)}")


def assert_reference_weights(weights: torch.Tensor, seq_len: int | None = None, name: str = "reference_weights") -> None:
    if weights.ndim != 4:
        raise ValueError(f"{name} must have shape [B,T-1,H,W], got {_shape(weights)}")
    if seq_len is not None and weights.shape[1] != seq_len - 1:
        raise ValueError(f"{name} must have {seq_len - 1} maps, got {_shape(weights)}")


def assert_latents(latents: torch.Tensor, image: torch.Tensor | None = None, name: str = "latents") -> None:
    if latents.ndim != 4 or latents.shape[1] != 4:
        raise ValueError(f"{name} must have shape [B,4,H/8,W/8], got {_shape(latents)}")
    if image is not None:
        expected_hw = (image.shape[-2] // 8, image.shape[-1] // 8)
        if tuple(latents.shape[-2:]) != expected_hw:
            raise ValueError(f"{name} spatial shape must be {expected_hw} for image {_shape(image)}, got {_shape(latents)}")


def assert_flow(flow: torch.Tensor, frames: torch.Tensor | None = None, name: str = "flow") -> None:
    valid = flow.ndim == 4 and flow.shape[1] == 2
    valid_batch = flow.ndim == 5 and flow.shape[2] == 2
    if not (valid or valid_batch):
        raise ValueError(f"{name} must have shape [B,2,H,W] or [B,T-1,2,H,W], got {_shape(flow)}")
    if frames is not None and flow.shape[0] != frames.shape[0]:
        raise ValueError(f"{name} batch {_shape(flow)} does not match frames {_shape(frames)}")
