import hashlib
from typing import Any

import torch


def derive_generator(seed: int, *parts: Any, device: str = "cpu") -> torch.Generator:
    """Deterministically derive a torch.Generator from a base seed plus key parts.

    Combining a fixed base seed with per-sample / per-frame-index identifiers
    (e.g. clip name, frame index) gives every sample its own reproducible noise
    stream, while the whole run stays reproducible end to end for a fixed seed.

    Note: determinism is only guaranteed within the same device family (CPU vs
    CUDA use different RNG algorithms), and CUDA-side determinism additionally
    depends on library/kernel choices outside this function's control.
    """
    key = str(seed) + ":" + ":".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed64 = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed64)
    return generator
