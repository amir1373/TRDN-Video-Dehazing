from typing import Iterable

import torch
import torch.nn as nn


def configure_torch_backends(config) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)
    torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)


def apply_channels_last(modules: Iterable[nn.Module | None], enabled: bool) -> None:
    if not enabled:
        return
    for module in modules:
        if module is not None:
            module.to(memory_format=torch.channels_last)
