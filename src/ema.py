from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn as nn


class EMAState:
    """Default-off EMA over trainable parameters, checkpointable by Accelerate."""

    def __init__(self, modules: Mapping[str, nn.Module | None], decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be between 0 and 1, got {decay}")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in self._named_trainable_parameters(modules).items()
        }

    @staticmethod
    def _named_trainable_parameters(
        modules: Mapping[str, nn.Module | None],
    ) -> Dict[str, nn.Parameter]:
        result = {}
        for module_name, module in modules.items():
            if module is None:
                continue
            for parameter_name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    result[f"{module_name}.{parameter_name}"] = parameter
        return result

    @torch.no_grad()
    def update(self, modules: Mapping[str, nn.Module | None]) -> None:
        current = self._named_trainable_parameters(modules)
        if current.keys() != self.shadow.keys():
            raise RuntimeError("EMA parameter set changed after initialization.")
        for name, value in current.items():
            self.shadow[name].mul_(self.decay).add_(
                value.to(self.shadow[name].device, self.shadow[name].dtype),
                alpha=1.0 - self.decay,
            )
        self.num_updates += 1

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict["num_updates"])
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in state_dict["shadow"].items()
        }

    def save_weights(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)


def load_ema_weights(
    modules: Mapping[str, nn.Module | None],
    path: str | Path,
    *,
    allow_module_subset: bool = False,
) -> dict:
    ema_path = Path(path)
    if not ema_path.is_file():
        raise FileNotFoundError(f"EMA weights do not exist: {ema_path}")
    payload = torch.load(ema_path, map_location="cpu", weights_only=True)
    shadow = payload.get("shadow", {})
    parameters = {
        f"{module_name}.{parameter_name}": parameter
        for module_name, module in modules.items()
        if module is not None
        for parameter_name, parameter in module.named_parameters()
    }
    applicable_shadow = {
        name: value
        for name, value in shadow.items()
        if name in parameters
    }
    missing = sorted(set(shadow) - set(parameters))
    if allow_module_subset:
        active_modules = {
            name for name, module in modules.items() if module is not None
        }
        missing = [
            name
            for name in missing
            if name.split(".", 1)[0] in active_modules
        ]
    if missing:
        raise RuntimeError(f"EMA weights contain unknown parameters: {missing[:5]}")
    with torch.no_grad():
        for name, value in applicable_shadow.items():
            parameter = parameters[name]
            parameter.copy_(value.to(parameter.device, parameter.dtype))
    return {
        "decay": float(payload["decay"]),
        "num_updates": int(payload["num_updates"]),
        "parameter_count": len(applicable_shadow),
    }
