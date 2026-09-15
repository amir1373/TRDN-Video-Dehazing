from pathlib import Path
from typing import Any, Dict

import yaml

from .config import TRDNConfig


PRESET_TO_CONFIG = {
    "precision": "mixed_precision",
    "allow_tf32": "allow_tf32",
    "cudnn_benchmark": "cudnn_benchmark",
    "attention_backend": "attention_backend",
    "batch_size": "batch_size",
    "gradient_checkpointing": "enable_unet_gradient_checkpointing",
    "torch_compile": "enable_torch_compile",
    "channels_last": "channels_last",
}


def _validate_preset_values(payload: Dict[str, Any]) -> None:
    if payload["precision"] not in {"fp16", "bf16", "no"}:
        raise ValueError("precision must be one of: fp16, bf16, no")
    if payload["attention_backend"] not in {"xformers", "sdpa"}:
        raise ValueError("attention_backend must be one of: xformers, sdpa")
    if type(payload["batch_size"]) is not int or payload["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    for key in (
        "allow_tf32",
        "cudnn_benchmark",
        "gradient_checkpointing",
        "torch_compile",
        "channels_last",
    ):
        if type(payload[key]) is not bool:
            raise ValueError(f"{key} must be a YAML boolean")


def load_numerics_preset(path: str | Path) -> Dict[str, Any]:
    preset_path = Path(path)
    payload = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Numerics preset must contain a mapping: {preset_path}")
    missing = [key for key in PRESET_TO_CONFIG if key not in payload]
    if missing:
        raise ValueError(f"Numerics preset is missing fields: {', '.join(missing)}")
    unresolved = [
        key
        for key in PRESET_TO_CONFIG
        if payload[key] is None or str(payload[key]).strip().upper() == "TODO"
    ]
    if unresolved:
        raise ValueError(
            "Numerics preset still contains TODO values from the unmeasured A40 scaffold: "
            + ", ".join(unresolved)
        )
    _validate_preset_values(payload)
    return payload


def apply_numerics_preset(config: TRDNConfig, path: str | Path) -> TRDNConfig:
    payload = load_numerics_preset(path)
    for preset_name, config_name in PRESET_TO_CONFIG.items():
        setattr(config, config_name, payload[preset_name])
    config.enable_xformers_if_available = config.attention_backend == "xformers"
    return config
