from pathlib import Path

import pytest

from src.config import TRDNConfig
from src.presets import apply_numerics_preset, load_numerics_preset


def test_a40_scaffold_refuses_unmeasured_todo_values():
    preset = Path(__file__).parents[1] / "configs" / "a40.yaml"

    with pytest.raises(ValueError, match="TODO"):
        load_numerics_preset(preset)


def test_filled_numerics_preset_applies_every_field(tmp_path: Path):
    preset = tmp_path / "measured.yaml"
    preset.write_text(
        "\n".join(
            [
                "precision: bf16",
                "allow_tf32: true",
                "cudnn_benchmark: true",
                "attention_backend: sdpa",
                "batch_size: 4",
                "gradient_checkpointing: false",
                "torch_compile: true",
                "channels_last: true",
            ]
        ),
        encoding="utf-8",
    )
    config = apply_numerics_preset(TRDNConfig(), preset)

    assert config.mixed_precision == "bf16"
    assert config.allow_tf32 is True
    assert config.cudnn_benchmark is True
    assert config.attention_backend == "sdpa"
    assert config.batch_size == 4
    assert config.enable_unet_gradient_checkpointing is False
    assert config.enable_torch_compile is True
    assert config.channels_last is True
    assert config.enable_xformers_if_available is False


def test_numerics_preset_rejects_string_booleans(tmp_path: Path):
    preset = tmp_path / "invalid.yaml"
    preset.write_text(
        "\n".join(
            [
                "precision: fp16",
                "allow_tf32: 'false'",
                "cudnn_benchmark: false",
                "attention_backend: xformers",
                "batch_size: 1",
                "gradient_checkpointing: true",
                "torch_compile: false",
                "channels_last: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allow_tf32"):
        load_numerics_preset(preset)
