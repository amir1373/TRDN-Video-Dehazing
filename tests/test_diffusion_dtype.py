import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.config import TRDNConfig
from src.diffusion_adapter import load_diffusion_backbone, resolve_frozen_dtype


class _FakeComponent(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=dtype))
        self.config = SimpleNamespace(cross_attention_dim=8)

    @classmethod
    def from_pretrained(cls, _model_id, subfolder=None, torch_dtype=torch.float32):
        return cls(torch_dtype)

    def to(self, _device):
        return self

    def enable_gradient_checkpointing(self):
        pass

    def enable_xformers_memory_efficient_attention(self):
        pass

    def set_default_attn_processor(self):
        pass


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, _model_id, subfolder=None):
        return cls()


class _FakeScheduler:
    @classmethod
    def from_pretrained(cls, _model_id, subfolder=None):
        return cls()


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
        ("no", torch.float32),
    ],
)
def test_frozen_backbone_dtype_matches_requested_cuda_precision(
    precision: str,
    expected: torch.dtype,
    monkeypatch,
):
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.AutoencoderKL = _FakeComponent
    fake_diffusers.UNet2DConditionModel = _FakeComponent
    fake_diffusers.DDPMScheduler = _FakeScheduler
    fake_diffusers.DDIMScheduler = _FakeScheduler
    fake_transformers = ModuleType("transformers")
    fake_transformers.CLIPTextModel = _FakeComponent
    fake_transformers.CLIPTokenizer = _FakeTokenizer
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    config = TRDNConfig(
        mixed_precision=precision,
        train_unet=False,
        enable_unet_gradient_checkpointing=False,
        enable_xformers_if_available=False,
        enable_torch_compile=False,
    )

    runtime = load_diffusion_backbone(config, device="cuda")

    assert next(runtime["vae"].parameters()).dtype == expected
    assert next(runtime["text_encoder"].parameters()).dtype == expected
    assert next(runtime["unet"].parameters()).dtype == expected
    assert resolve_frozen_dtype(precision, "cuda") == expected


def test_cpu_frozen_backbone_dtype_falls_back_to_fp32():
    assert resolve_frozen_dtype("fp16", "cpu") == torch.float32
    assert resolve_frozen_dtype("bf16", "cpu") == torch.float32
