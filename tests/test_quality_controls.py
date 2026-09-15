from pathlib import Path

import pytest
import torch

from src.config import TRDNConfig
from src.ema import EMAState, load_ema_weights
from src.train import apply_optional_lr_schedule, effective_learning_rates


def test_linear_lr_scaling_is_default_off_and_explicit_when_enabled():
    config = TRDNConfig(
        learning_rate=1e-5,
        temporal_learning_rate=1e-4,
        batch_size=4,
        gradient_accumulation_steps=2,
    )
    assert effective_learning_rates(config) == (1e-5, 1e-4)

    config.enable_linear_lr_scaling = True
    config.lr_reference_batch_size = 2
    assert effective_learning_rates(config) == pytest.approx((4e-5, 4e-4))


def test_warmup_cosine_schedule_is_default_off():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1.0}])
    config = TRDNConfig(lr_schedule="constant")
    apply_optional_lr_schedule(config, optimizer, [1.0], step=1, total_steps=10)
    assert optimizer.param_groups[0]["lr"] == 1.0

    config.lr_schedule = "warmup_cosine"
    config.lr_warmup_steps = 2
    apply_optional_lr_schedule(config, optimizer, [1.0], step=1, total_steps=10)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    apply_optional_lr_schedule(config, optimizer, [1.0], step=10, total_steps=10)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_ema_is_saved_separately_and_can_be_loaded(tmp_path: Path):
    source = torch.nn.Linear(2, 1, bias=False)
    target = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(1.0)
        target.weight.zero_()
    ema = EMAState({"unet": source}, decay=0.5)
    with torch.no_grad():
        source.weight.fill_(3.0)
    ema.update({"unet": source})
    path = tmp_path / "ema_weights.pt"
    ema.save_weights(path)

    report = load_ema_weights({"unet": target}, path)

    assert report["num_updates"] == 1
    assert torch.allclose(target.weight, torch.full_like(target.weight, 2.0))
