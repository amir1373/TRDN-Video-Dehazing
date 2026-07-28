from contextlib import nullcontext

import torch

import src.train as train_module
from src.config import TRDNConfig
from src.dataset import REVIDESequenceDataset


class FakeAccelerator:
    device = "cpu"

    @staticmethod
    def autocast():
        return nullcontext()


class DiffusionOnlyLoss:
    @staticmethod
    def lpips_loss(prediction, target):
        return torch.abs(prediction - target).mean()

    def __getattr__(self, name):
        raise AssertionError(f"Diffusion-only training invoked forbidden loss {name}")


def test_model_variants_resolve_expected_components():
    expected = {
        "full": (True, True, True),
        "no_raft": (False, True, True),
        "no_transformer": (True, False, True),
        "diffusion_only": (False, False, False),
    }
    for variant, resolved in expected.items():
        config = TRDNConfig(model_variant=variant)
        config.apply_model_variant()
        assert (
            config.use_raft_alignment,
            config.use_temporal_transformer,
            config.train_temporal_modules,
        ) == resolved


def test_diffusion_only_constructs_no_temporal_modules(monkeypatch):
    config = TRDNConfig(model_variant="diffusion_only")
    config.apply_model_variant()
    monkeypatch.setattr(
        train_module,
        "TemporalMemoryModule",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ConvLSTM must not be constructed")
        ),
    )

    modules = train_module.build_temporal_modules(config, 768, "cpu")

    assert modules == (None, None, None, None)


def test_diffusion_only_computes_only_defined_losses(monkeypatch):
    prediction = torch.full((1, 3, 8, 8), 0.4)
    diffusion_loss = torch.tensor(0.25)
    monkeypatch.setattr(
        train_module,
        "forward_diffusion_only_prediction",
        lambda *_args, **_kwargs: {
            "pred_img": prediction,
            "diffusion_loss": diffusion_loss,
        },
    )
    config = TRDNConfig(model_variant="diffusion_only")
    config.apply_model_variant()
    batch = {
        "frames": torch.zeros(1, 2, 3, 8, 8),
        "target_frame": torch.full((1, 3, 8, 8), 0.5),
        "mask": torch.ones(1, 1, 8, 8),
        "corrupted_frame": torch.full((1, 3, 8, 8), 0.6),
    }

    total, parts = train_module.compute_training_loss(
        FakeAccelerator(),
        {},
        None,
        None,
        None,
        None,
        None,
        DiffusionOnlyLoss(),
        batch,
        config,
    )

    assert set(parts) == {"diffusion", "l1", "lpips"}
    expected = (
        config.w_diffusion * parts["diffusion"]
        + config.w_l1 * parts["l1"]
        + config.w_lpips * parts["lpips"]
    )
    assert torch.equal(total, expected)


def test_diffusion_only_dataset_loads_current_frame_only(tmp_path):
    dataset = REVIDESequenceDataset(
        str(tmp_path),
        seq_len=10,
        crop_size=16,
        synthetic_if_empty=True,
        train_mode="dehaze",
        include_prev_frame=False,
        include_reference_frames=False,
    )
    sample = dataset[0]
    assert sample["frames"].shape == (1, 3, 16, 16)
    assert sample["hazy_frames"].shape[0] == 1
    assert sample["warped_references"].shape[0] == 0
