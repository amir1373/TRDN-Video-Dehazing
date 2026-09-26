from pathlib import Path

import torch

from src.dataset import REVIDESequenceDataset
from src.losses import latent_x0_min_snr_loss
from src.occlusion import OCCLUDER_FILL, occluder_mask, seeded_occluder_mask
from tests.conftest import make_fake_revide_root


def _occlude_dataset(root: Path, **kwargs) -> REVIDESequenceDataset:
    return REVIDESequenceDataset(
        str(root), split="train", seq_len=3, crop_size=64, random_crop=False,
        train_mode="occlude", val_fraction=0.0, **kwargs,
    )


def test_occluder_coverage_is_close_to_requested():
    for coverage in (0.15, 0.35, 0.55):
        measured = [float(occluder_mask(128, 128, coverage).mean()) for _ in range(30)]
        assert abs(sum(measured) / len(measured) - coverage) < 0.12


def test_seeded_occluder_is_deterministic_and_leaves_global_rng_alone():
    import random

    random.seed(7)
    before = random.random()
    random.seed(7)
    a = seeded_occluder_mask(64, 64, 0.35, "C005:3")
    b = seeded_occluder_mask(64, 64, 0.35, "C005:3")
    assert torch.equal(a, b)
    assert random.random() == before
    assert not torch.equal(a, seeded_occluder_mask(64, 64, 0.35, "C005:4"))


def test_occlude_mode_hides_the_same_pixels_in_every_frame(tmp_path: Path):
    root = make_fake_revide_root(tmp_path / "train", ["seq_a"], num_frames=6, size=64)
    dataset = _occlude_dataset(root, occlusion_eval_coverage=0.35)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        mask = sample["mask"]
        assert mask.shape == (1, 64, 64) and 0.05 < float(mask.mean()) < 0.8
        hidden = mask.bool().expand(3, -1, -1)
        for t in range(sample["frames"].shape[0]):
            assert torch.all(sample["frames"][t][hidden] == OCCLUDER_FILL)
            # Visible pixels are the untouched hazy frame, never the clean target.
            assert torch.equal(sample["frames"][t][~hidden], sample["hazy_frames"][t][~hidden])
        assert torch.equal(sample["corrupted_frame"], sample["frames"][-1])
        assert not torch.equal(sample["target_frame"], sample["corrupted_frame"])
        # The occluder is on the lens, so the previous window carries the same one.
        assert torch.equal(sample["prev_mask"], mask)
        # Fixed evaluation coverage gives identical occluders on every read.
        assert torch.equal(dataset[idx]["mask"], mask)


def test_training_occluders_vary(tmp_path: Path):
    root = make_fake_revide_root(tmp_path / "train", ["seq_a"], num_frames=6, size=64)
    dataset = _occlude_dataset(root)
    masks = [dataset[0]["mask"] for _ in range(4)]
    assert any(not torch.equal(masks[0], other) for other in masks[1:])


class _Scheduler:
    def __init__(self):
        betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000) ** 2
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)


def test_latent_x0_loss_is_min_snr_weighted_and_bounded():
    scheduler = _Scheduler()
    torch.manual_seed(0)
    clean = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(clean)
    for t in (10, 500, 999):
        timesteps = torch.tensor([t, t])
        a = scheduler.alphas_cumprod[t]
        noisy = a.sqrt() * clean + (1 - a).sqrt() * noise
        # A perfect epsilon prediction gives zero x0 error.
        assert latent_x0_min_snr_loss(scheduler, noisy, timesteps, noise, clean, 5.0) < 1e-8
        # An epsilon error of size e gives an effective epsilon-space weight min(SNR, 5)/SNR <= 1.
        eps_error = 0.1 * torch.randn_like(noise)
        loss = latent_x0_min_snr_loss(scheduler, noisy, timesteps, noise + eps_error, clean, 5.0)
        eps_mse = float((eps_error**2).mean())
        snr = float(a / (1 - a))
        assert abs(float(loss) - min(snr, 5.0) / snr * eps_mse) < 1e-4
        assert float(loss) <= eps_mse * 1.0001


def test_current_scope_occludes_only_the_current_frame(tmp_path: Path):
    root = make_fake_revide_root(tmp_path / "train", ["seq_a"], num_frames=6, size=64)
    dataset = _occlude_dataset(root, occlusion_eval_coverage=0.35, occlusion_scope="current")
    sample = dataset[0]
    hidden = sample["mask"].bool().expand(3, -1, -1)
    assert torch.all(sample["frames"][-1][hidden] == OCCLUDER_FILL)
    for t in range(sample["frames"].shape[0] - 1):
        assert torch.equal(sample["frames"][t], sample["hazy_frames"][t])
    # The previous window is not occluded: the occluder is transient.
    assert float(sample["prev_mask"].sum()) == 0
    assert sample["occlusion_scope"] == "current"
