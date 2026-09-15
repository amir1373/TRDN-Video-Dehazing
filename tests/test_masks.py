import torch

from src.masks import full_frame_mask, generate_haze_mask


def test_full_frame_mask_is_all_ones():
    mask = full_frame_mask(12, 9)
    assert mask.shape == (1, 12, 9)
    assert torch.all(mask == 1.0)


def test_generate_haze_mask_full_mode_matches_helper():
    mask = generate_haze_mask(10, 10, mode="full")
    assert torch.all(mask == 1.0)
    assert mask.shape == (1, 10, 10)


def test_generate_haze_mask_random_modes_are_bounded():
    for mode in ["rectangle", "ellipse", "blob", "perlin", "mixed"]:
        mask = generate_haze_mask(32, 32, mode=mode)
        assert mask.shape == (1, 32, 32)
        assert torch.all(mask >= 0.0) and torch.all(mask <= 1.0)
