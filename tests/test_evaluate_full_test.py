import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.config import TRDNConfig
from src.flow import flow_warped_temporal_consistency_error
from scripts.evaluate_full_test import build_test_dataset, group_index_by_clip
from tests.conftest import make_fake_revide_root


def test_build_test_dataset_and_group_index_by_clip_no_filtering(tmp_path: Path):
    """Task A: the eval script must iterate every clip and every window, with
    no sample filtering of any kind."""
    sequence_names = ["clip_a", "clip_b", "clip_c"]
    test_root = make_fake_revide_root(tmp_path / "Test", sequence_names, num_frames=6, size=8)
    config = TRDNConfig(test_root=str(test_root), seq_len=3)
    args = argparse.Namespace(train_mode="dehaze", mask_mode="auto", crop_size=8)

    dataset = build_test_dataset(config, args)
    assert dataset.train_mode == "dehaze"
    by_clip = group_index_by_clip(dataset)

    # every discovered sequence must be present, nothing dropped
    discovered_names = {dataset.sequences[seq_idx]["name"] for seq_idx in by_clip}
    assert discovered_names == set(sequence_names)

    # total windows must equal the sum over clips of (num_frames - seq_len + 1),
    # i.e. every valid window is included, none skipped
    expected_windows_per_clip = 6 - config.seq_len + 1
    for seq_idx, indices in by_clip.items():
        assert len(indices) == expected_windows_per_clip
        # ascending order (sequential frames) so consecutive-frame temporal
        # consistency can be computed directly while iterating
        end_indices = [dataset.index[i][1] for i in indices]
        assert end_indices == sorted(end_indices)


class _ZeroFlowModel(nn.Module):
    """Stand-in for RAFT that always predicts zero motion, used to test the
    forward-backward consistency masking math in flow_warped_temporal_consistency_error
    without downloading real RAFT weights (no network access in this environment)."""

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, num_flow_updates: int = 12):
        batch, _, height, width = img1.shape
        return [torch.zeros(batch, 2, height, width, device=img1.device, dtype=img1.dtype)]


def test_flow_warped_temporal_consistency_error_zero_motion_identical_frames():
    model = _ZeroFlowModel()
    frame = torch.rand(1, 3, 16, 16)
    error, coverage = flow_warped_temporal_consistency_error(frame, frame, model)
    assert error < 1e-5
    assert coverage == 1.0  # zero flow everywhere -> forward/backward trivially agree everywhere


def test_flow_warped_temporal_consistency_error_zero_motion_different_frames_equals_l1():
    model = _ZeroFlowModel()
    prev_pred = torch.rand(1, 3, 16, 16)
    curr_pred = torch.rand(1, 3, 16, 16)
    error, coverage = flow_warped_temporal_consistency_error(prev_pred, curr_pred, model)
    expected = torch.abs(prev_pred - curr_pred).mean().item()
    assert coverage == 1.0
    assert abs(error - expected) < 1e-5
