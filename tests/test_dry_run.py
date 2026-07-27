from src.train import dry_run_shape_test


def test_dry_run_shape_test_dehaze_mode():
    result = dry_run_shape_test(seq_len=4, image_size=32, train_mode="dehaze")
    assert result["train_mode"] == "dehaze"
    assert result["mask_mode"] == "full"
    assert result["frames"] == (1, 4, 3, 32, 32)
    assert result["warped_references"] == (1, 3, 3, 32, 32)
    assert result["reference_weights"] == (1, 3, 32, 32)
    # dehaze mode must expose the previous-frame window used by the
    # predictive temporal-consistency loss, and its warp must line up with
    # the current prediction's shape.
    assert "prev_frames" in result
    assert result["prev_frames"] == (1, 4, 3, 32, 32)
    assert result["predictive_temporal_warp"] == result["current_hazy"]


def test_dry_run_shape_test_reconstruct_synthetic_mode():
    result = dry_run_shape_test(seq_len=4, image_size=32, train_mode="reconstruct_synthetic")
    assert result["train_mode"] == "reconstruct_synthetic"
    assert result["mask_mode"] == "mixed"
    assert result["frames"] == (1, 4, 3, 32, 32)
    # Legacy mode does not use the predictive-loss prev-frame window.
    assert "prev_frames" not in result


def test_dry_run_shape_test_legacy_alias_still_runs():
    result = dry_run_shape_test(seq_len=4, image_size=32, train_mode="reconstruct")
    assert result["train_mode"] == "reconstruct_synthetic"
