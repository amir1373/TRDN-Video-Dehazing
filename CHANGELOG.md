# Changelog

## Unreleased -- Fix dehaze/eval protocol

An audit found that the trained/evaluated configuration did not measure video
dehazing at all, plus several related issues. This change fixes the pipeline
to measure the intended task, without changing or inventing any previously
reported numbers.

### Fixed

- **Ground-truth leak in training (root cause).** `train_mode="reconstruct"`
  used ground-truth clean frames as the T-1 temporal references and painted
  synthetic haze onto the clean target for the "hazy" input, so real REVIDE
  haze never reached the model. `train_mode="dehaze"` (real hazy frames as
  both references and input) is now the default in `src/config.py`. The old
  branch is kept -- renamed to `train_mode="reconstruct_synthetic"` (with a
  deprecated `"reconstruct"` alias that still works but warns) -- because
  previously reported numbers were produced with it and must remain
  explainable/reproducible. `src/dataset.py` now logs a loud warning whenever
  it is selected, and documents in-code that it is not a dehazing evaluation.

- **Validation set was the test set.** `TRDNConfig.root_for_split("val")`
  mapped to `test_root`, so checkpoints were being selected on test data.
  Validation is now a held-out subset of TRAINING sequences, chosen
  deterministically by a seeded hash of sequence name (`config.split_seed`,
  intentionally independent of the training `seed`) so no clip overlaps
  between train and val. `root_for_split("val")` now resolves to
  `train_root`; `test_root` is only meant to be read by the new
  `scripts/evaluate_full_test.py`.

- **Two losses fought dehazing in `dehaze` mode.**
  `temporal_consistency_loss` pulled predictions toward the fused *input*
  reference, which is fine when references are known-clean
  (`reconstruct_synthetic`) but actively pulls dehazed predictions back
  toward haze when references are real hazy frames. It has been renamed to
  `LossBundle.legacy_temporal_consistency_loss` and is only used in
  `reconstruct_synthetic` mode (unchanged behavior, for reproducibility). A
  new `LossBundle.predictive_temporal_consistency_loss` warps the *previous
  frame's prediction* into the current frame using the existing RAFT
  flow/`warp_with_flow` utility and penalizes disagreement with the current
  prediction -- it never references the input frames, so it cannot pull
  toward haze. `src/train.py` now runs an extra no-grad forward pass on the
  previous frame's window (see `forward_window_prediction` /
  `REVIDESequenceDataset`'s `prev_frames`) to obtain that previous
  prediction. `reference_preservation_loss` (which has the same problem
  outside the mask) now defaults to weight 0 (`config.w_reference`, was
  0.05); `weighted_total_loss` no longer requires a `"reference"` key so
  omitting the term can't `KeyError`.

- **Mask semantics were meaningless for dehazing.** Random
  rectangle/ellipse/blob/perlin masks have no relationship to real, global
  haze. Added `mask_mode="full"` (all-ones mask, full-frame restoration
  through the SD inpainting interface), which is now the default whenever
  `train_mode="dehaze"` (`mask_mode="auto"` resolves to `"full"` for dehaze
  and `"mixed"` for `reconstruct_synthetic`, preserving legacy behavior of
  that mode unless explicitly overridden).

- **Unseeded inference noise.** `infer_dehazed_batch` sampled
  `torch.randn(...)` with no generator, producing a measured 1.67 dB spread
  between two runs of the same configuration, plus inflated flicker from
  independent per-frame noise. Inference now derives a deterministic
  per-sample, per-frame-index generator from a seed (`src/seeding.py`) and
  uses DDIM with `eta=0`, so results are reproducible by default. Applies to
  `infer_dehazed_batch`, the new `infer_diffusion_only_batch`, and
  `validate_trdn`.

### Added

- `scripts/evaluate_full_test.py`: a clean, fully seeded, deterministic
  evaluation over the *entire* test set (no sample filtering, scoring, or
  "good candidate" selection of any kind). Reports PSNR, SSIM, LPIPS, and a
  flow-warped temporal consistency error (RAFT-warp prediction t-1 into t,
  mask invalid/occluded pixels via forward-backward flow consistency, report
  mean L1 on valid pixels -- not a naive inter-frame difference). Writes
  per-clip and aggregate results to JSON, including `N`, `seed`,
  `num_inference_steps`, `checkpoint_path`, `train_mode`, `mask_mode`, and the
  current git commit hash. `--num-steps` is a required argument (not
  hardcoded).
- `--diffusion-only` baseline in the same script: runs the SD inpainting
  backbone frame-by-frame with no RAFT/ConvLSTM/temporal transformer/
  reference selector, to isolate the temporal stack's contribution.
- `src/flow.py`: `flow_warped_temporal_consistency_error`, the forward-
  backward-consistency-masked flow-warped metric used by the eval script.
- `src/seeding.py`: `derive_generator`, a deterministic per-(seed, parts)
  `torch.Generator` factory.
- `tests/`: unit tests covering the ground-truth-leak fix (dehaze mode never
  hands the model a tensor equal to the target, and all references come from
  `hazy_files`), the train/val/test split's sequence-level disjointness, the
  new mask modes, the new/legacy loss functions, `warp_with_flow`'s
  generalization to non-RGB tensors, deterministic seeding, and
  `dry_run_shape_test` in both `train_mode`s.

### Changed

- `src/warp.py`: `warp_with_flow` now infers channel count from the input
  tensor instead of hardcoding 3, so the same utility (no new warping
  implementation) can warp both RGB predictions and 2-channel flow fields
  (needed by the eval script's forward-backward consistency check).
- Architecture (4 transformer layers / 8 heads / dim 256 / pool 8), the AdamW
  two-group optimizer setup, and checkpointing are unchanged.
