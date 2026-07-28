# Changelog

## Unreleased -- Fix dehaze/eval protocol

An audit found that the trained/evaluated configuration did not measure video
dehazing at all, plus several related issues. This change fixes the pipeline
to measure the intended task, without changing or inventing any previously
reported numbers.

### Runtime safety and A40 preparation

- LPIPS initialization now fails loudly and runs a non-zero startup probe
  instead of silently replacing perceptual loss with zero.
- Training logs every non-finite loss term and Accelerate-reported scaler
  overflow, skips the affected update, and records cumulative totals in the
  run manifest.
- Optional RAFT flow sanity checks reject non-finite or frame-implausible
  flow, and bf16 now loads frozen diffusion modules at the requested dtype.
- Added a strict TODO-only A40 numerics preset, matching run/checkpoint
  provenance, and warnings when an output directory contains runs with
  different numerics.
- Replaced the committed notebook with a `/workspace` RunPod driver that
  invokes preflight, training, full evaluation, figures, and tables in order.
- Added the CUDA-only `scripts/benchmark.py` harness. It records measured
  throughput, memory, numerical changes, scaler skips, data wait, RAFT share,
  and compile warmup without inventing results on CPU-only systems.

### Neutral summary

- Default training mode is now dehaze: temporal references and the current
  frame are taken from real hazy inputs.
- Validation is a seeded holdout of training sequences; test data is no
  longer used for model selection.
- Temporal consistency compares the current prediction against the warped
  previous prediction.
- Full-test evaluation uses seeded deterministic sampling and a flow-warped
  temporal consistency metric, with a diffusion-only frame-by-frame baseline.
- Run manifests, on-disk metric logs, checkpoint retention, reporting
  scripts, and real-data preflight checks make run provenance and cost
  explicit.

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
- Checkpoint metadata records the resolved modes, seed, git SHA, dataset root,
  temporal/crop dimensions, effective loss weights, and originating run
  manifest. Resume validates mode compatibility before loading weights.
- Each training run writes an isolated `run_manifest.json` and append-only
  `metrics.jsonl`; evaluation appends measured wall-clock and peak-memory
  usage to the same manifest.
- `scripts/preflight.py` performs real-data/reference-integrity checks,
  real-weight seed determinism checks, parameter/data inventory, temporal
  loss overhead timing, schedule estimates, and checkpoint storage estimates.
- `scripts/make_paper_figures.py` and `scripts/emit_paper_tables.py` generate
  deterministic figures and direction-aware tables directly from evaluation
  JSON rather than hand-transcribed values.

### Changed

- `src/warp.py`: `warp_with_flow` now infers channel count from the input
  tensor instead of hardcoding 3, so the same utility (no new warping
  implementation) can warp both RGB predictions and 2-channel flow fields
  (needed by the eval script's forward-backward consistency check).
- Architecture (4 transformer layers / 8 heads / dim 256 / pool 8), the AdamW
  two-group optimizer setup, and Accelerate's checkpoint contents are
  unchanged. Numbered `step_*` directories now retain only the configured
  newest count (default 3); named `best_*` checkpoints are never pruned.
