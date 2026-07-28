# Temporal Reference-Guided Diffusion Network (TRDN) for Video Dehazing

Research code for **Temporal Reference-Guided Diffusion Network for Video Dehazing using REVIDE**.

TRDN reconstructs a clean current frame from a hazy 10-frame sequence:

```text
[frame(t-9), ..., frame(t-1), frame(t)] -> clean frame(t)
```

The primary workflow is a VS Code notebook connected to a Google Colab GPU runtime. The reusable implementation lives in `src/`, and the notebook is kept in `notebooks/TRDN_REVIDE_Colab.ipynb`.

## Notebook / Repository Synchronization

The notebook is the execution interface, not a second implementation. All major research logic lives in `src/` and is imported by the notebook:

- datasets and REVIDE parsing: `src/dataset.py`
- masks and haze simulation: `src/masks.py`, `src/haze.py`
- RAFT and warping: `src/flow.py`, `src/warp.py`
- temporal modules: `src/convlstm.py`, `src/temporal_transformer.py`
- reference selection: `src/reference_selector.py`
- diffusion and conditioning: `src/diffusion_adapter.py`
- losses and metrics: `src/losses.py`, `src/metrics.py`
- training, validation, inference: `src/train.py`, `src/validate.py`, `src/inference.py`
- notebook execution helpers: `src/notebook_utils.py`

Future code changes should be made in `src/` first. The notebook should remain limited to setup, configuration, visualization, debugging, and launch cells.

## Research Motivation

Single-image dehazing often loses temporal information that is available in video. TRDN V1 uses previous hazy frames as temporal references, aligns them to the current frame with optical flow, learns which aligned references are reliable per pixel, and injects temporal/reference conditioning into a Stable Diffusion inpainting UNet.

## Architecture Overview

TRDN:

```text
Current hazy frame + previous 9 frames
        |
        v
Pretrained RAFT optical flow, frozen initially
        |
        v
Warp previous frames into current-frame coordinates
        |
        v
ConvLSTM TemporalMemoryModule
        |
        v
Temporal Retrieval Transformer
        |
        v
ReferenceSelectionModule with per-pixel reference weights
        |
        v
TemporalConditioningAdapter -> Stable Diffusion inpainting cross-attention tokens
        |
        v
Clean current-frame reconstruction
```

The repository also includes a clean placeholder for a future **Temporal Retrieval Transformer** version.

## Dataset Setup

The code expects REVIDE on Google Drive and lets you set `DATASET_ROOT` manually.

Supported layouts include:

```text
REVIDE/
  train/
    sequence_001/
      hazy/
        000001.png
      gt/
        000001.png
  val/
    sequence_001/
      hazy/
      gt/
```

and:

```text
REVIDE/
  train/
    hazy/
      sequence_001/
    gt/
      sequence_001/
  val/
    hazy/
      sequence_001/
    gt/
      sequence_001/
```

Recognized clean folder names include `gt`, `clean`, `clear`, `target`, and `groundtruth`. Recognized hazy folder names include `hazy`, `input`, `fog`, and `degraded`.

## Google Drive Setup

1. Upload or keep REVIDE in Google Drive, for example:

   ```text
   /content/drive/MyDrive/REVIDE
   ```

2. Paths are centralized in `src/config.py`:

   ```text
   TRAIN_ROOT = /content/drive/MyDrive/REVIDE_sequences/Train
   TEST_ROOT  = /content/drive/MyDrive/REVIDE_sequences/Test
   TRAIN_HAZY = /content/drive/MyDrive/REVIDE_sequences/Train/hazy
   TEST_HAZY  = /content/drive/MyDrive/REVIDE_sequences/Test/hazy
   FLOW_TRAIN = /content/drive/MyDrive/video_dehaze_flows/train
   FLOW_VAL   = /content/drive/MyDrive/video_dehaze_flows/val
   ```

3. Project outputs default to:

   ```text
   /content/drive/MyDrive/TRDN_REVIDE
   ```

## Colab Runtime Setup

Use a GPU runtime:

```text
Runtime -> Change runtime type -> GPU
```

Then run the notebook installation cell or install manually:

```bash
pip install -r requirements_colab.txt
```

Initial defaults:

```text
IMAGE_SIZE = 256
SEQ_LEN = 10
BATCH_SIZE = 1
NUM_WORKERS = 2
MIXED_PRECISION = "fp16"
```

## How to Run the Notebook

Open `notebooks/TRDN_REVIDE_Colab.ipynb` in VS Code, connect it to a Google Colab runtime, and use `Runtime -> Restart and Run All`.

The notebook is organized as 18 run-all-safe cells:

```text
1. Project Overview
2. Install Dependencies
3. Mount Google Drive
4. Clone/Open Repository
5. Set Paths
6. Imports from src/
7. Dataset Inspection
8. Dataset Visualization
9. Flow Visualization
10. Mask Visualization
11. Debug Forward Pass
12. Training Configuration
13. Training Launch
14. Validation
15. Inference
16. Attention Visualization
17. Reference Weight Visualization
18. Checkpoint Management
```

The notebook includes dedicated debug cells for:

- dataset inspection
- mask generation
- haze simulation
- RAFT flow
- warped references
- ConvLSTM memory
- reference weights
- diffusion output
- dry-run tensor shape checks

Run `notebooks/DATASET_INSPECTOR.ipynb` before training to verify REVIDE sequence discovery, frame counts, and image sizes.

## Training Modes

`src/config.py` exposes:

```python
train_mode = "dehaze"  # or "reconstruct_synthetic" (legacy alias: "reconstruct")
```

`dehaze` (default):

```text
input      = real hazy current frame + previous real hazy frames
references = real hazy frames
target     = real clean current frame
mask_mode  = "full" (all-ones; full-frame restoration through the SD inpainting interface)
```

This is the actual video dehazing task and the only mode whose numbers should
be reported as a dehazing result.

`reconstruct_synthetic` (legacy name: `reconstruct`):

```text
input      = clean target frame with SYNTHETIC haze painted inside a random mask
references = GROUND-TRUTH CLEAN frames
target     = clean current frame
mask_mode  = "mixed" (random rectangle/ellipse/blob/perlin occlusion, unrelated to real haze)
```

Real REVIDE haze never reaches the model in this mode, and the temporal
references are ground truth. **This is not a dehazing evaluation.** It is
kept only to explain/reproduce previously-reported numbers, is not the
default, and logs a loud warning whenever it is selected. See "Evaluation
protocol" below.

Both modes use REVIDE only.

## How to Train

Notebook:

1. Set `DATASET_ROOT`.
2. Run setup/debug cells.
3. Set `cfg.run_training_now = True`.
4. Run the training cell.

Script:

```bash
python scripts/train_colab.py \
  --project-root /content/drive/MyDrive/TRDN_REVIDE \
  --max-train-steps 1000
```

Resume:

```bash
python scripts/train_colab.py \
  --resume-from-checkpoint /content/drive/MyDrive/TRDN_REVIDE/checkpoints/last
```

Resume validates `train_mode` and the resolved `mask_mode` from checkpoint
metadata before loading weights. Use `--allow-mode-mismatch` only for a
deliberate override. Numbered checkpoints retain the newest three by default;
set `--keep-last-n-checkpoints` to change that count. Named `best_*`
checkpoints are never pruned.

Every run creates `logs/runs/<run>/run_manifest.json` and an append-only
`metrics.jsonl`. The manifest records the resolved config, git state, split
inventory, parameter counts, runtime environment, elapsed time, and peak GPU
memory.

Before a billed training run, execute the real-data/real-weight preflight:

```bash
python scripts/preflight.py \
  --checkpoint /content/drive/MyDrive/TRDN_REVIDE/checkpoints/last \
  --dataset-root /content/drive/MyDrive/REVIDE_sequences \
  --project-root /content/drive/MyDrive/TRDN_REVIDE
```

Preflight refuses synthetic fallback and writes `preflight_report.json`.

## How to Validate

```bash
python scripts/validate_colab.py \
  --checkpoint /content/drive/MyDrive/TRDN_REVIDE/checkpoints/best_psnr
```

Validation reports PSNR, SSIM, and LPIPS on the **validation split** (a
held-out subset of training sequences -- see "Evaluation protocol" below), and
uses seeded DDIM inference (eta=0) by default so results are reproducible run
to run. It does not fake or include training results, and it never reads test
data.

## Evaluation Protocol

- **Only `train_mode="dehaze"` measures video dehazing.** It is the default in
  `src/config.py`. `train_mode="reconstruct_synthetic"` (legacy name
  `"reconstruct"`) uses ground-truth clean frames as temporal references and
  paints synthetic haze inside a random mask onto an otherwise-clean target;
  real REVIDE haze never reaches the model in that mode. It is **not a
  dehazing benchmark** -- it exists only to explain/reproduce previously
  reported numbers, and `src/dataset.py` logs a loud warning whenever it is
  selected.
- **The validation split is a held-out subset of TRAINING sequences, never
  the test set.** `src/dataset.py`'s `split_train_val_sequence_names()`
  deterministically partitions ~10% of training sequence names (by name, via
  a seeded hash, so no clip from a given sequence appears on both sides) into
  a validation split, seeded by `config.split_seed` (independent of the
  general training `seed`, so changing the run seed never reshuffles which
  sequences are held out). `config.root_for_split("val")` resolves to
  `train_root`, not `test_root`.
- **Test data is never used for model selection.** `config.test_root` is only
  meant to be read by `scripts/evaluate_full_test.py`, a final, one-shot,
  fully seeded evaluation over the *entire* test set with no sample
  filtering, scoring, or "good candidate" selection of any kind. Checkpoints
  must be selected using validation metrics only (as `src/train.py` already
  does), never by peeking at test-set performance.
- **Reported metrics**: PSNR, SSIM, LPIPS, plus a flow-warped temporal
  consistency error (RAFT-warp prediction t-1 into t, mask invalid/occluded
  pixels via forward-backward flow consistency, report mean L1 on the valid
  pixels) -- not a naive inter-frame difference, since a naive diff conflates
  real motion with actual flicker.
- **Determinism**: inference uses DDIM with eta=0 and a per-sample,
  per-frame-index deterministic noise generator (`src/seeding.py`), so two
  runs with the same seed produce identical metrics. Random per-run noise
  previously caused a 1.67 dB spread between two runs of the same
  configuration; this is now fixed by default.

### Running the full test-set evaluation

```bash
python scripts/evaluate_full_test.py \
  --checkpoint /content/drive/MyDrive/TRDN_REVIDE/checkpoints/best_psnr \
  --num-steps 30 \
  --seed 1234
```

Writes a JSON report (per-clip and aggregate PSNR/SSIM/LPIPS/temporal
consistency, plus `N`, `seed`, `num_inference_steps`, `checkpoint_path`,
`train_mode`, `mask_mode`, and the current git commit hash) next to the
checkpoint by default, or to `--output`.

Paper artifacts must be generated from that JSON:

```bash
python scripts/make_paper_figures.py \
  --checkpoint /content/drive/MyDrive/TRDN_REVIDE/checkpoints/best_psnr \
  --eval-json /content/drive/MyDrive/TRDN_REVIDE/evaluate_full_test.json \
  --dataset-root /content/drive/MyDrive/REVIDE_sequences \
  --output-dir /content/drive/MyDrive/TRDN_REVIDE/paper_artifacts

python scripts/emit_paper_tables.py \
  --eval-json /content/drive/MyDrive/TRDN_REVIDE/evaluate_full_test.json \
  --output-dir /content/drive/MyDrive/TRDN_REVIDE/paper_artifacts
```

Metric charts and table values are read only from evaluation JSON. Qualitative
indices are a seeded draw over the complete test-window index and are recorded
with every figure sidecar; no quality-based filtering is performed.

### Diffusion-only baseline (isolating the temporal contribution)

```bash
python scripts/evaluate_full_test.py \
  --checkpoint /content/drive/MyDrive/TRDN_REVIDE/checkpoints/best_psnr \
  --num-steps 30 --seed 1234 --diffusion-only
```

Runs the SD inpainting backbone frame-by-frame with no RAFT, ConvLSTM,
temporal transformer, or reference selector, reusing the same eval script and
JSON schema, so the temporal stack's contribution can be measured by
comparing against the full-pipeline run above.

## How to Run Inference

```bash
python scripts/inference_colab.py \
  --checkpoint /content/drive/MyDrive/TRDN_REVIDE/checkpoints/best_psnr \
  --index 0
```

Predictions are written to `outputs/` under the configured project root.

## Debug Forward Pass

Fast local/CPU smoke test:

```bash
python debug_forward_pass.py --skip-diffusion --no-raft
```

Full Colab GPU debug pass:

```bash
python debug_forward_pass.py
```

Outputs are written to `debug_outputs/` and include masks, warped references, flow RGB, temporal memory maps, reference maps, parameter counts, memory usage, and diffusion predictions when diffusion is enabled.

## Repository Structure

```text
TRDN-Video-Dehazing/
  README.md
  LICENSE
  .gitignore
  requirements.txt
  requirements_colab.txt
  notebooks/
    TRDN_REVIDE_Colab.ipynb
  src/
    __init__.py
    config.py
    dataset.py
    masks.py
    haze.py
    flow.py
    warp.py
    convlstm.py
    reference_selector.py
    temporal_transformer.py
    diffusion_adapter.py
    losses.py
    metrics.py
    seeding.py
    train.py
    validate.py
    inference.py
  scripts/
    train_colab.py
    validate_colab.py
    inference_colab.py
    evaluate_full_test.py
  tests/
  outputs/
  checkpoints/
  logs/
  visualizations/
  debug_outputs/
```

## Troubleshooting

- **No REVIDE clips found:** Check `TRAIN_ROOT`, `TEST_ROOT`, and folder names in `src/config.py`. The notebook falls back to synthetic debug samples so shape tests can still run.
- **CUDA out of memory:** Keep `BATCH_SIZE=1`, `IMAGE_SIZE=256`, mixed precision enabled, and UNet gradient checkpointing enabled. Disable RAFT alignment for quick tests.
- **RAFT download is slow:** The first run downloads torchvision RAFT weights. Subsequent Colab sessions may need to download again unless cached.
- **Stable Diffusion download/auth issues:** Ensure the Colab runtime has internet access. If Hugging Face authentication is required in your environment, run `huggingface-cli login` without committing tokens.
- **VS Code cannot find `src`:** Run the notebook from the repository root or add the repo root to `sys.path`, as done in the notebook setup cell.
- **Checkpoint loading mismatch:** Use checkpoints saved by this repository version, especially when loading with Accelerate.

## Citation Placeholder

If you use this research code, cite the project and REVIDE dataset. A formal BibTeX entry can be added after publication:

```bibtex
@misc{trdn_video_dehazing,
  title={Temporal Reference-Guided Diffusion Network for Video Dehazing},
  author={Your Name},
  year={2026},
  note={Research code}
}
```
