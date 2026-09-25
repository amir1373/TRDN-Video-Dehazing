# Next TRDN Run: H1 Anti-Hallucination Latent Fidelity

## Goal

Run one clean, controlled TRDN experiment to test whether explicitly constraining the U-Net's predicted clean latent reduces hallucination and improves restoration fidelity/PSNR.

This run should be treated as a new experiment and must **not** overwrite Run2 or any existing retraining result.

## Experiment ID

**H1 — clean-x0 latent fidelity loss**

Baseline for comparison: **Run2 final checkpoint** (reported full-test PSNR 21.784 dB).

## Single intended change

Add a clean-latent x0 fidelity term to the existing training objective:

```python
latent_x0_loss = F.mse_loss(
    pred_x0.float(),
    clean_latents.detach().float(),
)
```

where:

- `pred_x0` is the clean latent reconstructed from the U-Net epsilon prediction using the existing `estimate_x0_from_epsilon(...)`.
- `clean_latents` is the VAE latent of the ground-truth clean target frame already computed during training.
- Do not backpropagate into the VAE.

Add a config weight:

```python
w_latent_x0: float = 0.5
```

and include it in `weighted_total_loss(...)`:

```python
total += config.w_latent_x0 * parts["latent_x0"]
```

Log it as `loss_latent_x0` in `metrics.jsonl`.

## Important: keep the experiment interpretable

For H1, do **not** simultaneously add RGB MSE, gradient loss, residual-TV, physics loss, LoRA, 512 px training, selector redesign, or other architecture changes. Those can be follow-up runs if H1 is useful.

Keep the rest of the Run2 loss configuration unchanged so the effect of the new anti-hallucination term is attributable.

## Initialization and optimizer

Initialize from the **Run2 final checkpoint**, but do not repeat the T0 restart mistake of using a fresh optimizer with the large constant LR.

Preferred continuation settings if a full optimizer/scheduler resume is not possible:

```text
UNet LR:       1e-6
Temporal LR:   1e-5
Schedule:      warmup + cosine
Warmup:        250 steps
Length:        2-5 epochs
Batch/crop:    same as Run2
```

Do not use the T0-style fresh constant `1e-5 / 1e-4` restart, because that continuation reduced PSNR substantially.

## Reproducibility requirement

Before launching H1, fix actual training seeding in `src/train.py`. The current `config.seed` is logged but is not sufficient by itself.

Seed at least:

```python
import random
import numpy as np
import torch

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
```

Also seed the DataLoader generator/worker initialization so `seed=1234` is a real reproducible training seed.

Primary H1 run: **seed 1234**.

If budget permits after the primary run, replicate with 2025 and 4242, but do not delay the primary result for those replicates.

## Evaluation

Use the same full-test evaluation protocol as Run2:

- full REVIDE test set
- same crop size
- same DDIM step count
- same inference seed
- same prompt/guidance
- report PSNR, SSIM, LPIPS, and temporal consistency
- do not select a checkpoint using test performance

For the cleanest comparison, evaluate **both Run2 and H1 with the exact same evaluator commit/settings** after H1 finishes.

Report at minimum:

```text
Run2 PSNR / SSIM / LPIPS
H1   PSNR / SSIM / LPIPS
H1 - Run2 delta
per-scene PSNR delta
```

## Success criterion

The primary question is whether the x0 fidelity constraint improves PSNR without causing a material SSIM or temporal-consistency regression.

A positive result should motivate follow-up ablations such as:

1. H2: RGB-MSE fidelity loss.
2. H3: x0 + RGB-MSE.
3. H4: x0 + residual-TV.
4. H5: residual/image-to-image latent diffusion.

Do not fold those into H1.


---

# Follow-up PSNR Roadmap: VideoEENet, Temporal Retrieval, and TRDN Masking

These experiments should remain separate from H1. H1 must stay a single-change clean-x0 latent-fidelity experiment.

## VideoEENet V8 — true identity-preserving residual output

Current `scripts/model.py` computes:

```python
return torch.sigmoid(residual + current)
```

This is not an identity residual path: if `residual == 0`, the output is `sigmoid(current)`, not `current`.

Test a logit-space residual:

```python
eps = 1e-4
current_safe = current.clamp(eps, 1 - eps)
current_logit = torch.logit(current_safe)
prediction = torch.sigmoid(current_logit + residual)
```

This guarantees zero predicted residual returns the hazy input exactly.

Keep all other V1 settings fixed for V8.

## VideoEENet V9 — PSNR-oriented loss

Current training uses pure L1 while headline quality is PSNR.

Run a separate experiment with:

```python
mse = F.mse_loss(prediction, target)
l1 = F.l1_loss(prediction, target)
loss = mse + 0.10 * l1
```

Log both components. Do not combine V8 and V9 in the first comparison; test each independently, then a V8+V9 combined run only if one or both are favorable.

## VideoEENet V10 — explicit temporal alignment before ConvLSTM

The current ConvLSTM receives unaligned frame features and must implicitly solve motion plus dehazing.

Add explicit alignment to the current frame before temporal fusion. Preferred order:

1. current frame is the reference,
2. estimate optical flow / alignment for each earlier hazy frame to current,
3. warp or flow-guide feature alignment,
4. encode/fuse aligned features,
5. ConvLSTM,
6. decoder.

Start with frozen RAFT for a controlled version. A later version may replace direct RGB warping with feature-space deformable alignment.

Keep a no-alignment control with identical loss and training schedule.

## VideoEENet V11 — fuller EENet backbone with multiscale skips

The current VideoEENet encoder/decoder is simpler than the full EENet implementation in `amir1373/EENet-Dehazing`.

A stronger VideoEENet-V2 can use:

```text
each frame
  -> full dual-domain EENet encoder
  -> temporal fusion at the bottleneck
  -> full EENet decoder
  -> current-frame multiscale skip features
  -> clean current frame
```

Preserve high-resolution current-frame features through skip connections so temporal modules correct haze without having to regenerate exact geometry from the bottleneck alone.

Treat this as a larger architectural experiment, not a small ablation.

## R1 — long-range scene/frame retrieval for BOTH models

Do not always use only the immediately preceding N frames.

For each target frame, search a larger causal buffer, initially the previous 30 frames:

```text
previous 30 hazy frames
  -> cheap matching/retrieval score
  -> top-K references (K ~ 8 or 9)
  -> temporal model
```

The score should use only hazy/input-side information, never clean GT at validation/test time. Candidate terms may include:

- structural/content similarity,
- optical-flow confidence,
- occlusion / forward-backward consistency,
- motion magnitude,
- temporal distance,
- an estimated haze severity score.

For online/causal reporting, search only past frames. If an offline bidirectional experiment is added, report it separately.

### Efficient implementation

Prefer precomputing candidate descriptors and top-K indices for every training/evaluation target frame. Do not run a 30-frame retrieval network inside every training step unless necessary.

For TRDN, retrieved references feed the existing RAFT -> temporal modules -> selector path.

For VideoEENet, retrieved references should preferably be aligned before ConvLSTM fusion.

## R2 — patch-level retrieval

Whole-frame matching may reject a frame that is useful only in part of the image.

A later version can retrieve per-region / per-patch temporal evidence:

```text
target patches
  -> retrieve useful aligned patches across temporal memory
  -> confidence-weighted fusion
```

This is a larger architecture change and should follow R1.

## TRDN M1 — learned soft haze-severity conditioning mask

In real dehazing mode TRDN currently uses an all-ones mask. In the current code path, the hazy conditioning image is still encoded and provided to the inpainting U-Net, so the mask channel can be repurposed as a spatial dehazing-strength map.

Use a soft map:

```text
M(x,y) in [0,1]
low  -> relatively clear region
high -> stronger haze / larger restoration required
```

### Training target for the mask estimator

During supervised training only, derive a pseudo-target haze map from paired hazy/clean frames, for example:

```python
difference = (hazy - clean).abs().mean(dim=1, keepdim=True)
pseudo_mask = smooth_and_normalize(difference)
```

Train a small `HazeMapEstimator` from the hazy current frame to this pseudo-target.

At inference, only the hazy frame is available:

```text
hazy current frame
  -> HazeMapEstimator
  -> soft haze mask
  -> TRDN inpainting conditioning channel
```

Do not use rectangle/ellipse/blob/Perlin masks for real dehazing; those remain legacy synthetic-occlusion tools.

## TRDN M2 — transmission map + physics consistency

A more physically interpretable follow-up is to predict transmission `t(x)` and use:

```text
haze severity M(x) = 1 - t(x)
```

Optionally add a re-hazing consistency term using:

```text
I_hazy ~= J_pred * t + A * (1 - t)
```

where `J_pred` is the predicted clean image and `A` is estimated atmospheric light.

Keep this separate from M1 because it changes both the conditioning map and the objective.

# A40 timing estimates

These are planning estimates, not measured A40 benchmarks for the new variants.

Measured September reference:

- TRDN full model: ~2.29 s/step while sharing an RTX A6000, 4,840 steps for five epochs, ~3.6 h/run including full evaluation.
- VideoEENet: ~2.5 min/epoch in the same campaign, ~75 min for 30 epochs.
- Dedicated A40 and A6000 performance for these workloads is expected to be in the same broad range; use the first 100-200 steps of every new run to replace the estimates with measured ETA.

Assume 256x256 crops and the current dataset unless stated otherwise.

| Experiment | Assumed training length | Estimated dedicated A40 wall time | Notes |
|---|---:|---:|---|
| H1 clean-x0 TRDN | 2 epochs | ~1.5-2.0 h incl. eval | Extra loss is cheap; diffusion dominates |
| H1 clean-x0 TRDN | 5 epochs | ~3.5-4.0 h incl. eval | Closest comparison to prior 5-epoch continuation runs |
| H2 RGB-MSE TRDN | 5 epochs | ~3.5-4.0 h | Negligible extra compute |
| H3 x0 + RGB-MSE TRDN | 5 epochs | ~3.5-4.1 h | Negligible extra compute |
| H4 x0 + residual-TV TRDN | 5 epochs | ~3.6-4.2 h | Small tensor-op overhead |
| V8 VideoEENet residual fix | 30 epochs | ~1.0-1.5 h | Same architecture/throughput as V1 |
| V9 VideoEENet MSE+L1 | 30 epochs | ~1.0-1.5 h | Loss overhead negligible |
| V8+V9 combined | 30 epochs | ~1.0-1.5 h | Run only after individual ablations |
| V10 VideoEENet + frozen RAFT alignment | 30 epochs | ~2-3 h | Depends strongly on whether flow is cached/precomputed |
| V11 full-EENet multiscale VideoEENet | 30 epochs | ~2-4 h | Larger encoder/decoder; benchmark first epoch |
| R1 VideoEENet top-K retrieval | 30 epochs | ~1.2-2 h if top-K is precomputed | Online 30-frame search could push this toward ~2-3 h |
| R1 TRDN top-K retrieval | 5 epochs | ~3.8-4.5 h if top-K is precomputed | Online retrieval may increase to ~5-8 h |
| R2 patch-level retrieval VideoEENet | 30 epochs | ~2-4 h | Architecture-dependent |
| R2 patch-level retrieval TRDN | 5 epochs | ~5-8 h | Architecture-dependent; benchmark before full run |
| M1 TRDN learned soft haze mask | 5 epochs | ~3.8-4.5 h | Small mask CNN should add modest overhead |
| M2 TRDN transmission + physics loss | 5 epochs | ~4-5 h | Extra estimator + re-hazing loss |
| TRDN 512x512 follow-up | 5 epochs | ~10-16 h | Latent spatial area is 4x vs 256; memory may also force slower settings |

## Recommended execution order

For best information per GPU dollar:

1. H1: clean-x0 TRDN.
2. V8: identity-preserving VideoEENet residual.
3. V9: MSE + 0.1 L1 VideoEENet.
4. If V8 or V9 helps, run V8+V9.
5. M1: learned TRDN soft haze map.
6. R1: precomputed long-range top-K retrieval, first on VideoEENet, then TRDN.
7. V10: explicit alignment before ConvLSTM.
8. V11 / R2 / M2 only after the cheaper experiments establish the direction.

Do not combine multiple untested changes into one first run. Every first-pass experiment should have a matched control, fixed seeds, identical data split, and identical evaluator.
