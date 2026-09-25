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
