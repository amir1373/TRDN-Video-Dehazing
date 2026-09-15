# TRDN Loss Report

This project trains TRDN from REVIDE sequences only. No LAION, captioned image dataset, or separate Stable Diffusion dataset is required.

## 1. Diffusion Noise Loss

Stable Diffusion inpainting is trained in latent space. The clean target frame is encoded by the frozen VAE:

```text
target [B,3,H,W] -> latents [B,4,H/8,W/8]
```

Gaussian noise is sampled and added at random scheduler timesteps. The inpainting UNet predicts the noise residual. The loss is:

```text
MSE(noise_pred, noise)
```

This is the main diffusion objective.

## 2. L1 Reconstruction Loss

The predicted clean latent estimate is decoded through the frozen VAE and compared to the clean target frame:

```text
L1(pred_image, target_clean)
```

This stabilizes visible reconstruction quality and discourages latent-only improvements that do not improve decoded images.

## 3. LPIPS Loss

LPIPS compares perceptual features between the decoded prediction and clean target:

```text
LPIPS(pred_image, target_clean)
```

If LPIPS is unavailable in the environment, the code returns a zero tensor and documents that perceptual loss is disabled.

## 4. Temporal Consistency Loss

Reference weights are applied to warped previous frames:

```text
weighted_reference = sum_i weights_i * warped_reference_i
```

The decoded prediction is encouraged to remain consistent with the selected temporal evidence:

```text
L1(pred_image, weighted_reference.detach())
```

The weighted reference is detached so this term stabilizes reconstruction without collapsing reference selection into a trivial copy objective.

## 5. Flow Consistency Loss

The weighted warped reference is compared with the current frame:

```text
L1(weighted_reference, current_frame.detach())
```

This penalizes unreliable alignment and gives the reference selector a signal about flow/warp agreement.

## 6. Reference Preservation Loss

Outside masked/corrupted regions, TRDN should preserve reliable temporal information:

```text
mean(abs(pred_image - weighted_reference.detach()) * (1 - mask))
```

This supports the reconstruct training mode, where masked regions should be regenerated but unmasked regions should remain stable.

## Weighted Sum

The final training loss is:

```text
total =
  w_diffusion * diffusion_noise
+ w_l1        * l1_reconstruction
+ w_lpips     * lpips
+ w_temporal  * temporal_consistency
+ w_flow      * flow_consistency
+ w_reference * reference_preservation
```

Default weights live in `src/config.py` and can be changed from the notebook or scripts.
