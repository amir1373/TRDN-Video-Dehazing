# TRDN Accuracy Audit

This audit records quality-sensitive choices without changing the default
measurement. Options added here are default-off unless they preserve existing
behavior exactly.

| Area | Current setting | Concern | Recommendation |
| --- | --- | --- | --- |
| VAE ceiling | Predictions pass through the Stable Diffusion VAE. | VAE round-trip error caps PSNR, SSIM, and LPIPS regardless of training quality. | Run `scripts/vae_ceiling.py` at the training resolution before training. It uses the deterministic posterior mode and reports the three metrics. |
| Resolution | `crop_size=256`; latent size is 32x32. | The backbone was designed around 512x512 inputs. Moving to 512 makes the latent 64x64, four times as many spatial positions. GPU memory and step time will rise substantially, but the exact cost is implementation-dependent. | Benchmark 512 separately on the A40. Do not mix 256 and 512 results in one ablation table. |
| EMA | Disabled (`enable_ema=false`). | Fine-tuned diffusion weights can benefit from smoothing. | Measure `enable_ema=true` with an explicit `ema_decay`. EMA state is checkpointed and exported as `ema_weights.pt`; evaluate only with explicit `--use-ema`. |
| LR schedule | Constant (`lr_schedule=constant`), no warmup. | Early updates may be abrupt and a constant terminal LR may leave quality unrealized. | Measure default-off `warmup_cosine` with an explicit `lr_warmup_steps`; keep the constant run as the controlled baseline. |
| Batch/LR coupling | LR is unchanged when batch size changes. | A benchmark-selected larger effective batch changes optimization dynamics. | Either keep LR fixed for a pure systems comparison or explicitly enable linear scaling with a stated reference batch. Never compare the two as the same training ablation. |
| Inference steps | Training config default and notebook evaluation use 30 DDIM steps. | More steps can improve restoration but cost more and are not guaranteed to help monotonically. | Use `--step-sweep` to write complete reports per measured step count. The script deliberately does not select the best test result automatically. |
| Classifier-free guidance | `guidance_scale=1.0`, equivalent to no CFG second pass. | Guidance can trade fidelity for sharper but hallucinated content. | Sweep explicitly. Values other than 1.0 run conditional and unconditional U-Net passes; retain 1.0 unless validation supports a change. |
| Text conditioning | Exact prompt: `a clear clean dehazed video frame`. | The prompt is hand-authored rather than learned or dataset-derived and may bias output. | Treat prompt changes as measured validation ablations. The prompt is now configurable and recorded in checkpoint/evaluation metadata. |
| Checkpoint selection | `best_psnr` and `best_ssim` are tracked on the held-out training validation split. The RunPod driver evaluates `best_psnr`. | There is no early stopping, and the two best checkpoints can differ. | State the selected metric in reporting. Do not select on test metrics. Add early stopping only as a separately reviewed training-policy change. |
| Validation fidelity | Defaults are 4 validation batches and 10 DDIM steps every 500 training steps. | A small, lower-step proxy can rank checkpoints differently from the final 30-step full validation behavior. | Increase `validation_max_batches` and `validation_num_inference_steps` if budget permits, then keep them fixed across ablations. |
| LPIPS backbone | AlexNet LPIPS. | This is standard but only one perceptual measure and can disagree with restoration fidelity. | Keep it for comparability; inspect PSNR, SSIM, temporal error, and qualitative outputs together. |
| Full U-Net tuning | Full U-Net training is enabled; LoRA is disabled. | Full tuning may improve capacity but costs memory and can overfit a small dataset. | Compare LoRA only as an explicit training ablation with identical evaluation and numerics. |

No A40 timing, throughput, or memory values are claimed here. Those values
must come from `scripts/benchmark.py` on the rented GPU.
