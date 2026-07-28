import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .assertions import (
    assert_frames,
    assert_image,
    assert_latents,
    assert_mask,
    assert_reference_weights,
    assert_temporal_memory,
    assert_warped_references,
)
from .config import TRDNConfig
from .convlstm import TemporalMemoryModule
from .dataset import REVIDESequenceDataset
from .diffusion_adapter import estimate_x0_from_epsilon, get_text_embeddings, prepare_inpainting_inputs, decode_latents_to_images, encode_images_to_latents
from .ema import EMAState
from .flow import compute_warped_references_batch, load_raft
from .losses import LossBundle, weighted_total_loss
from .provenance import (
    JsonlMetricLogger,
    checkpoint_metadata,
    ensure_project_config_compatible,
    create_run_manifest,
    find_numerics_mismatches,
    find_seed_mismatches,
    make_run_dir,
    mean_records,
    peak_gpu_memory_bytes,
    prune_step_checkpoints,
    update_manifest,
    validate_checkpoint_modes,
    write_json,
)
from .progress import ProgressReporter
from .reference_selector import ReferenceSelectionModule
from .diffusion_adapter import TemporalConditioningAdapter, load_diffusion_backbone
from .temporal_transformer import TemporalRetrievalTransformer
from .validate import validate_trdn
from .warp import warp_with_flow


def make_datasets(
    config: TRDNConfig,
    *,
    validate_structure: bool = True,
) -> Tuple[REVIDESequenceDataset, REVIDESequenceDataset]:
    train_dataset = REVIDESequenceDataset(
        config.root_for_split(config.train_split),
        split=config.train_split,
        seq_len=config.seq_len,
        crop_size=config.crop_size,
        random_crop=True,
        extensions=config.image_extensions,
        synthetic_if_empty=False,
        train_mode=config.train_mode,
        mask_mode=config.mask_mode,
        val_fraction=config.val_fraction,
        split_seed=config.split_seed,
    )
    val_dataset = REVIDESequenceDataset(
        config.root_for_split(config.val_split),
        split=config.val_split,
        seq_len=config.seq_len,
        crop_size=config.crop_size,
        random_crop=False,
        extensions=config.image_extensions,
        synthetic_if_empty=False,
        train_mode=config.train_mode,
        mask_mode=config.mask_mode,
        val_fraction=config.val_fraction,
        split_seed=config.split_seed,
        include_prev_frame=False,  # validation only ever runs infer_dehazed_batch, which doesn't use it
    )
    if validate_structure:
        train_dataset.assert_valid_structure("train")
        val_dataset.assert_valid_structure("validation")
    return train_dataset, val_dataset


def make_dataloaders(config: TRDNConfig) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = make_datasets(config)
    worker_options = (
        {
            "persistent_workers": config.persistent_workers,
            "prefetch_factor": config.prefetch_factor,
        }
        if config.num_workers > 0
        else {}
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **worker_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def make_test_dataset_for_manifest(config: TRDNConfig) -> REVIDESequenceDataset:
    """Discover test sequences and windows for inventory only.

    No samples are loaded, and synthetic fallback is disabled so the manifest
    cannot turn an absent test set into a plausible-looking clip count.
    """
    return REVIDESequenceDataset(
        config.root_for_split("test"),
        split="test",
        seq_len=config.seq_len,
        crop_size=config.crop_size,
        random_crop=False,
        extensions=config.image_extensions,
        synthetic_if_empty=False,
        train_mode=config.train_mode,
        mask_mode=config.mask_mode,
        include_prev_frame=False,
    )


def build_temporal_modules(
    config: TRDNConfig, cross_attention_dim: int, device: str
) -> Tuple[torch.nn.Module, torch.nn.Module | None, torch.nn.Module, torch.nn.Module]:
    temporal_memory = TemporalMemoryModule(hidden_dim=64).to(device)
    temporal_transformer = (
        TemporalRetrievalTransformer(
            memory_dim=64,
            token_dim=config.transformer_token_dim,
            num_layers=config.transformer_num_layers,
            num_heads=config.transformer_num_heads,
            pool_size=config.transformer_pool_size,
            max_seq_len=config.seq_len,
        ).to(device)
        if config.use_temporal_transformer
        else None
    )
    reference_selector = ReferenceSelectionModule(num_references=config.seq_len - 1).to(device)
    conditioning_adapter = TemporalConditioningAdapter(cross_attention_dim=cross_attention_dim, num_tokens=16).to(device)
    return temporal_memory, temporal_transformer, reference_selector, conditioning_adapter


def build_optimizer(config: TRDNConfig, unet, temporal_memory, temporal_transformer, reference_selector, conditioning_adapter):
    learning_rate, temporal_learning_rate = effective_learning_rates(config)
    groups = []
    if config.train_unet:
        groups.append({"params": [p for p in unet.parameters() if p.requires_grad], "lr": learning_rate})
    if config.train_temporal_modules:
        temporal_params = list(temporal_memory.parameters()) + list(reference_selector.parameters()) + list(conditioning_adapter.parameters())
        if temporal_transformer is not None:
            temporal_params += list(temporal_transformer.parameters())
        groups.append({"params": temporal_params, "lr": temporal_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def effective_learning_rates(config: TRDNConfig) -> tuple[float, float]:
    if not config.enable_linear_lr_scaling:
        return config.learning_rate, config.temporal_learning_rate
    if config.lr_reference_batch_size <= 0:
        raise ValueError("lr_reference_batch_size must be positive.")
    scale = (
        config.batch_size
        * config.gradient_accumulation_steps
        / config.lr_reference_batch_size
    )
    return config.learning_rate * scale, config.temporal_learning_rate * scale


def apply_optional_lr_schedule(
    config: TRDNConfig,
    optimizer: torch.optim.Optimizer,
    base_learning_rates: list[float],
    step: int,
    total_steps: int,
) -> None:
    if config.lr_schedule == "constant":
        return
    if config.lr_schedule != "warmup_cosine":
        raise ValueError(f"Unsupported lr_schedule={config.lr_schedule!r}")
    if config.lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative.")
    if step <= config.lr_warmup_steps and config.lr_warmup_steps > 0:
        factor = step / config.lr_warmup_steps
    else:
        decay_steps = max(total_steps - config.lr_warmup_steps, 1)
        progress = min(
            max((step - config.lr_warmup_steps) / decay_steps, 0.0),
            1.0,
        )
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group, base_lr in zip(optimizer.param_groups, base_learning_rates):
        group["lr"] = base_lr * factor


def forward_window_prediction(
    accelerator: Accelerator,
    diffusion: Dict[str, Any],
    temporal_memory: torch.nn.Module,
    temporal_transformer: torch.nn.Module | None,
    reference_selector: torch.nn.Module,
    conditioning_adapter: torch.nn.Module,
    raft_model: torch.nn.Module | None,
    frames: torch.Tensor,
    mask: torch.Tensor,
    corrupted: torch.Tensor,
    target: torch.Tensor,
    seq_len: int,
    autocast: bool = True,
    timing: Dict[str, float] | None = None,
    text_prompt: str = "a clear clean dehazed video frame",
) -> Dict[str, torch.Tensor]:
    """Run the full temporal + diffusion stack for one [B,T,3,H,W] window.

    Shared by the main (current-frame) training step and, in dehaze mode, the
    extra previous-frame pass used to compute a true predictive temporal
    consistency loss (see predictive_temporal_consistency_loss in src/losses.py).
    """
    assert_frames(frames, seq_len=seq_len)
    assert_mask(mask, target)
    current = frames[:, -1]

    with torch.no_grad():
        if timing is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            raft_started = time.perf_counter()
        warped_refs, flows = compute_warped_references_batch(frames, raft_model)
        if timing is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            timing["raft_seconds"] = timing.get("raft_seconds", 0.0) + (
                time.perf_counter() - raft_started
            )
    assert_warped_references(warped_refs, seq_len=seq_len)

    aligned_frames = torch.cat([warped_refs, current.unsqueeze(1)], dim=1)
    ctx = accelerator.autocast() if autocast else nullcontext()
    with ctx:
        memory = temporal_memory(aligned_frames)
        prior_logits = None
        if temporal_transformer is not None:
            transformer_out = temporal_transformer(aligned_frames, memory)
            memory = transformer_out["enhanced_memory"]
            prior_logits = transformer_out["reference_prior_logits"]
        assert_temporal_memory(memory, batch=frames.shape[0])
        ref = reference_selector(warped_refs, memory, prior_logits=prior_logits)
        assert_reference_weights(ref["weights"], seq_len=seq_len)
        cond_tokens = conditioning_adapter(memory, ref["reference_feature"])
        text = get_text_embeddings(
            diffusion["tokenizer"],
            diffusion["text_encoder"],
            frames.shape[0],
            prompt=text_prompt,
        ).to(cond_tokens.dtype)
        encoder_hidden_states = torch.cat([text, cond_tokens], dim=1)

        with torch.no_grad():
            latents = encode_images_to_latents(diffusion["vae"], target)
        assert_latents(latents, target)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, diffusion["noise_scheduler"].config.num_train_timesteps, (latents.shape[0],), device=latents.device
        ).long()
        noisy_latents = diffusion["noise_scheduler"].add_noise(latents, noise, timesteps)
        model_input = prepare_inpainting_inputs(diffusion["vae"], noisy_latents, mask, corrupted)
        noise_pred = diffusion["unet"](model_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())
        pred_x0 = estimate_x0_from_epsilon(diffusion["noise_scheduler"], noisy_latents, timesteps, noise_pred)
        pred_img = decode_latents_to_images(diffusion["vae"], pred_x0)

    return {
        "pred_img": pred_img,
        "diffusion_loss": diffusion_loss,
        "warped_refs": warped_refs,
        "flows": flows,
        "weights": ref["weights"],
        "weighted_reference": ref["weighted_reference"],
        "current": current,
    }


def compute_training_loss(
    accelerator: Accelerator,
    diffusion: Dict[str, Any],
    temporal_memory: torch.nn.Module,
    temporal_transformer: torch.nn.Module | None,
    reference_selector: torch.nn.Module,
    conditioning_adapter: torch.nn.Module,
    raft_model: torch.nn.Module | None,
    loss_bundle: LossBundle,
    batch: Dict[str, Any],
    config: TRDNConfig,
    temporal_loss_enabled: bool = True,
    timing: Dict[str, float] | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    frames = batch["frames"].to(accelerator.device, non_blocking=True)
    target = batch["target_frame"].to(accelerator.device, non_blocking=True)
    mask = batch["mask"].to(accelerator.device, non_blocking=True)
    corrupted = batch["corrupted_frame"].to(accelerator.device, non_blocking=True)

    current_out = forward_window_prediction(
        accelerator,
        diffusion,
        temporal_memory,
        temporal_transformer,
        reference_selector,
        conditioning_adapter,
        raft_model,
        frames,
        mask,
        corrupted,
        target,
        seq_len=config.seq_len,
        timing=timing,
        text_prompt=config.text_prompt,
    )
    pred_img = current_out["pred_img"]
    warped_refs = current_out["warped_refs"]
    flows = current_out["flows"]
    current = current_out["current"]

    if not temporal_loss_enabled:
        temporal_loss = pred_img.new_zeros(())
    elif config.train_mode == "dehaze" and "prev_frames" in batch:
        prev_frames = batch["prev_frames"].to(accelerator.device, non_blocking=True)
        prev_target = batch["prev_target_frame"].to(accelerator.device, non_blocking=True)
        prev_mask = batch["prev_mask"].to(accelerator.device, non_blocking=True)
        prev_corrupted = batch["prev_corrupted_frame"].to(accelerator.device, non_blocking=True)
        with torch.no_grad():
            prev_out = forward_window_prediction(
                accelerator,
                diffusion,
                temporal_memory,
                temporal_transformer,
                reference_selector,
                conditioning_adapter,
                raft_model,
                prev_frames,
                prev_mask,
                prev_corrupted,
                prev_target,
                seq_len=config.seq_len,
                timing=timing,
                text_prompt=config.text_prompt,
            )
        temporal_loss = loss_bundle.predictive_temporal_consistency_loss(
            pred_img, prev_out["pred_img"], flows[:, -1]
        )
    else:
        temporal_loss = loss_bundle.legacy_temporal_consistency_loss(
            pred_img, warped_refs, current_out["weights"]
        )

    with accelerator.autocast():
        parts = {
            "diffusion": current_out["diffusion_loss"],
            "l1": F.l1_loss(pred_img, target),
            "lpips": loss_bundle.lpips_loss(pred_img, target),
            "temporal": temporal_loss,
            "flow": loss_bundle.flow_consistency_loss(
                warped_refs, current, current_out["weights"]
            ),
        }
        if config.w_reference != 0.0:
            parts["reference"] = loss_bundle.reference_preservation_loss(
                pred_img, current_out["weighted_reference"], mask
            )
        total_loss = weighted_total_loss(config, parts)
    return total_loss, parts


def save_checkpoint(
    accelerator: Accelerator,
    checkpoint_dir: Path,
    step: int,
    best_psnr: float,
    best_ssim: float,
    config: TRDNConfig,
    run_manifest_path: Path,
    name: str | None = None,
    ema: EMAState | None = None,
    validation_state: Dict[str, Any] | None = None,
) -> None:
    out_dir = checkpoint_dir / (name or f"step_{step:06d}")
    accelerator.save_state(str(out_dir))
    if accelerator.is_main_process:
        write_json(
            out_dir / "metadata.json",
            checkpoint_metadata(
                config,
                step,
                best_psnr,
                best_ssim,
                run_manifest_path,
                validation_state,
            ),
        )
        if ema is not None:
            ema.save_weights(out_dir / "ema_weights.pt")
        if name is None:
            prune_step_checkpoints(
                checkpoint_dir,
                config.keep_last_n_checkpoints,
                config.always_keep_best,
            )


def _step_metric_record(
    parts: Dict[str, torch.Tensor],
    total_loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
) -> Dict[str, Any]:
    def finite_scalar(value: torch.Tensor) -> float | None:
        scalar = float(value.detach().cpu())
        return scalar if math.isfinite(scalar) else None

    loss_names = ("diffusion", "l1", "lpips", "temporal", "flow", "reference")
    learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    record: Dict[str, Any] = {
        "event": "step",
        "step": step,
        "epoch": epoch,
        "total_loss": finite_scalar(total_loss),
        "lr": learning_rates[0],
        "learning_rates": learning_rates,
    }
    for name in loss_names:
        value = parts.get(name)
        record[f"loss_{name}"] = finite_scalar(value) if value is not None else None
    return record


def nonfinite_loss_terms(
    parts: Dict[str, torch.Tensor],
    total_loss: torch.Tensor,
) -> list[str]:
    values = {**parts, "total": total_loss}
    return [
        name
        for name, value in values.items()
        if not bool(torch.isfinite(value.detach()).all().item())
    ]


def train_trdn(config: TRDNConfig) -> Dict[str, Any]:
    if config.validation_num_samples <= 0:
        raise ValueError("validation_num_samples must be positive.")
    if config.validation_num_inference_steps <= 0:
        raise ValueError("validation_num_inference_steps must be positive.")
    if config.checkpoint_selection_metric not in {"psnr", "ssim"}:
        raise ValueError(
            "checkpoint_selection_metric must be either 'psnr' or 'ssim'."
        )
    if config.enable_early_stopping and config.early_stopping_patience <= 0:
        raise ValueError(
            "early_stopping_patience must be positive when early stopping is enabled."
        )
    seed_mismatches = find_seed_mismatches(Path(config.paths()["logs"]), config)
    print(f"Resolved training seed: {config.seed}")
    if seed_mismatches:
        print("=" * 88)
        print("WARNING: SEED DIFFERS FROM A SIBLING RUN")
        print(json.dumps(seed_mismatches, indent=2, sort_keys=True))
        print("=" * 88)
    ensure_project_config_compatible(config)
    paths = config.ensure_dirs()
    resume_metadata: Dict[str, Any] = {}
    if config.resume_from_checkpoint:
        # Validate before model construction and, critically, before weights load.
        resume_metadata = validate_checkpoint_modes(config.resume_from_checkpoint, config)

    resumed_manifest_path = (
        Path(str(resume_metadata.get("run_manifest_path", "")))
        if resume_metadata.get("run_manifest_path")
        else None
    )
    if resumed_manifest_path is not None and resumed_manifest_path.is_file():
        run_dir = resumed_manifest_path.parent
    else:
        run_dir = make_run_dir(Path(paths["logs"]), config.run_name)
    manifest_path = run_dir / "run_manifest.json"
    metric_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")
    numerics_mismatches = find_numerics_mismatches(Path(paths["logs"]), config)
    if numerics_mismatches:
        print("=" * 88)
        print("WARNING: NUMERICS SETTINGS DIFFER FROM AN EXISTING RUN")
        print(json.dumps(numerics_mismatches, indent=2, sort_keys=True))
        print("Ablation results are not directly comparable unless numerics match.")
        print("=" * 88)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=str(run_dir),
    )
    accelerator.init_trackers("TRDN_REVIDE")
    if accelerator.is_main_process:
        write_json(run_dir / "config.json", config.to_dict())

    diffusion = load_diffusion_backbone(config, device=device)
    temporal_memory, temporal_transformer, reference_selector, conditioning_adapter = build_temporal_modules(
        config, diffusion["unet"].config.cross_attention_dim, device
    )
    loss_bundle = LossBundle(device=device)
    optimizer = build_optimizer(
        config,
        diffusion["unet"],
        temporal_memory,
        temporal_transformer,
        reference_selector,
        conditioning_adapter,
    )
    train_loader, val_loader = make_dataloaders(config)
    test_dataset = make_test_dataset_for_manifest(config)
    raft_model = (
        load_raft(
            device,
            config.freeze_raft,
            config.validate_raft_flow,
            config.raft_max_flow_factor,
        )
        if config.use_raft_alignment and torch.cuda.is_available()
        else None
    )

    modules = {
        "unet": diffusion["unet"],
        "temporal_memory": temporal_memory,
        "temporal_transformer": temporal_transformer,
        "reference_selector": reference_selector,
        "conditioning_adapter": conditioning_adapter,
    }
    ema = EMAState(modules, config.ema_decay) if config.enable_ema else None
    if ema is not None:
        accelerator.register_for_checkpointing(ema)
    base_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    prior_manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if accelerator.is_main_process and not (
        resumed_manifest_path is not None and resumed_manifest_path.is_file()
    ):
        manifest = create_run_manifest(
            manifest_path,
            config,
            {
                "train": train_loader.dataset,
                "val": val_loader.dataset,
                "test": test_dataset,
            },
            modules,
            optimizer,
        )
        if numerics_mismatches:
            manifest["numerics_mismatch_warnings"] = numerics_mismatches
            write_json(manifest_path, manifest)
        print("TRDN RUN MANIFEST")
        print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    elif accelerator.is_main_process:
        resumes = list(prior_manifest.get("resumes", []))
        resumes.append(
            {
                "checkpoint": str(Path(config.resume_from_checkpoint).resolve()),
                "from_step": int(resume_metadata.get("step", 0)),
                "resumed_at_unix": time.time(),
            }
        )
        update_manifest(
            manifest_path,
            {
                "status": "running",
                "resumes": resumes,
                "seed_mismatch_warnings": seed_mismatches,
            },
        )

    if temporal_transformer is not None:
        (
            diffusion["unet"],
            temporal_memory,
            temporal_transformer,
            reference_selector,
            conditioning_adapter,
            optimizer,
            train_loader,
        ) = accelerator.prepare(
            diffusion["unet"],
            temporal_memory,
            temporal_transformer,
            reference_selector,
            conditioning_adapter,
            optimizer,
            train_loader,
        )
    else:
        (
            diffusion["unet"],
            temporal_memory,
            reference_selector,
            conditioning_adapter,
            optimizer,
            train_loader,
        ) = accelerator.prepare(
            diffusion["unet"],
            temporal_memory,
            reference_selector,
            conditioning_adapter,
            optimizer,
            train_loader,
        )
    diffusion["vae"].to(accelerator.device)
    diffusion["text_encoder"].to(accelerator.device)
    if raft_model is not None:
        raft_model.to(accelerator.device).eval()

    global_step = int(resume_metadata.get("step", 0))
    best_psnr = float(resume_metadata.get("best_psnr", -1.0))
    best_ssim = float(resume_metadata.get("best_ssim", -1.0))
    resumed_selection = resume_metadata.get("checkpoint_selection", {})
    early_stopping_bad_validation_count = int(
        resumed_selection.get("early_stopping_bad_validation_count", 0)
    )
    validation_state: Dict[str, Any] = {
        "num_samples": int(resumed_selection.get("validation_num_samples", 0)),
        "num_inference_steps": int(
            resumed_selection.get(
                "validation_num_inference_steps",
                config.validation_num_inference_steps,
            )
        ),
        "seed": int(
            resumed_selection.get("validation_seed", config.validation_seed)
        ),
        "step": resumed_selection.get("validation_step"),
        "early_stopping_bad_validation_count": early_stopping_bad_validation_count,
    }
    validation_history = list(prior_manifest.get("validation_passes", []))
    stopped_early = False
    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)

    target_train_steps = (
        config.max_train_steps
        if config.max_train_steps and config.max_train_steps > 0
        else config.num_epochs * len(train_loader)
    )
    progress = ProgressReporter(
        total=target_train_steps,
        initial=global_step,
        desc=f"Training TRDN (epoch 1/{config.num_epochs})",
        leave=True,
        position=0,
        enabled=accelerator.is_main_process,
    )
    prior_training = {}
    if manifest_path.exists():
        prior_training = json.loads(manifest_path.read_text(encoding="utf-8")).get("training", {})
    non_finite_loss_steps = int(prior_training.get("non_finite_loss_steps", 0))
    non_finite_loss_terms_count = int(prior_training.get("non_finite_loss_terms", 0))
    optimizer_steps_skipped = int(prior_training.get("optimizer_steps_skipped", 0))
    for epoch_index in range(config.num_epochs):
        epoch = epoch_index + 1
        progress.set_description(f"Training TRDN (epoch {epoch}/{config.num_epochs})")
        epoch_records = []
        epoch_validation: Dict[str, float] = {}
        for batch in train_loader:
            if global_step >= target_train_steps:
                break
            attempted_step = global_step + 1
            apply_optional_lr_schedule(
                config,
                optimizer,
                base_learning_rates,
                attempted_step,
                target_train_steps,
            )
            grad_norm = None
            overflow_step_skipped = False
            with accelerator.accumulate(diffusion["unet"]):
                total_loss, parts = compute_training_loss(
                    accelerator,
                    diffusion,
                    temporal_memory,
                    temporal_transformer,
                    reference_selector,
                    conditioning_adapter,
                    raft_model,
                    loss_bundle,
                    batch,
                    config,
                )
                bad_terms = nonfinite_loss_terms(parts, total_loss)
                if bad_terms:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(
                            list(diffusion["unet"].parameters())
                            + list(temporal_memory.parameters())
                            + ([] if temporal_transformer is None else list(temporal_transformer.parameters()))
                            + list(reference_selector.parameters())
                            + list(conditioning_adapter.parameters()),
                            config.max_grad_norm,
                        )
                    optimizer.step()
                    overflow_step_skipped = bool(
                        accelerator.sync_gradients
                        and getattr(accelerator, "optimizer_step_was_skipped", False)
                    )
                    if (
                        ema is not None
                        and accelerator.sync_gradients
                        and not overflow_step_skipped
                    ):
                        ema.update(
                            {
                                "unet": accelerator.unwrap_model(diffusion["unet"]),
                                "temporal_memory": accelerator.unwrap_model(temporal_memory),
                                "temporal_transformer": (
                                    accelerator.unwrap_model(temporal_transformer)
                                    if temporal_transformer is not None
                                    else None
                                ),
                                "reference_selector": accelerator.unwrap_model(reference_selector),
                                "conditioning_adapter": accelerator.unwrap_model(conditioning_adapter),
                            }
                        )
                    optimizer.zero_grad(set_to_none=True)

            global_step += 1
            step_record = _step_metric_record(parts, total_loss, optimizer, global_step, epoch)
            step_record["nonfinite_terms"] = bad_terms
            step_record["optimizer_step_skipped"] = overflow_step_skipped
            progress.set_postfix(
                {
                    "loss": (
                        f"{step_record['total_loss']:.5f}"
                        if step_record["total_loss"] is not None
                        else "nonfinite"
                    ),
                    "lr": f"{step_record['lr']:.3e}",
                }
            )
            progress.update(1)
            epoch_records.append(step_record)
            if accelerator.is_main_process:
                metric_logger.append(step_record)
                if bad_terms:
                    non_finite_loss_steps += 1
                    non_finite_loss_terms_count += len(bad_terms)
                    metric_logger.append(
                        {
                            "event": "nonfinite_loss",
                            "step": attempted_step,
                            "epoch": epoch,
                            "terms": bad_terms,
                        }
                    )
                    progress.write(
                        f"NON-FINITE LOSS: step={attempted_step} terms={','.join(bad_terms)}; "
                        "optimizer update skipped."
                    )
                if overflow_step_skipped:
                    optimizer_steps_skipped += 1
                    metric_logger.append(
                        {
                            "event": "optimizer_step_skipped",
                            "step": attempted_step,
                            "epoch": epoch,
                            "reason": "accelerate_gradient_scaler_overflow",
                            "mixed_precision": config.mixed_precision,
                        }
                    )
                    progress.write(
                        f"OPTIMIZER STEP SKIPPED: step={attempted_step} "
                        "Accelerate reported gradient-scaler overflow."
                    )

            if accelerator.is_main_process and global_step % config.log_every == 0:
                tracker_logs = {
                    f"train/{key.removeprefix('loss_')}_loss": value
                    for key, value in step_record.items()
                    if key.startswith("loss_") and value is not None
                }
                if step_record["total_loss"] is not None:
                    tracker_logs["train/total_loss"] = step_record["total_loss"]
                tracker_logs["safety/non_finite_loss_steps"] = non_finite_loss_steps
                tracker_logs["safety/optimizer_steps_skipped"] = optimizer_steps_skipped
                if grad_norm is not None:
                    tracker_logs["train/grad_norm"] = float(
                        grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm
                    )
                accelerator.log(tracker_logs, step=global_step)

            if global_step % config.validate_every == 0:
                validation_started = time.perf_counter()
                metrics = validate_trdn(
                    val_loader,
                    diffusion,
                    temporal_memory,
                    temporal_transformer,
                    reference_selector,
                    conditioning_adapter,
                    loss_bundle,
                    str(accelerator.device),
                    raft_model=raft_model,
                    num_samples=config.validation_num_samples,
                    num_steps=config.validation_num_inference_steps,
                    seed=config.validation_seed,
                    text_prompt=config.text_prompt,
                    guidance_scale=config.guidance_scale,
                )
                validation_wall_clock_seconds = float(
                    metrics.get(
                        "wall_clock_seconds",
                        time.perf_counter() - validation_started,
                    )
                )
                validation_samples = int(
                    metrics.get(
                        "num_samples",
                        min(config.validation_num_samples, len(val_loader.dataset)),
                    )
                )
                epoch_validation = {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }
                accelerator.log(
                    {f"val/{key}": value for key, value in epoch_validation.items()},
                    step=global_step,
                )
                selected_metric = config.checkpoint_selection_metric
                selected_best_before = (
                    best_psnr if selected_metric == "psnr" else best_ssim
                )
                selected_value = epoch_validation[selected_metric]
                selected_improved = selected_value > selected_best_before
                psnr_improved = epoch_validation["psnr"] > best_psnr
                ssim_improved = epoch_validation["ssim"] > best_ssim
                if psnr_improved:
                    best_psnr = epoch_validation["psnr"]
                if ssim_improved:
                    best_ssim = epoch_validation["ssim"]
                if selected_improved:
                    early_stopping_bad_validation_count = 0
                else:
                    early_stopping_bad_validation_count += 1
                stopped_early = (
                    config.enable_early_stopping
                    and early_stopping_bad_validation_count
                    >= config.early_stopping_patience
                )
                validation_state = {
                    "step": global_step,
                    "num_samples": validation_samples,
                    "num_inference_steps": int(
                        metrics.get(
                            "num_inference_steps",
                            config.validation_num_inference_steps,
                        )
                    ),
                    "seed": int(metrics.get("seed", config.validation_seed)),
                    "wall_clock_seconds": validation_wall_clock_seconds,
                    "unet_forward_passes": int(
                        metrics.get(
                            "unet_forward_passes",
                            validation_samples
                            * config.validation_num_inference_steps
                            * (2 if config.guidance_scale != 1.0 else 1),
                        )
                    ),
                    "selection_metric": selected_metric,
                    "selection_value": selected_value,
                    "selection_improved": selected_improved,
                    "early_stopping_bad_validation_count": (
                        early_stopping_bad_validation_count
                    ),
                }
                validation_history.append(dict(validation_state))
                if accelerator.is_main_process:
                    metric_logger.append(
                        {
                            "event": "validation",
                            "step": global_step,
                            "epoch": epoch,
                            **{f"val_{key}": value for key, value in epoch_validation.items()},
                            "selection_metric": selected_metric,
                            "selection_value": selected_value,
                            "selection_improved": selected_improved,
                        }
                    )
                    update_manifest(
                        manifest_path,
                        {
                            "checkpoint_selection": {
                                "metric": selected_metric,
                                "checkpoint_name": f"best_{selected_metric}",
                                "current_value": (
                                    best_psnr
                                    if selected_metric == "psnr"
                                    else best_ssim
                                ),
                                "validation_num_samples_configured": (
                                    config.validation_num_samples
                                ),
                                "validation_num_samples_actual": validation_samples,
                                "validation_num_inference_steps": (
                                    config.validation_num_inference_steps
                                ),
                                "validation_seed": config.validation_seed,
                                "early_stopping_enabled": (
                                    config.enable_early_stopping
                                ),
                                "early_stopping_patience": (
                                    config.early_stopping_patience
                                ),
                                "early_stopping_bad_validation_count": (
                                    early_stopping_bad_validation_count
                                ),
                                "stopped_early": stopped_early,
                            },
                            "validation_passes": validation_history,
                        },
                    )
                if psnr_improved:
                    save_checkpoint(
                        accelerator,
                        Path(paths["checkpoints"]),
                        global_step,
                        best_psnr,
                        best_ssim,
                        config,
                        manifest_path,
                        "best_psnr",
                        ema,
                        validation_state,
                    )
                if ssim_improved:
                    save_checkpoint(
                        accelerator,
                        Path(paths["checkpoints"]),
                        global_step,
                        best_psnr,
                        best_ssim,
                        config,
                        manifest_path,
                        "best_ssim",
                        ema,
                        validation_state,
                    )

            if global_step % config.checkpoint_every == 0:
                save_checkpoint(
                    accelerator,
                    Path(paths["checkpoints"]),
                    global_step,
                    best_psnr,
                    best_ssim,
                    config,
                    manifest_path,
                    ema=ema,
                    validation_state=validation_state,
                )
            if stopped_early:
                progress.write(
                    "EARLY STOPPING: "
                    f"metric={config.checkpoint_selection_metric} "
                    f"patience={config.early_stopping_patience} "
                    f"step={global_step}"
                )
                break

        if accelerator.is_main_process and epoch_records:
            aggregate_keys = [
                "loss_diffusion",
                "loss_l1",
                "loss_lpips",
                "loss_temporal",
                "loss_flow",
                "loss_reference",
                "total_loss",
                "lr",
            ]
            metric_logger.append(
                {
                    "event": "epoch",
                    "step": global_step,
                    "epoch": epoch,
                    **mean_records(epoch_records, aggregate_keys),
                    **{f"val_{key}": value for key, value in epoch_validation.items()},
                }
            )
        if stopped_early or global_step >= target_train_steps:
            break

    save_checkpoint(
        accelerator,
        Path(paths["checkpoints"]),
        global_step,
        best_psnr,
        best_ssim,
        config,
        manifest_path,
        "last",
        ema,
        validation_state,
    )
    elapsed_seconds = time.perf_counter() - started
    selected_best = (
        best_psnr if config.checkpoint_selection_metric == "psnr" else best_ssim
    )
    result = {
        "step": float(global_step),
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "selection_metric": config.checkpoint_selection_metric,
        "selection_value": selected_best if selected_best >= 0 else None,
        "stopped_early": stopped_early,
    }
    if accelerator.is_main_process:
        update_manifest(
            manifest_path,
            {
                "status": "completed",
                "finished_at_unix": time.time(),
                "training": {
                    "wall_clock_seconds": elapsed_seconds,
                    "peak_gpu_memory_bytes": peak_gpu_memory_bytes(),
                    "non_finite_loss_steps": non_finite_loss_steps,
                    "non_finite_loss_terms": non_finite_loss_terms_count,
                    "optimizer_steps_skipped": optimizer_steps_skipped,
                    "checkpoint_selection_metric": (
                        config.checkpoint_selection_metric
                    ),
                    "checkpoint_selection_value": (
                        selected_best if selected_best >= 0 else None
                    ),
                    "validation_num_samples": validation_state["num_samples"],
                    "validation_num_inference_steps": validation_state[
                        "num_inference_steps"
                    ],
                    "validation_seed": validation_state["seed"],
                    "stopped_early": stopped_early,
                    "ema": (
                        {
                            "enabled": True,
                            "decay": ema.decay,
                            "num_updates": ema.num_updates,
                        }
                        if ema is not None
                        else {"enabled": False}
                    ),
                    "result": result,
                },
            },
        )
    accelerator.end_training()
    progress.close()
    return result


def _run_temporal_stack(frames: torch.Tensor, seq_len: int, device: str) -> Dict[str, Any]:
    warped_refs, flows = compute_warped_references_batch(frames, raft_model=None)
    memory_module = TemporalMemoryModule(hidden_dim=64).to(device)
    transformer = TemporalRetrievalTransformer(
        max_seq_len=seq_len,
        pool_size=4,
        token_dim=128,
        num_layers=1,
        num_heads=4,
    ).to(device)
    selector = ReferenceSelectionModule(num_references=seq_len - 1).to(device)
    adapter = TemporalConditioningAdapter(cross_attention_dim=768, num_tokens=16).to(device)
    aligned = torch.cat([warped_refs, frames[:, -1:].contiguous()], dim=1)
    memory = memory_module(aligned)
    transformer_out = transformer(aligned, memory)
    memory = transformer_out["enhanced_memory"]
    ref = selector(warped_refs, memory, prior_logits=transformer_out["reference_prior_logits"])
    tokens = adapter(memory, ref["reference_feature"])
    return {
        "warped_refs": warped_refs,
        "flows": flows,
        "memory": memory,
        "transformer_out": transformer_out,
        "ref": ref,
        "tokens": tokens,
    }


def dry_run_shape_test(
    seq_len: int = 10, image_size: int = 64, batch_size: int = 1, train_mode: str = "dehaze", mask_mode: str = "auto"
) -> Dict[str, tuple]:
    """Exercise dataset construction (synthetic fallback, no real REVIDE needed)
    plus the temporal stack for a given train_mode/mask_mode, without loading
    Stable Diffusion. Must pass for both "dehaze" and "reconstruct_synthetic".
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = REVIDESequenceDataset(
        root="__dry_run_no_such_path__",
        split="train",
        seq_len=seq_len,
        crop_size=image_size,
        random_crop=False,
        synthetic_if_empty=True,
        train_mode=train_mode,
        mask_mode=mask_mode,
    )
    batch = {
        key: (value.unsqueeze(0).to(device) if torch.is_tensor(value) else [value]) for key, value in dataset[0].items()
    }
    frames = batch["frames"]
    mask = batch["mask"]
    current_stack = _run_temporal_stack(frames, seq_len, device)

    result = {
        "train_mode": dataset.train_mode,
        "mask_mode": dataset._resolve_mask_mode(),
        "frames": tuple(frames.shape),
        "current_hazy": tuple(frames[:, -1].shape),
        "target_clean": tuple(batch["target_frame"].shape),
        "mask": tuple(mask.shape),
        "warped_references": tuple(current_stack["warped_refs"].shape),
        "flows": tuple(current_stack["flows"].shape),
        "temporal_memory": tuple(current_stack["memory"].shape),
        "transformer_tokens": tuple(current_stack["transformer_out"]["tokens"].shape),
        "reference_weights": tuple(current_stack["ref"]["weights"].shape),
        "conditioning_tokens": tuple(current_stack["tokens"].shape),
    }

    if "prev_frames" in batch:
        prev_stack = _run_temporal_stack(batch["prev_frames"], seq_len, device)
        # Verify the predictive temporal-consistency wiring's shapes line up:
        # a placeholder "previous prediction" warped by the current window's
        # t-1 -> t flow must match the current prediction's shape exactly.
        placeholder_pred_current = frames[:, -1]
        placeholder_pred_prev = batch["prev_frames"][:, -1]
        warped_prev = warp_with_flow(placeholder_pred_prev, current_stack["flows"][:, -1])
        assert_image(warped_prev, channels=3, name="warped_prev_prediction")
        if tuple(warped_prev.shape) != tuple(placeholder_pred_current.shape):
            raise ValueError(
                f"predictive temporal consistency shape mismatch: {tuple(warped_prev.shape)} vs {tuple(placeholder_pred_current.shape)}"
            )
        result["prev_frames"] = tuple(batch["prev_frames"].shape)
        result["prev_target_clean"] = tuple(batch["prev_target_frame"].shape)
        result["prev_warped_references"] = tuple(prev_stack["warped_refs"].shape)
        result["predictive_temporal_warp"] = tuple(warped_prev.shape)

    return result
