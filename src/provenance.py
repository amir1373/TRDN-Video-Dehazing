import json
import hashlib
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch

from .config import TRDNConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
LOSS_WEIGHT_FIELDS = (
    "w_diffusion",
    "w_l1",
    "w_lpips",
    "w_temporal",
    "w_flow",
    "w_reference",
)
NUMERICS_FIELDS = (
    "mixed_precision",
    "allow_tf32",
    "cudnn_benchmark",
    "attention_backend",
    "batch_size",
    "enable_unet_gradient_checkpointing",
    "enable_torch_compile",
    "channels_last",
)
RESUME_MUTABLE_CONFIG_FIELDS = {
    "resume_from_checkpoint",
    "allow_mode_mismatch",
    "allow_output_collision",
    "max_train_steps",
    "num_epochs",
    "run_name",
    "log_every",
    "validate_every",
    "checkpoint_every",
    "keep_last_n_checkpoints",
    "always_keep_best",
}


def _run_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def git_state() -> Dict[str, Any]:
    try:
        commit = _run_git("rev-parse", "HEAD")
        dirty = bool(_run_git("status", "--porcelain"))
    except Exception:
        commit, dirty = "unknown", None
    return {"commit_sha": commit, "dirty": dirty}


def effective_mask_mode(train_mode: str, mask_mode: str) -> str:
    if mask_mode != "auto":
        return mask_mode
    return "full" if train_mode == "dehaze" else "mixed"


def loss_weights(config: TRDNConfig) -> Dict[str, float]:
    return {name.removeprefix("w_"): float(getattr(config, name)) for name in LOSS_WEIGHT_FIELDS}


def numerics_settings(config: TRDNConfig) -> Dict[str, Any]:
    return {name: getattr(config, name) for name in NUMERICS_FIELDS}


def comparable_run_config(config: TRDNConfig | Mapping[str, Any]) -> Dict[str, Any]:
    values = config.to_dict() if isinstance(config, TRDNConfig) else dict(config)
    return {
        key: value
        for key, value in values.items()
        if key not in RESUME_MUTABLE_CONFIG_FIELDS
    }


def config_fingerprint(config: TRDNConfig | Mapping[str, Any]) -> str:
    encoded = json.dumps(
        comparable_run_config(config),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_project_config_compatible(config: TRDNConfig) -> Path:
    root = Path(config.project_root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "project_config.json"
    current = {
        "config_fingerprint": config_fingerprint(config),
        "config": comparable_run_config(config),
    }
    existing_sources = []
    if marker.exists():
        existing_sources.append((marker, json.loads(marker.read_text(encoding="utf-8"))))
    for manifest_path in sorted((root / "logs" / "runs").glob("*/run_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Output directory collision check could not read {manifest_path}: {exc}"
            ) from exc
        if not isinstance(manifest.get("config"), dict):
            raise RuntimeError(
                f"Output directory collision check found no config in {manifest_path}."
            )
        existing_config = comparable_run_config(manifest["config"])
        existing_sources.append(
            (
                manifest_path,
                {
                    "config_fingerprint": config_fingerprint(existing_config),
                    "config": existing_config,
                },
            )
        )

    for source_path, existing in existing_sources:
        if existing.get("config_fingerprint") != current["config_fingerprint"]:
            existing_config = existing.get("config", {})
            differences = {
                key: {"existing": existing_config.get(key), "current": value}
                for key, value in current["config"].items()
                if existing_config.get(key) != value
            }
            if not config.allow_output_collision:
                raise RuntimeError(
                    "Output directory collision: project_root already belongs to a different "
                    f"configuration ({source_path}). Use a unique project root for each ablation. "
                    "Only pass --allow-output-collision after reviewing these differences: "
                    + json.dumps(differences, sort_keys=True, default=str)
                )
    if not marker.exists():
        write_json(marker, current)
    return marker


def checkpoint_metadata(
    config: TRDNConfig,
    step: int,
    best_psnr: float,
    best_ssim: float,
    run_manifest_path: Path,
) -> Dict[str, Any]:
    state = git_state()
    return {
        "step": int(step),
        "best_psnr": float(best_psnr),
        "best_ssim": float(best_ssim),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_mode": config.train_mode,
        "mask_mode": effective_mask_mode(config.train_mode, config.mask_mode),
        "configured_mask_mode": config.mask_mode,
        "seed": int(config.seed),
        "git_commit_sha": state["commit_sha"],
        "git_dirty": state["dirty"],
        "dataset_root": config.dataset_root,
        "seq_len": int(config.seq_len),
        "crop_size": int(config.crop_size),
        "loss_weights": loss_weights(config),
        "numerics": numerics_settings(config),
        "config_fingerprint": config_fingerprint(config),
        "quality_settings": {
            "text_prompt": config.text_prompt,
            "guidance_scale": config.guidance_scale,
            "enable_ema": config.enable_ema,
            "ema_decay": config.ema_decay,
            "lr_schedule": config.lr_schedule,
            "lr_warmup_steps": config.lr_warmup_steps,
            "enable_linear_lr_scaling": config.enable_linear_lr_scaling,
            "lr_reference_batch_size": config.lr_reference_batch_size,
        },
        "checkpoint_retention": {
            "keep_last_n_checkpoints": int(config.keep_last_n_checkpoints),
            "always_keep_best": bool(config.always_keep_best),
        },
        "run_dir": str(run_manifest_path.parent.resolve()),
        "run_manifest_path": str(run_manifest_path.resolve()),
    }


def prune_step_checkpoints(
    checkpoint_dir: Path,
    keep_last_n: int,
    always_keep_best: bool = True,
) -> list[Path]:
    if keep_last_n < 0:
        raise ValueError(f"keep_last_n_checkpoints must be >= 0, got {keep_last_n}")
    if not always_keep_best:
        raise ValueError(
            "always_keep_best=False is not permitted: best_* checkpoints are never deleted."
        )
    # Retention is intentionally scoped to step_* directories. best_* is never
    # a candidate.
    candidates = []
    for path in checkpoint_dir.glob("step_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("step_")
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    removed = []
    for _step, path in candidates[keep_last_n:]:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def load_checkpoint_metadata(checkpoint_path: str | Path) -> Dict[str, Any]:
    metadata_path = Path(checkpoint_path) / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint metadata is missing at {metadata_path}. "
            "Refusing to load weights whose train_mode and mask_mode cannot be verified."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def validate_checkpoint_modes(
    checkpoint_path: str | Path,
    config: TRDNConfig,
    allow_mode_mismatch: bool | None = None,
) -> Dict[str, Any]:
    allow = config.allow_mode_mismatch if allow_mode_mismatch is None else allow_mode_mismatch
    try:
        metadata = load_checkpoint_metadata(checkpoint_path)
    except FileNotFoundError:
        if allow:
            return {}
        raise

    saved_train_mode = metadata.get("train_mode")
    saved_mask_mode = metadata.get("mask_mode")
    current_mask_mode = effective_mask_mode(config.train_mode, config.mask_mode)
    missing = [
        name
        for name, value in (("train_mode", saved_train_mode), ("mask_mode", saved_mask_mode))
        if value is None
    ]
    mismatches = []
    if saved_train_mode is not None and saved_train_mode != config.train_mode:
        mismatches.append(f"train_mode checkpoint={saved_train_mode!r} current={config.train_mode!r}")
    if saved_mask_mode is not None and saved_mask_mode != current_mask_mode:
        mismatches.append(f"mask_mode checkpoint={saved_mask_mode!r} current={current_mask_mode!r}")

    if (missing or mismatches) and not allow:
        details = []
        if missing:
            details.append(f"missing metadata fields: {', '.join(missing)}")
        details.extend(mismatches)
        raise ValueError(
            "Checkpoint mode validation failed: "
            + "; ".join(details)
            + ". Pass --allow-mode-mismatch only for a deliberate, documented override."
        )
    return metadata


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def make_run_dir(logs_root: Path, run_name: str = "") -> Path:
    if run_name:
        name = run_name
    else:
        state = git_state()
        short_sha = str(state["commit_sha"])[:8]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{timestamp}_{short_sha}"
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"run_name must be a single directory name, got {run_name!r}")
    runs_root = logs_root / "runs"
    run_dir = runs_root / name
    suffix = 1
    while run_dir.exists() and any(run_dir.iterdir()):
        run_dir = runs_root / f"{name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def dataset_size(dataset: Any) -> Dict[str, Any]:
    real_clips = len(getattr(dataset, "index", []))
    synthetic_clips = int(getattr(dataset, "synthetic_len", 0))
    return {
        "num_sequences": len(getattr(dataset, "sequences", [])),
        "num_clips": real_clips,
        "effective_num_samples": len(dataset),
        "uses_synthetic_fallback": real_clips == 0 and synthetic_clips > 0,
    }


def trainable_parameter_counts(
    modules: Mapping[str, torch.nn.Module | None],
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, Any]:
    optimized_parameter_ids = (
        {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        if optimizer is not None
        else None
    )
    counts = {
        name: (
            sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
                and (
                    optimized_parameter_ids is None
                    or id(parameter) in optimized_parameter_ids
                )
            )
            if module is not None
            else 0
        )
        for name, module in modules.items()
    }
    return {"by_module": counts, "total": sum(counts.values())}


def runtime_environment(mixed_precision: str) -> Dict[str, Any]:
    try:
        import diffusers

        diffusers_version = diffusers.__version__
    except Exception:
        diffusers_version = "unavailable"
    return {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "diffusers_version": diffusers_version,
        "mixed_precision": mixed_precision,
    }


def create_run_manifest(
    path: Path,
    config: TRDNConfig,
    datasets: Mapping[str, Any],
    modules: Mapping[str, torch.nn.Module | None],
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "run_dir": str(path.parent.resolve()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "git": git_state(),
        "config": config.to_dict(),
        "dataset_sizes": {name: dataset_size(dataset) for name, dataset in datasets.items()},
        "trainable_parameters": trainable_parameter_counts(modules, optimizer),
        "environment": runtime_environment(config.mixed_precision),
        "numerics": numerics_settings(config),
        "metrics_log": str((path.parent / "metrics.jsonl").resolve()),
        "evaluations": [],
    }
    write_json(path, manifest)
    return manifest


def find_numerics_mismatches(
    logs_root: Path,
    config: TRDNConfig,
) -> list[Dict[str, Any]]:
    current = numerics_settings(config)
    mismatches = []
    for manifest_path in sorted((logs_root / "runs").glob("*/run_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        existing = manifest.get("numerics")
        if not isinstance(existing, dict):
            continue
        differences = {
            key: {"existing": existing.get(key), "current": value}
            for key, value in current.items()
            if existing.get(key) != value
        }
        if differences:
            mismatches.append(
                {
                    "manifest_path": str(manifest_path.resolve()),
                    "differences": differences,
                }
            )
    return mismatches


def find_seed_mismatches(logs_root: Path, config: TRDNConfig) -> list[Dict[str, Any]]:
    mismatches = []
    project_root = logs_root.parent
    candidates = {
        path.resolve()
        for path in (
            list((project_root / "logs" / "runs").glob("*/run_manifest.json"))
            + list(
                project_root.parent.glob(
                    "*/logs/runs/*/run_manifest.json"
                )
            )
        )
    }
    for manifest_path in sorted(candidates):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            mismatches.append(
                {
                    "manifest_path": str(manifest_path),
                    "warning": f"could not inspect sibling seed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        existing_seed = manifest.get("config", {}).get("seed")
        if existing_seed is not None and int(existing_seed) != int(config.seed):
            mismatches.append(
                {
                    "manifest_path": str(manifest_path.resolve()),
                    "existing_seed": int(existing_seed),
                    "current_seed": int(config.seed),
                }
            )
    return mismatches


def update_manifest(path: Path, updates: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest.update(updates)
    write_json(path, manifest)
    return manifest


def append_evaluation_to_manifest(path: Path, evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    evaluations = list(manifest.get("evaluations", []))
    evaluations.append(dict(evaluation))
    manifest["evaluations"] = evaluations
    write_json(path, manifest)
    return manifest


def peak_gpu_memory_bytes() -> int:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


class JsonlMetricLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("recorded_at_unix", time.time())
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
            handle.flush()


def mean_records(records: Iterable[Mapping[str, float]], keys: Iterable[str]) -> Dict[str, float | None]:
    rows = list(records)
    result: Dict[str, float | None] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = sum(values) / len(values) if values else None
    return result
