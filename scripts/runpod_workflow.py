"""Operational helpers for the ordered RunPod notebook workflow."""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runpod_jobs import launch as launch_detached

VARIANTS = ("full", "no_raft", "no_transformer", "diffusion_only")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run(command: list[str]) -> None:
    print(f"START {datetime.now(timezone.utc).isoformat()}")
    print("COMMAND " + " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def environment(args: argparse.Namespace) -> None:
    if args.output.is_file() and not args.force:
        print(f"SKIP: environment report exists at {args.output}")
        print(args.output.read_text(encoding="utf-8"))
        return
    if args.install:
        _run([sys.executable, "-m", "pip", "install", "-r", str(args.requirements)])
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("RunPod environment check failed: CUDA is unavailable.")
    try:
        import diffusers
    except ImportError as exc:
        raise RuntimeError("diffusers is not importable after dependency setup.") from exc
    properties = torch.cuda.get_device_properties(0)
    report = {
        "status": "PASS",
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "vram_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "diffusers_version": diffusers.__version__,
        "python": sys.version,
    }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2))


def dataset_check(args: argparse.Namespace) -> None:
    if args.output.is_file() and not args.force:
        print(f"SKIP: dataset report exists at {args.output}")
        print(args.output.read_text(encoding="utf-8"))
        return
    if not args.dataset_root.is_dir() or not any(args.dataset_root.iterdir()):
        raise RuntimeError(f"Dataset root is missing or empty: {args.dataset_root}")
    from src.dataset import REVIDESequenceDataset
    from src.config import TRDNConfig

    config = TRDNConfig(
        seed=args.seed,
        seq_len=args.seq_len,
        crop_size=args.crop_size,
    )
    config.override_dataset_root(str(args.dataset_root))
    reports = {}
    for split in ("train", "val", "test"):
        dataset = REVIDESequenceDataset(
            root=config.root_for_split(split),
            split=split,
            seq_len=config.seq_len,
            crop_size=config.crop_size,
            random_crop=False,
            synthetic_if_empty=False,
            train_mode="dehaze",
            mask_mode="auto",
            split_seed=config.seed,
            include_prev_frame=split != "test",
        )
        dataset.assert_valid_structure(split)
        inventory = dataset.layout_inventory()
        inventory["sample_windows"] = len(dataset)
        inventory["frame_count"] = sum(
            int(sequence["hazy_frames"]) for sequence in inventory["sequences"]
        )
        reports[split] = inventory
    report = {
        "status": "PASS",
        "pairing_rule": "natural_sort_then_exact_or_known_modality_normalized_stem",
        "splits": reports,
    }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2))


def gate(args: argparse.Namespace) -> None:
    vae = _load_json(args.vae_json)
    preflight = _load_json(args.preflight_json)
    benchmark = _load_json(args.benchmark_json)
    if preflight.get("status") != "PASS":
        raise RuntimeError("Preflight did not pass. Training must not start.")
    projections = benchmark.get("projections", {})
    summary = {
        "STOP_BEFORE_TRAINING": True,
        "review_files": [
            str(args.vae_json.resolve()),
            str(args.preflight_json.resolve()),
            str(args.benchmark_json.resolve()),
        ],
        "vae_ceiling": vae.get("aggregate"),
        "preflight_status": preflight.get("status"),
        "benchmark_device": benchmark.get("device"),
        "recommended_preset": benchmark.get("recommended_preset"),
        "projected_seconds_all_runs": projections.get("seconds_all_runs"),
        "projected_checkpoint_bytes_all_runs": projections.get(
            "checkpoint_bytes_all_runs"
        ),
        "projection_basis": projections.get("basis"),
    }
    print(json.dumps(summary, indent=2))


def lock_numerics(args: argparse.Namespace) -> None:
    import yaml

    from src.presets import load_numerics_preset

    if args.confirm != "LOCK_A40":
        raise RuntimeError(
            "Numerics were not locked. Set confirmation to exactly LOCK_A40 after review."
        )
    benchmark = _load_json(args.benchmark_json)
    recommendation = benchmark.get("recommended_preset", {}).get("values")
    if not isinstance(recommendation, dict):
        raise RuntimeError("Benchmark report has no measured recommended_preset.values.")
    values = dict(recommendation)
    if args.overrides_json:
        override_path = Path(args.overrides_json)
        overrides = (
            _load_json(override_path)
            if override_path.is_file()
            else json.loads(args.overrides_json)
        )
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown numerics overrides: {unknown}")
        values.update(overrides)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(values, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    verified = load_numerics_preset(temporary)
    temporary.replace(args.output)
    print(json.dumps({"locked_preset": verified, "path": str(args.output)}, indent=2))


def _sibling_manifests(runs_root: Path) -> list[Path]:
    return sorted(runs_root.glob("*/logs/runs/*/run_manifest.json"))


def _latest_step_checkpoint(project_root: Path) -> Path | None:
    candidates = []
    for path in (project_root / "checkpoints").glob("step_*"):
        suffix = path.name.removeprefix("step_")
        if path.is_dir() and suffix.isdigit():
            candidates.append((int(suffix), path))
    return max(candidates, default=(0, None))[1]


def launch_training(args: argparse.Namespace) -> None:
    from src.config import TRDNConfig
    from src.presets import apply_numerics_preset, load_numerics_preset
    from src.provenance import numerics_settings

    preset = load_numerics_preset(args.preset)
    config = TRDNConfig(
        project_root=str(args.project_root),
        model_variant=args.variant,
        seed=args.seed,
    )
    apply_numerics_preset(config, args.preset)
    expected_numerics = numerics_settings(config)
    for manifest_path in _sibling_manifests(args.runs_root):
        manifest = _load_json(manifest_path)
        sibling_numerics = manifest.get("numerics")
        sibling_seed = manifest.get("config", {}).get("seed")
        if sibling_numerics != expected_numerics:
            raise RuntimeError(
                f"Refusing launch: sibling numerics differ in {manifest_path}."
            )
        if sibling_seed is not None and int(sibling_seed) != args.seed:
            raise RuntimeError(
                f"Refusing launch: sibling seed={sibling_seed}, requested={args.seed} "
                f"in {manifest_path}."
            )
    run_dir = args.project_root / "logs" / "runs" / args.run_name
    status_path = run_dir / ".runpod_job" / "status.json"
    status = _load_json(status_path) if status_path.is_file() else {}
    resume = args.resume_from_checkpoint
    if (
        not resume
        and args.resume_if_interrupted
        and status.get("status") in {"failed", "running"}
    ):
        checkpoint = _latest_step_checkpoint(args.project_root)
        if checkpoint is not None:
            resume = str(checkpoint)
            print(f"Interrupted job: resuming exact numbered checkpoint {resume}")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_colab.py"),
        "--dataset-root",
        str(args.dataset_root),
        "--project-root",
        str(args.project_root),
        "--run-name",
        args.run_name,
        "--model-variant",
        args.variant,
        "--preset",
        str(args.preset),
        "--seed",
        str(args.seed),
        "--num-epochs",
        str(args.num_epochs),
        "--max-train-steps",
        str(args.max_train_steps),
        "--num-workers",
        str(args.num_workers),
        "--seq-len",
        str(args.seq_len),
        "--crop-size",
        str(args.crop_size),
        "--validation-num-samples",
        str(args.validation_num_samples),
        "--validation-num-steps",
        str(args.validation_num_steps),
        "--validation-seed",
        str(args.seed),
        "--checkpoint-selection-metric",
        args.checkpoint_selection_metric,
    ]
    if resume:
        command.extend(["--resume-from-checkpoint", resume])
    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "variant": args.variant,
                "output_directory": str(args.project_root),
                "run_directory": str(run_dir),
                "seed": args.seed,
                "numerics": preset,
                "resume_from_checkpoint": resume or None,
            },
            indent=2,
        )
    )
    launch_detached(
        argparse.Namespace(
            name=args.run_name,
            run_dir=run_dir,
            cwd=REPO_ROOT,
            force=args.force,
            command=command,
        )
    )


def evaluate_all(args: argparse.Namespace) -> None:
    summaries = {}
    for variant in VARIANTS:
        project_root = args.runs_root / variant
        manifests = sorted((project_root / "logs" / "runs").glob("*/run_manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError(
                f"Expected exactly one run manifest for {variant}, found {len(manifests)}."
            )
        manifest = _load_json(manifests[0])
        if manifest.get("status") != "completed":
            raise RuntimeError(f"Run {variant} is not complete: {manifest.get('status')}")
        checkpoint_name = manifest.get("checkpoint_selection", {}).get(
            "checkpoint_name"
        )
        checkpoint = project_root / "checkpoints" / str(checkpoint_name)
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"Selected checkpoint for {variant} is missing: {checkpoint}"
            )
        output = args.eval_dir / f"{variant}.json"
        if output.is_file() and not args.force:
            print(f"SKIP: evaluation exists for {variant}: {output}")
        else:
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_full_test.py"),
                "--checkpoint",
                str(checkpoint),
                "--project-root",
                str(project_root),
                "--dataset-root",
                str(args.dataset_root),
                "--num-steps",
                str(args.num_steps),
                "--seed",
                str(args.seed),
                "--seq-len",
                str(args.seq_len),
                "--crop-size",
                str(args.crop_size),
                "--model-variant",
                variant,
                "--variant",
                variant,
                "--preset",
                str(args.preset),
                "--output",
                str(output),
            ]
            _run(command)
        report = _load_json(output)
        summaries[variant] = {
            "clips_found": report.get("clips_total_found"),
            "clips_evaluated": report.get("N_clips"),
            "clips_skipped": report.get("clips_skipped"),
            "frames_evaluated": report.get("N_frames"),
            "skipped_reasons": report.get("skipped_clips"),
        }
    print("SAMPLE ACCOUNTING")
    print(json.dumps(summaries, indent=2))


def bundle(args: argparse.Namespace) -> None:
    if args.output.is_file() and not args.force:
        print(
            json.dumps(
                {
                    "status": "SKIP",
                    "archive": str(args.output),
                    "bytes": args.output.stat().st_size,
                },
                indent=2,
            )
        )
        return
    required = [args.preset]
    for variant in VARIANTS:
        project_root = args.runs_root / variant
        required.extend(
            [
                project_root / "logs" / "runs" / variant / "run_manifest.json",
                project_root / "logs" / "runs" / variant / "metrics.jsonl",
                args.eval_dir / f"{variant}.json",
            ]
        )
    required.extend(
        [
            args.artifacts_dir / "shared_sample_selection.json",
            args.artifacts_dir / "figure_checklist.json",
            args.artifacts_dir / "paper_results.md",
            args.artifacts_dir / "paper_results.csv",
        ]
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Bundle inputs are missing: " + ", ".join(map(str, missing)))
    figure_files = [
        path
        for path in args.artifacts_dir.iterdir()
        if path.is_file()
        and (
            path.suffix in {".png", ".pdf"}
            or path.name.endswith(".sidecar.json")
            or path.name in {"reference_weights.json", "figure_checklist.json"}
        )
    ]
    bundle_files = list(
        {
            path.resolve(): path
            for path in [*required, *figure_files]
        }.values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in bundle_files:
            resolved = path.resolve()
            try:
                name = resolved.relative_to(args.workspace_root.resolve())
            except ValueError:
                name = Path("external") / resolved.name
            archive.write(resolved, name.as_posix())
    print(
        json.dumps(
            {
                "status": "PASS",
                "archive": str(args.output.resolve()),
                "bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    environment_parser = subparsers.add_parser("environment")
    environment_parser.add_argument("--output", type=Path, required=True)
    environment_parser.add_argument(
        "--requirements", type=Path, default=REPO_ROOT / "requirements.txt"
    )
    environment_parser.add_argument("--install", action="store_true")
    environment_parser.add_argument("--force", action="store_true")
    environment_parser.set_defaults(func=environment)

    dataset_parser = subparsers.add_parser("dataset-check")
    dataset_parser.add_argument("--dataset-root", type=Path, required=True)
    dataset_parser.add_argument("--output", type=Path, required=True)
    dataset_parser.add_argument("--seed", type=int, default=1234)
    dataset_parser.add_argument("--seq-len", type=int, default=10)
    dataset_parser.add_argument("--crop-size", type=int, default=256)
    dataset_parser.add_argument("--force", action="store_true")
    dataset_parser.set_defaults(func=dataset_check)

    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument("--vae-json", type=Path, required=True)
    gate_parser.add_argument("--preflight-json", type=Path, required=True)
    gate_parser.add_argument("--benchmark-json", type=Path, required=True)
    gate_parser.set_defaults(func=gate)

    lock_parser = subparsers.add_parser("lock-numerics")
    lock_parser.add_argument("--benchmark-json", type=Path, required=True)
    lock_parser.add_argument("--output", type=Path, required=True)
    lock_parser.add_argument("--confirm", required=True)
    lock_parser.add_argument("--overrides-json", default="")
    lock_parser.set_defaults(func=lock_numerics)

    launch_parser = subparsers.add_parser("launch-training")
    launch_parser.add_argument("--variant", choices=VARIANTS, required=True)
    launch_parser.add_argument("--run-name", required=True)
    launch_parser.add_argument("--project-root", type=Path, required=True)
    launch_parser.add_argument("--runs-root", type=Path, required=True)
    launch_parser.add_argument("--dataset-root", type=Path, required=True)
    launch_parser.add_argument("--preset", type=Path, required=True)
    launch_parser.add_argument("--seed", type=int, default=1234)
    launch_parser.add_argument("--num-epochs", type=int, default=30)
    launch_parser.add_argument("--max-train-steps", type=int, default=0)
    launch_parser.add_argument("--num-workers", type=int, default=4)
    launch_parser.add_argument("--seq-len", type=int, default=10)
    launch_parser.add_argument("--crop-size", type=int, default=256)
    launch_parser.add_argument("--validation-num-samples", type=int, default=32)
    launch_parser.add_argument("--validation-num-steps", type=int, default=30)
    launch_parser.add_argument(
        "--checkpoint-selection-metric", choices=["psnr", "ssim"], default="psnr"
    )
    launch_parser.add_argument("--resume-from-checkpoint", default="")
    launch_parser.add_argument("--resume-if-interrupted", action="store_true")
    launch_parser.add_argument("--force", action="store_true")
    launch_parser.set_defaults(func=launch_training)

    evaluate_parser = subparsers.add_parser("evaluate-all")
    evaluate_parser.add_argument("--runs-root", type=Path, required=True)
    evaluate_parser.add_argument("--dataset-root", type=Path, required=True)
    evaluate_parser.add_argument("--eval-dir", type=Path, required=True)
    evaluate_parser.add_argument("--preset", type=Path, required=True)
    evaluate_parser.add_argument("--num-steps", type=int, required=True)
    evaluate_parser.add_argument("--seed", type=int, default=1234)
    evaluate_parser.add_argument("--seq-len", type=int, default=10)
    evaluate_parser.add_argument("--crop-size", type=int, default=256)
    evaluate_parser.add_argument("--force", action="store_true")
    evaluate_parser.set_defaults(func=evaluate_all)

    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("--workspace-root", type=Path, required=True)
    bundle_parser.add_argument("--runs-root", type=Path, required=True)
    bundle_parser.add_argument("--eval-dir", type=Path, required=True)
    bundle_parser.add_argument("--artifacts-dir", type=Path, required=True)
    bundle_parser.add_argument("--preset", type=Path, required=True)
    bundle_parser.add_argument("--output", type=Path, required=True)
    bundle_parser.add_argument("--force", action="store_true")
    bundle_parser.set_defaults(func=bundle)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
