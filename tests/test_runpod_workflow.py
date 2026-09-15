import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.benchmark import _recommended_preset
from scripts.runpod_workflow import lock_numerics
from src.presets import load_numerics_preset
from src.provenance import make_run_dir


REPO_ROOT = Path(__file__).parents[1]


def _row(name, category, samples_per_second, **overrides):
    defaults = {
        "mixed_precision": "fp16",
        "allow_tf32": False,
        "cudnn_benchmark": False,
        "attention_backend": "xformers",
        "batch_size": 1,
        "gradient_checkpointing": True,
        "torch_compile": False,
        "channels_last": False,
        "num_workers": 2,
    }
    defaults.update(overrides)
    return {
        "name": name,
        "category": category,
        "status": "measured",
        "samples_per_second": samples_per_second,
        "overrides": defaults,
    }


def test_benchmark_recommendation_uses_measured_source_rows():
    rows = [
        _row("baseline", "baseline", 1.0),
        _row(
            "combined",
            "batch_checkpointing_combined",
            3.0,
            batch_size=4,
            gradient_checkpointing=False,
        ),
        _row("bf16", "precision", 2.0, mixed_precision="bf16"),
        _row("tf32", "tf32", 1.5, allow_tf32=True),
        _row("cudnn", "cudnn_benchmark", 1.4, cudnn_benchmark=True),
        _row("sdpa", "attention_backend", 1.3, attention_backend="sdpa"),
        _row("compile", "torch_compile", 1.2, torch_compile=True),
        _row("channels", "channels_last", 1.1, channels_last=True),
    ]
    recommendation = _recommended_preset(rows)
    assert recommendation["values"]["precision"] == "bf16"
    assert recommendation["values"]["batch_size"] == 4
    assert recommendation["values"]["gradient_checkpointing"] is False
    assert set(recommendation["source_rows"].values()) <= {
        row["name"] for row in rows
    }


def test_lock_numerics_writes_validated_yaml(tmp_path):
    report = {
        "recommended_preset": {
            "values": {
                "preset_name": "a40_locked",
                "precision": "bf16",
                "allow_tf32": True,
                "cudnn_benchmark": False,
                "attention_backend": "sdpa",
                "batch_size": 2,
                "gradient_checkpointing": True,
                "torch_compile": False,
                "channels_last": False,
            }
        }
    }
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "a40.yaml"
    lock_numerics(
        argparse.Namespace(
            benchmark_json=benchmark_path,
            output=output,
            confirm="LOCK_A40",
            overrides_json=json.dumps({"batch_size": 4}),
        )
    )
    assert load_numerics_preset(output)["batch_size"] == 4
    assert "TODO" not in output.read_text(encoding="utf-8")


def test_detached_monitor_can_disconnect_and_reconnect(tmp_path):
    run_dir = tmp_path / "run"
    marker = tmp_path / "completed.txt"
    child_code = (
        "import pathlib,time;"
        "print('step=1 loss=1.0 ETA 1s',flush=True);"
        "time.sleep(1.2);"
        f"pathlib.Path({str(marker)!r}).write_text('done');"
        "print('step=2 loss=0.5 ETA 0s',flush=True)"
    )
    launch = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "runpod_jobs.py"),
        "launch",
        "--name",
        "detached_test",
        "--run-dir",
        str(run_dir),
        "--cwd",
        str(REPO_ROOT),
        "--",
        sys.executable,
        "-c",
        child_code,
    ]
    subprocess.run(launch, cwd=REPO_ROOT, check=True)
    monitor = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "runpod_jobs.py"),
        "monitor",
        "--run-dir",
        str(run_dir),
    ]
    first = subprocess.run(
        monitor, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    assert "status=running" in first.stdout
    assert not marker.exists()
    deadline = time.time() + 10
    while time.time() < deadline:
        state = json.loads(
            (run_dir / ".runpod_job" / "status.json").read_text(encoding="utf-8")
        )
        if state.get("status") == "completed":
            break
        time.sleep(0.1)
    second = subprocess.run(
        monitor, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    assert "status=completed" in second.stdout
    assert "step=2 loss=0.5 ETA 0s" in second.stdout
    assert marker.read_text(encoding="utf-8") == "done"


def test_training_run_dir_can_be_precreated_by_detached_launcher(tmp_path):
    run_dir = tmp_path / "logs" / "runs" / "full"
    state_dir = run_dir / ".runpod_job"
    state_dir.mkdir(parents=True)
    (state_dir / "status.json").write_text("{}", encoding="utf-8")
    assert make_run_dir(tmp_path / "logs", "full") == run_dir
