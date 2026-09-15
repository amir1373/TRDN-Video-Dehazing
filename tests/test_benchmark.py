import sys
from pathlib import Path

import pytest

from scripts import benchmark
from src.config import TRDNConfig


def test_benchmark_refuses_cpu_without_writing_report(tmp_path: Path, monkeypatch, capsys):
    output = tmp_path / "benchmark_report.json"
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        benchmark.main()

    assert exc_info.value.code == 2
    assert "CUDA is unavailable" in capsys.readouterr().err
    assert not output.exists()


def test_benchmark_matrix_covers_required_a40_sweeps():
    rows = list(benchmark._variant_matrix(TRDNConfig(), 8, [0, 2, 4]))
    names = {name for name, _category, _overrides in rows}
    categories = {category for _name, category, _overrides in rows}

    assert {
        "baseline",
        "gradient_checkpointing_off",
        "batch_size_2",
        "batch_size_4",
        "batch_size_8",
        "precision_bf16",
        "tf32_on",
        "cudnn_benchmark_on",
        "channels_last_on",
        "attention_sdpa",
        "torch_compile_on",
        "num_workers_0",
        "num_workers_4",
    } <= names
    assert "batch_checkpointing_combined" in categories
