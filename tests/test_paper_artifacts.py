import csv
import json
from pathlib import Path

from scripts.emit_paper_tables import best_rows, load_reports, render_markdown, write_csv
from scripts.make_paper_figures import (
    generate_metric_figures,
    generate_reference_weight_artifacts,
    load_eval_reports,
    select_sample_indices,
)


def _eval_report(variant: str, psnr: float, lpips: float, runtime: float) -> dict:
    return {
        "schema_version": 2,
        "variant": variant,
        "checkpoint_path": "checkpoint",
        "checkpoint_git_sha": "abc123",
        "git_commit": "def456",
        "is_full_test": True,
        "N_frames": 20,
        "N_clips": 2,
        "seed": 1234,
        "num_inference_steps": 30,
        "seq_len": 10,
        "runtime_seconds": runtime,
        "aggregate": {
            "psnr": {"mean": psnr, "std": 0.1},
            "ssim": {"mean": 0.8, "std": 0.01},
            "lpips": {"mean": lpips, "std": 0.01},
            "temporal_consistency_l1": {"mean": 0.04, "std": 0.01},
        },
        "reference_weights_by_offset": [
            {"offset": -2, "mean": 0.4, "std": 0.1, "count": 100},
            {"offset": -1, "mean": 0.6, "std": 0.1, "count": 100},
        ],
    }


def _write_reports(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    paths[0].write_text(json.dumps(_eval_report("A", 30.0, 0.20, 12.0)), encoding="utf-8")
    paths[1].write_text(json.dumps(_eval_report("B", 29.0, 0.10, 10.0)), encoding="utf-8")
    return paths


def test_table_best_marking_uses_metric_direction(tmp_path: Path):
    paths = _write_reports(tmp_path)
    reports = load_reports(paths)

    best = best_rows(reports)
    markdown = render_markdown(reports)

    assert best["psnr"] == {0}
    assert best["lpips"] == {1}
    assert best["runtime_seconds"] == {1}
    assert "| A | 20 | **30**" in markdown
    assert "| B | 20 | 29" in markdown
    assert "**0.1**" in markdown
    assert "N=20" in markdown
    assert "checkpoint SHA=abc123" in markdown

    csv_path = tmp_path / "table.csv"
    write_csv(csv_path, reports)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows[0]["psnr_is_best"] == "True"
    assert rows[0]["lpips_is_best"] == "False"
    assert rows[1]["lpips_is_best"] == "True"


def test_figure_selection_and_sidecars_are_byte_identical(tmp_path: Path):
    paths = _write_reports(tmp_path)
    reports = load_eval_reports(paths)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    indices = select_sample_indices(20, 4, seed=77)
    assert indices == select_sample_indices(20, 4, seed=77)

    for output_dir in (tmp_path / "run_a", tmp_path / "run_b"):
        generate_metric_figures(reports, paths, checkpoint, output_dir, 77, indices)
        generate_reference_weight_artifacts(reports, paths, checkpoint, output_dir, 77, indices)

    for name in (
        "variant_metrics.sidecar.json",
        "temporal_window_comparison.sidecar.json",
        "reference_weights.sidecar.json",
    ):
        assert (tmp_path / "run_a" / name).read_bytes() == (tmp_path / "run_b" / name).read_bytes()
    assert json.loads((tmp_path / "run_a" / "reference_weights.json").read_text())[
        "variants"
    ][0]["weights"][0]["mean"] == 0.4
