"""Emit paper-ready Markdown and CSV tables directly from evaluation JSON."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


METRICS = (
    ("psnr", "PSNR", "higher"),
    ("ssim", "SSIM", "higher"),
    ("lpips", "LPIPS", "lower"),
    ("temporal_consistency_l1", "Temporal error", "lower"),
    ("runtime_seconds", "Runtime (s)", "lower"),
)


def _metric_value(report: Dict[str, Any], metric: str) -> float | None:
    if metric == "runtime_seconds":
        value = report.get("runtime_seconds")
    else:
        value = report.get("aggregate", {}).get(metric, {}).get("mean")
    return float(value) if value is not None else None


def load_reports(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    reports = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("is_full_test") is False:
            raise ValueError(f"Refusing to emit a paper table from non-full evaluation: {path}")
        if report.get("clips_available") is not None and (
            report.get("clips_evaluated") != report.get("clips_available")
        ):
            raise ValueError(f"Refusing incomplete-coverage paper table: {path}")
        report["_source_path"] = str(path.resolve())
        reports.append(report)
    if not reports:
        raise ValueError("At least one evaluation JSON is required.")
    return reports


def best_rows(reports: List[Dict[str, Any]]) -> Dict[str, set[int]]:
    result: Dict[str, set[int]] = {}
    for metric, _label, direction in METRICS:
        values = [(index, _metric_value(report, metric)) for index, report in enumerate(reports)]
        available = [(index, value) for index, value in values if value is not None]
        if not available:
            result[metric] = set()
            continue
        best_value = (
            max(value for _index, value in available)
            if direction == "higher"
            else min(value for _index, value in available)
        )
        result[metric] = {index for index, value in available if value == best_value}
    return result


def _display(value: float | None) -> str:
    return "NA" if value is None else format(value, ".10g")


def caption_line(reports: List[Dict[str, Any]]) -> str:
    def values(key: str, fallback: str = "unknown") -> str:
        unique = []
        for report in reports:
            value = report.get(key, fallback)
            if value not in unique:
                unique.append(value)
        return ", ".join(str(value) for value in unique)

    return (
        f"Results over N={values('N_frames')} frames; seed={values('seed')}; "
        f"DDIM steps={values('num_inference_steps')}; "
        f"checkpoint SHA={values('checkpoint_git_sha')}."
    )


def render_markdown(reports: List[Dict[str, Any]]) -> str:
    best = best_rows(reports)
    headers = ["Variant", "N"] + [label for _metric, label, _direction in METRICS]
    lines = [caption_line(reports), "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")
    for index, report in enumerate(reports):
        row = [str(report.get("variant", f"variant_{index + 1}")), str(report.get("N_frames", "unknown"))]
        for metric, _label, _direction in METRICS:
            rendered = _display(_metric_value(report, metric))
            if index in best[metric] and rendered != "NA":
                rendered = f"**{rendered}**"
            row.append(rendered)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, reports: List[Dict[str, Any]]) -> None:
    best = best_rows(reports)
    fieldnames = ["variant", "N_frames"]
    for metric, _label, _direction in METRICS:
        fieldnames.extend([metric, f"{metric}_is_best"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, report in enumerate(reports):
            row: Dict[str, Any] = {
                "variant": report.get("variant", f"variant_{index + 1}"),
                "N_frames": report.get("N_frames"),
            }
            for metric, _label, _direction in METRICS:
                row[metric] = _metric_value(report, metric)
                row[f"{metric}_is_best"] = index in best[metric]
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", default="paper_results")
    return parser


def run(args: argparse.Namespace) -> List[Path]:
    reports = load_reports(args.eval_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(reports)
    markdown_path = args.output_dir / f"{args.basename}.md"
    csv_path = args.output_dir / f"{args.basename}.csv"
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    write_csv(csv_path, reports)
    print(markdown, end="")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {csv_path}")
    return [markdown_path, csv_path]


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
