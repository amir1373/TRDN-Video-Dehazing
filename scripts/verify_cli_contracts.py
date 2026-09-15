"""Verify README/notebook CLI flags against each target script's argparse help."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBCOMMANDS = {
    "runpod_workflow.py": (
        "environment",
        "dataset-check",
        "gate",
        "lock-numerics",
        "launch-training",
        "evaluate-all",
        "bundle",
    ),
    "runpod_jobs.py": ("launch", "monitor", "list", "stop"),
}


def _help_flags(script: Path) -> set[str]:
    commands = [()] + [
        (subcommand,) for subcommand in SUBCOMMANDS.get(script.name, ())
    ]
    flags = set()
    for prefix in commands:
        result = subprocess.run(
            [sys.executable, str(script), *prefix, "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{script} {' '.join(prefix)} --help failed "
                f"({result.returncode}): {result.stderr.strip()}"
            )
        flags.update(re.findall(r"--[a-z0-9][a-z0-9-]*", result.stdout))
    return flags


def _calls_from_notebook(path: Path) -> list[dict]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    calls = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        scripts = sorted(set(re.findall(r'"([a-z0-9_]+\.py)"', source)))
        flags = (
            sorted(set(re.findall(r'"(--[a-z0-9-]+)"', source)))
            if len(scripts) == 1
            else []
        )
        for script in scripts:
            calls.append(
                {
                    "call_site": f"{path.name} cell {index}",
                    "script": f"scripts/{script}",
                    "flags": flags,
                }
            )
    return calls


def _notebook_calls() -> list[dict]:
    return [
        call
        for path in (
            REPO_ROOT / "notebooks" / "TRDN_REVIDE_Colab.ipynb",
            REPO_ROOT / "notebooks" / "TRDN_REVIDE_RunPod.ipynb",
        )
        for call in _calls_from_notebook(path)
    ]


def _readme_commands() -> Iterable[tuple[int, list[str]]]:
    lines = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    in_bash = False
    current = ""
    start_line = 0
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped == "```bash":
            in_bash = True
            continue
        if in_bash and stripped == "```":
            if current:
                yield start_line, current.split()
                current = ""
            in_bash = False
            continue
        if not in_bash:
            continue
        if stripped.startswith("python "):
            if current:
                yield start_line, current.split()
            current = stripped.removesuffix("\\").strip()
            start_line = line_number
        elif current and stripped:
            current += " " + stripped.removesuffix("\\").strip()
        elif current:
            yield start_line, current.split()
            current = ""


def _readme_calls() -> list[dict]:
    calls = []
    for line, tokens in _readme_commands():
        if len(tokens) < 2 or not tokens[1].endswith(".py"):
            continue
        script = tokens[1].replace("\\", "/")
        flags = sorted(token for token in tokens[2:] if token.startswith("--"))
        calls.append(
            {
                "call_site": f"README.md:{line}",
                "script": script,
                "flags": flags,
            }
        )
    return calls


def verify() -> list[dict]:
    rows = []
    cache: dict[str, set[str]] = {}
    for call in [*_notebook_calls(), *_readme_calls()]:
        script_path = REPO_ROOT / call["script"]
        if not script_path.is_file():
            rows.append({**call, "status": "MISMATCH", "missing": ["script_not_found"]})
            continue
        if call["script"] not in cache:
            cache[call["script"]] = _help_flags(script_path)
        available = cache[call["script"]]
        missing = sorted(set(call["flags"]) - available)
        rows.append(
            {
                **call,
                "status": "OK" if not missing else "MISMATCH",
                "missing": missing,
            }
        )
    return rows


def main() -> None:
    rows = verify()
    print("| Call site | Script | Flags | Status |")
    print("| --- | --- | --- | --- |")
    for row in rows:
        flags = ", ".join(row["flags"]) or "(none)"
        detail = row["status"]
        if row["missing"]:
            detail += ": " + ", ".join(row["missing"])
        print(f"| {row['call_site']} | {row['script']} | {flags} | {detail} |")
    if any(row["status"] != "OK" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
