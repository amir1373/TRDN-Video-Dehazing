"""Launch, monitor, list, and stop detached RunPod jobs safely."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _paths(run_dir: Path) -> tuple[Path, Path, Path]:
    state_dir = run_dir / ".runpod_job"
    return state_dir / "status.json", state_dir / "job.log", state_dir


def launch(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    status_path, log_path, state_dir = _paths(run_dir)
    state = _read_json(status_path)
    if state.get("status") == "running" and _alive(int(state.get("pid", 0))):
        print(f"SKIP: {args.name} is already running as PID {state['pid']}.")
        return
    if state.get("status") == "completed" and not args.force:
        print(f"SKIP: {args.name} already completed. Use --force to relaunch.")
        return
    if not args.command:
        raise ValueError("A command is required after --.")
    state_dir.mkdir(parents=True, exist_ok=True)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    initial_state = {
        "name": args.name,
        "status": "launching",
        "command": command,
        "cwd": str(args.cwd.resolve()),
        "log_path": str(log_path),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(status_path, initial_state)
    runner = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_run",
        "--status",
        str(status_path),
        "--name",
        args.name,
        "--",
        *command,
    ]
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    kwargs = {
        "cwd": str(args.cwd.resolve()),
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(runner, **kwargs)
    log_handle.close()
    current_state = _read_json(status_path)
    for key, value in initial_state.items():
        current_state.setdefault(key, value)
    if current_state.get("status") == "launching":
        current_state["status"] = "running"
    current_state["pid"] = process.pid
    _write_json(status_path, current_state)
    print(f"LAUNCHED {args.name}: PID={process.pid}")
    print(f"run_dir={run_dir}")
    print(f"log={log_path}")


def _runner(args: argparse.Namespace) -> None:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    status = _read_json(args.status)
    status.setdefault("name", args.name)
    status.setdefault("command", command)
    status.setdefault("started_at_utc", datetime.now(timezone.utc).isoformat())
    status["pid"] = os.getpid()
    status["status"] = "running"
    child = None
    stop_requested = False

    def terminate(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, terminate)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, terminate)
    try:
        child = subprocess.Popen(command)
        status["child_pid"] = child.pid
        _write_json(args.status, status)
        exit_code = child.wait()
        status.update(
            {
                "status": (
                    "stopped"
                    if stop_requested
                    else "completed"
                    if exit_code == 0
                    else "failed"
                ),
                "exit_code": exit_code,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(args.status, status)
        raise SystemExit(exit_code)
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            status.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_json(args.status, status)
        raise


def _summary(state: dict, lines: list[str]) -> str:
    started = state.get("started_at_utc")
    elapsed = "unknown"
    if started:
        start_time = datetime.fromisoformat(started)
        elapsed = f"{(datetime.now(timezone.utc) - start_time).total_seconds():.0f}s"
    progress = next(
        (
            line.strip()
            for line in reversed(lines)
            if re.search(r"(progress|step=|loss=|ETA)", line, re.IGNORECASE)
        ),
        "no progress line yet",
    )
    return (
        f"{state.get('name', 'job')}: status={state.get('status', 'unknown')} "
        f"pid={state.get('pid')} elapsed={elapsed}\n{progress}"
    )


def monitor(args: argparse.Namespace) -> None:
    status_path, log_path, _state_dir = _paths(args.run_dir.resolve())
    last_size = 0
    while True:
        state = _read_json(status_path)
        if not state:
            raise FileNotFoundError(f"No job state at {status_path}")
        lines = (
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if log_path.is_file()
            else []
        )
        print(_summary(state, lines))
        for line in lines[-args.tail_lines :]:
            print(line)
        if not args.follow or state.get("status") != "running":
            return
        time.sleep(args.poll_seconds)
        if log_path.is_file() and log_path.stat().st_size == last_size:
            print("Waiting for new log output...")
        last_size = log_path.stat().st_size if log_path.is_file() else 0


def list_jobs(args: argparse.Namespace) -> None:
    found = False
    for status_path in sorted(args.runs_root.rglob(".runpod_job/status.json")):
        found = True
        state = _read_json(status_path)
        print(_summary(state, []))
        print(f"  {status_path.parent.parent}")
    if not found:
        print(f"No jobs found under {args.runs_root}.")


def stop(args: argparse.Namespace) -> None:
    status_path, _log_path, _state_dir = _paths(args.run_dir.resolve())
    state = _read_json(status_path)
    pid = int(state.get("pid", 0))
    if state.get("status") != "running" or not _alive(pid):
        print(f"SKIP: job is not running ({state.get('status', 'missing')}).")
        return
    if os.name == "nt":
        os.kill(pid, signal.SIGTERM)
    else:
        os.killpg(pid, signal.SIGTERM)
    state["stop_requested_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(status_path, state)
    print(f"Stop requested for PID {pid}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--name", required=True)
    launch_parser.add_argument("--run-dir", type=Path, required=True)
    launch_parser.add_argument("--cwd", type=Path, default=Path.cwd())
    launch_parser.add_argument("--force", action="store_true")
    launch_parser.add_argument("command", nargs=argparse.REMAINDER)
    launch_parser.set_defaults(func=launch)

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--run-dir", type=Path, required=True)
    monitor_parser.add_argument("--follow", action="store_true")
    monitor_parser.add_argument("--tail-lines", type=int, default=20)
    monitor_parser.add_argument("--poll-seconds", type=float, default=10.0)
    monitor_parser.set_defaults(func=monitor)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--runs-root", type=Path, required=True)
    list_parser.set_defaults(func=list_jobs)

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--run-dir", type=Path, required=True)
    stop_parser.set_defaults(func=stop)

    runner_parser = subparsers.add_parser("_run")
    runner_parser.add_argument("--status", type=Path, required=True)
    runner_parser.add_argument("--name", required=True)
    runner_parser.add_argument("command", nargs=argparse.REMAINDER)
    runner_parser.set_defaults(func=_runner)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
