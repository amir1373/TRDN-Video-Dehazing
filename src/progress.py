import os
import sys
import time
from typing import Any, TextIO

from tqdm.auto import tqdm


def _interactive(file: TextIO) -> bool:
    forced = os.environ.get("TRDN_FORCE_TQDM", "").strip().lower()
    if forced in {"1", "true", "yes"}:
        return True
    if forced in {"0", "false", "no"}:
        return False
    return bool(getattr(file, "isatty", lambda: False)())


class ProgressReporter:
    """tqdm in terminals, periodic ETA lines in redirected logs."""

    def __init__(
        self,
        total: int,
        desc: str,
        *,
        initial: int = 0,
        leave: bool = True,
        position: int = 0,
        file: TextIO | None = None,
        enabled: bool = True,
    ):
        if total < 0:
            raise ValueError(f"Progress total must be non-negative, got {total}")
        self.total = total
        self.desc = desc
        self.current = initial
        self.file = file or sys.stderr
        self.started = time.perf_counter()
        self.postfix: dict[str, Any] = {}
        self.enabled = enabled
        self.interactive = _interactive(self.file)
        self._line_interval = max(total // 10, 1) if total else 1
        self._last_line = initial
        self._bar = (
            tqdm(
                total=total,
                initial=initial,
                desc=desc,
                leave=leave,
                position=position,
                dynamic_ncols=True,
                file=self.file,
            )
            if self.interactive and self.enabled
            else None
        )
        if self.enabled and not self.interactive:
            self._print_line("start")

    def _eta_seconds(self) -> float | None:
        completed = self.current
        if completed <= 0 or self.total <= completed:
            return 0.0 if self.total <= completed else None
        elapsed = time.perf_counter() - self.started
        return elapsed / completed * (self.total - completed)

    def _print_line(self, state: str = "progress") -> None:
        eta = self._eta_seconds()
        eta_text = "unknown" if eta is None else f"{eta:.1f}s"
        percent = 100.0 if self.total == 0 else 100.0 * self.current / max(self.total, 1)
        postfix = " ".join(f"{key}={value}" for key, value in self.postfix.items())
        suffix = f" {postfix}" if postfix else ""
        print(
            f"{self.desc}: {state} {self.current}/{self.total} "
            f"({percent:.1f}%) ETA {eta_text}{suffix}",
            file=self.file,
            flush=True,
        )

    def update(self, amount: int = 1) -> None:
        self.current = min(self.current + amount, self.total)
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.update(amount)
            return
        if self.current == self.total or self.current - self._last_line >= self._line_interval:
            self._last_line = self.current
            self._print_line("progress")

    def set_postfix(self, values: dict[str, Any]) -> None:
        self.postfix = values
        if self._bar is not None:
            self._bar.set_postfix(values, refresh=False)

    def set_description(self, description: str) -> None:
        self.desc = description
        if self._bar is not None:
            self._bar.set_description(description, refresh=False)

    def write(self, message: str) -> None:
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.write(message, file=self.file)
        else:
            print(message, file=self.file, flush=True)

    def close(self) -> None:
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.close()
        elif self.current < self.total:
            self._print_line("stopped")
        else:
            self._print_line("complete")

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
