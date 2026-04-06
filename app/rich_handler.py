"""
Rich console handler and progress manager for the orchestrator.

Uses a single shared Console so that Rich's Live display and logging
automatically coordinate: log messages print above the progress area,
and the progress bar stays at the bottom.
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from app.rich_formatter import _extract_event_tag, RichFormatter
from app.models import OrchestratorEvent

# ---------------------------------------------------------------------------
# Shared Console — one instance for the entire process
# ---------------------------------------------------------------------------

_shared_console: Console | None = None


def get_console() -> Console:
    global _shared_console
    if _shared_console is None:
        _shared_console = Console()
    return _shared_console


# ---------------------------------------------------------------------------
# Console suppression — messages hidden from console (still in file log)
# ---------------------------------------------------------------------------

_SUPPRESSED_EVENTS: set[OrchestratorEvent] = {
    OrchestratorEvent.STATE_SAVED,
    OrchestratorEvent.SLEEPING,
}

_SUPPRESSED_LOGGER_PREFIXES: tuple[str, ...] = (
    "orchestrator.scheduler",
    "orchestrator.state",
)


def _should_suppress(record: logging.LogRecord) -> bool:
    event, _ = _extract_event_tag(record.getMessage())
    if event in _SUPPRESSED_EVENTS:
        return True
    for prefix in _SUPPRESSED_LOGGER_PREFIXES:
        if record.name == prefix or record.name.startswith(prefix + "."):
            return True
    return False


def _short(text: str, max_len: int = 60) -> str:
    text = text.replace("\n", " ").strip()
    return text[: max_len - 3] + "..." if len(text) > max_len else text


# ---------------------------------------------------------------------------
# RichConsoleHandler
# ---------------------------------------------------------------------------


class RichConsoleHandler(logging.Handler):
    """Logging handler that prints styled messages via the shared Console.

    No manual Live coordination needed — when ProgressManager's Live is
    active on the same Console, Rich automatically pauses it, prints the
    log line, then resumes the progress display.
    """

    def __init__(
        self,
        console: Console | None = None,
        truncate_length: int = 300,
    ) -> None:
        super().__init__()
        self.console = console or get_console()
        self._formatter = RichFormatter(truncate_length=truncate_length)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.acquire()
            if _should_suppress(record):
                return
            self._print_record(record)
        finally:
            self.release()

    def _print_record(self, record: logging.LogRecord) -> None:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        self.console.print(
            Text(ts, style="dim"),
            self._formatter.format(record),
            sep="  ",
        )


# ---------------------------------------------------------------------------
# ProgressManager
# ---------------------------------------------------------------------------


class ProgressManager:
    """Singleton that owns the Rich Live + Progress display.

    Uses the *same* shared Console as RichConsoleHandler so Rich can
    coordinate log output and progress bars automatically.

    Tracks:
    - Sleep countdown  (blue,   determinate bar)
    - Planner wait     (yellow, indeterminate spinner)
    - Active workers   (green,  indeterminate spinners, one per worker)
    """

    _instance: ClassVar[ProgressManager | None] = None

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or get_console()
        self._live: Live | None = None
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )
        self._sleep_task: TaskID | None = None
        self._planner_task: TaskID | None = None
        self._worker_tasks: dict[str, TaskID] = {}
        self._planner_start: float = 0.0
        self._worker_starts: dict[str, float] = {}

    @classmethod
    def get(cls, console: Console | None = None) -> ProgressManager:
        if cls._instance is None:
            cls._instance = cls(console)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        if cls._instance is not None:
            cls._instance.stop()
            cls._instance = None

    # -- lifecycle --

    def start(self) -> None:
        if self._live is not None:
            return
        self._live = Live(
            self._progress,
            console=self.console,
            refresh_per_second=2,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._clear_all_tasks()

    def is_active(self) -> bool:
        return self._live is not None

    # -- sleep countdown (blue) --

    def start_sleep(self, seconds: int) -> None:
        self._sleep_task = self._progress.add_task(
            "[dim blue]Sleeping...[/dim blue]",
            total=seconds,
        )

    def update_sleep(self, elapsed: float) -> None:
        if self._sleep_task is not None:
            self._progress.update(self._sleep_task, completed=elapsed)

    def stop_sleep(self) -> None:
        if self._sleep_task is not None:
            self._progress.remove_task(self._sleep_task)
            self._sleep_task = None

    # -- planner wait (yellow) --

    def start_planner_wait(
        self,
        model: str = "",
        action: str = "Planning next action",
    ) -> None:
        label = "[bold yellow]Claude Code"
        if model:
            label += f" ({model})"
        label += f" — {action}"
        self._planner_start = time.monotonic()
        self._planner_task = self._progress.add_task(label, total=None)

    def update_planner_wait(self, output_chars: int = 0) -> None:
        pass  # TimeElapsedColumn handles elapsed automatically

    def stop_planner_wait(self) -> None:
        if self._planner_task is not None:
            self._progress.remove_task(self._planner_task)
            self._planner_task = None

    # -- worker tracking (green) --

    def add_worker(
        self,
        task_id: str,
        worker_id: str,
        pid: int | None = None,
        description: str = "",
    ) -> None:
        label = f"[bold green]{worker_id}"
        if pid:
            label += f" (pid={pid})"
        if description:
            label += f" — {_short(description)}"
        self._worker_starts[task_id] = time.monotonic()
        tid = self._progress.add_task(label, total=None)
        self._worker_tasks[task_id] = tid

    def update_worker(self, task_id: str) -> None:
        if task_id in self._worker_tasks:
            self._progress.update(self._worker_tasks[task_id], advance=0)

    def remove_worker(self, task_id: str) -> None:
        if task_id in self._worker_tasks:
            self._progress.remove_task(self._worker_tasks.pop(task_id))
            self._worker_starts.pop(task_id, None)

    # -- internals --

    def _clear_all_tasks(self) -> None:
        for tid in [self._sleep_task, self._planner_task]:
            if tid is not None:
                try:
                    self._progress.remove_task(tid)
                except Exception:
                    pass
        self._sleep_task = None
        self._planner_task = None
        for tid in list(self._worker_tasks.values()):
            try:
                self._progress.remove_task(tid)
            except Exception:
                pass
        self._worker_tasks.clear()
        self._worker_starts.clear()
