"""
Rich console handler and broker-only runtime controller singleton.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import ClassVar

from rich.console import Console
from rich.text import Text

from app.console_logging import should_suppress_console_record
from app.rich_formatter import RichFormatter
from app.runtime_console import ConsoleRuntimeController

_shared_console: Console | None = None


def _should_force_terminal() -> bool:
    if sys.stdout.isatty():
        return True
    return str(os.environ.get("TERM_PROGRAM", "")).lower() == "vscode"


def get_console() -> Console:
    global _shared_console
    if _shared_console is None:
        _shared_console = Console(force_terminal=_should_force_terminal())
    return _shared_console


def _should_suppress(record: logging.LogRecord) -> bool:
    return should_suppress_console_record(record)


class RichConsoleHandler(logging.Handler):
    """Logging handler that prints styled messages via the shared Console."""

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


class ProgressManager:
    """Compatibility singleton that now owns the brokered runtime controller."""

    _instance: ClassVar[ProgressManager | None] = None

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or get_console()
        self._controller = ConsoleRuntimeController(console=self.console)

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

    def start(self) -> None:
        self._controller.start()

    def stop(self) -> None:
        self._controller.stop()

    def is_active(self) -> bool:
        return self._controller.is_active()

    @property
    def controller(self) -> ConsoleRuntimeController:
        return self._controller

    # Legacy compatibility no-ops.
    def start_sleep(self, seconds: int) -> None:
        del seconds

    def update_sleep(self, elapsed: float) -> None:
        del elapsed

    def stop_sleep(self) -> None:
        return None

    def start_planner_wait(self, model: str = "", action: str = "Planning next action") -> None:
        del model, action

    def update_planner_wait(self, output_chars: int = 0) -> None:
        del output_chars

    def stop_planner_wait(self) -> None:
        return None

    def add_worker(self, task_id: str, worker_id: str, pid: int | None = None, description: str = "") -> None:
        del task_id, worker_id, pid, description

    def update_worker(self, task_id: str) -> None:
        del task_id

    def remove_worker(self, task_id: str) -> None:
        del task_id
