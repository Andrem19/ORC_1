"""
Base adapter interface for CLI agents.

All adapters implement this protocol so the orchestrator core
is decoupled from specific CLI implementations.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import time as _time


@dataclass
class AdapterResponse:
    """Normalized response from any adapter."""
    success: bool
    raw_output: str
    exit_code: int = 0
    error: str = ""
    timed_out: bool = False
    duration_seconds: float = 0.0


@dataclass
class ProcessHandle:
    """Live handle to a running worker subprocess. NOT serializable."""
    process: subprocess.Popen | None  # None for fake/testing adapters
    task_id: str
    worker_id: str
    started_at: float = field(default_factory=_time.monotonic)
    partial_output: str = ""


class BaseAdapter(ABC):
    """Abstract base for CLI adapters."""

    @abstractmethod
    def invoke(self, prompt: str, timeout: int = 120, **kwargs: Any) -> AdapterResponse:
        """Send a prompt to the CLI and return a normalized response (blocking)."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the underlying CLI is available."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Return adapter name for logging."""
        ...

    def start(self, prompt: str, **kwargs: Any) -> ProcessHandle:
        """Launch a background process. Returns immediately."""
        raise NotImplementedError(f"{self.name()} does not support background execution")

    def check(self, handle: ProcessHandle) -> tuple[str, bool]:
        """Non-blocking check. Returns (new_output, is_finished)."""
        raise NotImplementedError(f"{self.name()} does not support background execution")

    def terminate(self, handle: ProcessHandle) -> None:
        """Terminate the background process gracefully."""
        if handle.process is None:
            return
        handle.process.terminate()
        try:
            handle.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.process.kill()
