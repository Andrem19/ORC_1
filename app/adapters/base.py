"""
Base adapter interface for CLI agents.

All adapters implement this protocol so the orchestrator core
is decoupled from specific CLI implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AdapterResponse:
    """Normalized response from any adapter."""
    success: bool
    raw_output: str
    exit_code: int = 0
    error: str = ""
    timed_out: bool = False
    duration_seconds: float = 0.0


class BaseAdapter(ABC):
    """Abstract base for CLI adapters."""

    @abstractmethod
    def invoke(self, prompt: str, timeout: int = 120, **kwargs: Any) -> AdapterResponse:
        """Send a prompt to the CLI and return a normalized response.

        Args:
            prompt: The text prompt to send.
            timeout: Maximum seconds to wait.

        Returns:
            AdapterResponse with normalized fields.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the underlying CLI is available."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Return adapter name for logging."""
        ...
