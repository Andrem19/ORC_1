"""
Thin worker adapter wrapper for the broker-only runtime.
"""

from __future__ import annotations

from app.adapters.base import BaseAdapter


class WorkerService:
    """Container for the worker adapter used by the brokered runtime."""

    def __init__(self, adapter: BaseAdapter, timeout: int = 300) -> None:
        self.adapter = adapter
        self.timeout = timeout

