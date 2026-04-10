"""
Direct execution runtime.
"""

from __future__ import annotations

__all__ = ["DirectExecutionService"]


def __getattr__(name: str):
    if name == "DirectExecutionService":
        from app.services.direct_execution.service import DirectExecutionService

        return DirectExecutionService
    raise AttributeError(name)
