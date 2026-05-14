"""
Track failed async-operation handles to prevent cascading tool errors.

When an MCP tool reports that an async operation (walk-forward, study,
analysis, etc.) has failed, the model sometimes tries to query or analyse
the failed resource again, producing repeated contract/misuse errors that
trigger the error-loop detector and force a salvage.

This module extracts the failed handle IDs from tool error responses and
provides a preflight check that blocks subsequent calls referencing those
IDs, returning actionable guidance instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_HANDLE_FIELDS = ("job_id", "run_id", "operation_id", "snapshot_id")

# Error tokens that signal a runtime operation failure (not contract misuse).
_RUNTIME_FAILURE_TOKENS = (
    "operation_failed",
    "ended with status 'failed'",
    "ended with status 'cancelled'",
    "ended with status 'timeout'",
    "job.*failed",
)


def _extract_handle_ids(arguments: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in _HANDLE_FIELDS:
        val = str(arguments.get(key) or "").strip()
        if val:
            ids.append(val)
    return ids


def _is_runtime_failure(result_payload: dict[str, Any]) -> bool:
    text = json.dumps(result_payload, ensure_ascii=False).lower()
    return any(tok in text for tok in _RUNTIME_FAILURE_TOKENS)


def extract_failed_handles(
    *,
    arguments: dict[str, Any],
) -> dict[str, str]:
    return {hid: "runtime_operation_failed" for hid in _extract_handle_ids(arguments)}


@dataclass
class FailedHandleTracker:
    tracks: dict[str, str] = field(default_factory=dict)

    def update_from_result(
        self,
        *,
        result_payload: dict[str, Any],
        arguments: dict[str, Any],
    ) -> None:
        if not _is_runtime_failure(result_payload):
            return
        new = extract_failed_handles(
            arguments=arguments,
        )
        self.tracks.update(new)

    def check_arguments(self, arguments: dict[str, Any]) -> str | None:
        for key in _HANDLE_FIELDS:
            val = str(arguments.get(key) or "").strip()
            if val and val in self.tracks:
                return (
                    f"Handle {key}='{val}' belongs to a previously failed operation. "
                    "Do not query or analyse failed resources. "
                    "Either start a new operation or return a final_report describing the outcome."
                )
        return None

    @property
    def failed_ids(self) -> set[str]:
        return set(self.tracks.keys())


__all__ = ["FailedHandleTracker"]
