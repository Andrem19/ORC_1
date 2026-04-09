"""
Helpers for stable task-result identity and deduplication.
"""

from __future__ import annotations

import hashlib

from app.models import TaskResult


def build_result_fingerprint(result: TaskResult) -> str:
    summary = (result.summary or "")[:500]
    error = (result.error or "")[:500]
    payload = "|".join(
        [
            result.task_id or "",
            result.worker_id or "",
            result.status or "",
            error,
            summary,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()


def ensure_result_fingerprint(result: TaskResult) -> str:
    if not result.result_fingerprint:
        result.result_fingerprint = build_result_fingerprint(result)
    return result.result_fingerprint
