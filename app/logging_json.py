"""
Structured JSON-lines logging helpers for file-only post-mortem traces.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any

_STANDARD_RECORD_FIELDS = {
    "args",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonLinesFormatter(logging.Formatter):
    """Serialize log records into compact JSONL entries."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "session_id": getattr(record, "session_id", ""),
            "event_kind": str(getattr(record, "event_kind", "") or ""),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _STANDARD_RECORD_FIELDS or key in payload:
                continue
            payload[key] = _json_safe(value)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        elif record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, BaseException):
        return {"type": value.__class__.__name__, "message": str(value)}
    if isinstance(value, traceback.StackSummary):
        return value.format()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return repr(value)
    return repr(value)

