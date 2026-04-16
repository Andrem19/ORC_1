"""
Durable-handle hygiene helpers for direct execution.
"""

from __future__ import annotations

from typing import Any

DURABLE_HANDLE_FIELDS = (
    "project_id",
    "job_id",
    "run_id",
    "snapshot_id",
    "operation_id",
    "branch_id",
)

TRANSPORT_ID_FIELDS = frozenset(
    {
        "request_id",
        "correlation_id",
        "server_session_id",
        "session_id",
        "trace_id",
    }
)

_PLACEHOLDER_HANDLE_VALUES = frozenset(
    {
        "<job_id>",
        "<operation_id>",
        "<project_id>",
        "<run_id>",
        "<snapshot_id>",
        "job_id",
        "operation_id",
        "project_id",
        "run_id",
        "snapshot_id",
        "latest",
        "pending",
        "none",
        "null",
        "0",
        "unknown",
    }
)

_SESSION_HEX_LENGTHS = {24, 32, 40}
_STRICT_OBSERVE_ACTIONS = frozenset(
    {
        "detail",
        "events",
        "inspect",
        "logs",
        "result",
        "source",
        "status",
        "summary",
        "trades",
        "validation",
    }
)
_STRICT_OBSERVE_VIEWS = frozenset(
    {
        "artifacts",
        "compare_summary",
        "detail",
        "equity_chart",
        "events",
        "json",
        "logs",
        "meta",
        "price_chart",
        "result",
        "signal_breakdown",
        "source",
        "status",
        "summary",
        "text",
        "trades",
        "validation",
        "version_detail",
    }
)


def is_transport_id_field(field_name: str) -> bool:
    return str(field_name or "").strip().lower() in TRANSPORT_ID_FIELDS


def is_placeholder_handle_value(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return True
    return normalized in _PLACEHOLDER_HANDLE_VALUES


def looks_like_session_identifier(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text or len(text) not in _SESSION_HEX_LENGTHS:
        return False
    return all(ch in "0123456789abcdef" for ch in text)


def looks_like_numeric_identifier(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.isdigit()


def is_suspicious_handle_value(value: Any, *, field_name: str) -> bool:
    normalized_field = str(field_name or "").strip().lower()
    if normalized_field not in DURABLE_HANDLE_FIELDS:
        return False
    if is_placeholder_handle_value(value):
        return True
    if normalized_field == "project_id":
        return False
    return looks_like_numeric_identifier(value) or looks_like_session_identifier(value)


def should_validate_observe_handle(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    field_name: str,
) -> bool:
    normalized_tool = str(tool_name or "").strip()
    normalized_field = str(field_name or "").strip()
    if normalized_field not in {"job_id", "run_id", "operation_id"}:
        return False
    action = str(arguments.get("action") or "").strip().lower()
    view = str(arguments.get("view") or "").strip().lower()
    if action in _STRICT_OBSERVE_ACTIONS or view in _STRICT_OBSERVE_VIEWS:
        return True
    if normalized_tool == "backtests_runs":
        return action in {"inspect", "status"} or view in {
            "compare_summary",
            "detail",
            "equity_chart",
            "events",
            "price_chart",
            "signal_breakdown",
            "status",
            "trades",
        }
    return False


def durable_created_id_values(*, sources: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, raw_value in source.items():
            key_text = str(key or "").strip().lower()
            if is_transport_id_field(key_text):
                continue
            if not key_text.endswith("_id") and key_text not in {"id", "result_id"}:
                continue
            if key_text in DURABLE_HANDLE_FIELDS and is_suspicious_handle_value(raw_value, field_name=key_text):
                continue
            value = str(raw_value or "").strip()
            if not value:
                continue
            if key_text in {"id", "result_id"} and (looks_like_numeric_identifier(value) or looks_like_session_identifier(value)):
                continue
            if value not in values:
                values.append(value)
    return values


__all__ = [
    "DURABLE_HANDLE_FIELDS",
    "TRANSPORT_ID_FIELDS",
    "durable_created_id_values",
    "is_placeholder_handle_value",
    "is_suspicious_handle_value",
    "is_transport_id_field",
    "looks_like_numeric_identifier",
    "looks_like_session_identifier",
    "should_validate_observe_handle",
]
