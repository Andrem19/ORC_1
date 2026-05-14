"""
Local semantic preflight for suspicious durable handles.
"""

from __future__ import annotations

from typing import Any

from app.services.direct_execution.handle_hygiene import (
    DURABLE_HANDLE_FIELDS,
    is_suspicious_handle_value,
    should_validate_observe_handle,
)
from app.services.mcp_catalog.models import McpCatalogSnapshot


def handle_local_preflight(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    catalog_snapshot: McpCatalogSnapshot,
    confirmed_handles: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    tool = catalog_snapshot.get_tool(tool_name)
    if tool is None:
        return None
    confirmed = {
        str(key).strip(): str(value).strip()
        for key, value in dict(confirmed_handles or {}).items()
        if str(key).strip() and str(value).strip()
    }
    for field_name in DURABLE_HANDLE_FIELDS:
        if field_name not in tool.accepted_handle_fields:
            continue
        value = arguments.get(field_name)
        if value is None:
            continue
        if confirmed.get(field_name) == str(value).strip():
            continue
        if not should_validate_observe_handle(
            tool_name=tool_name,
            arguments=arguments,
            field_name=field_name,
        ):
            continue
        if not is_suspicious_handle_value(value, field_name=field_name):
            continue
        return {
            "ok": False,
            "error_class": "agent_contract_misuse",
            "summary": (
                f"{tool_name} received suspicious {field_name}='{value}'. "
                "Use a durable handle from a prior successful result instead of transport/session metadata."
            ),
            "details": {
                "tool_name": str(tool_name or "").strip(),
                "field_name": field_name,
                "handle_value": str(value),
                "reason_code": "suspicious_durable_handle",
                "accepted_handle_fields": list(tool.accepted_handle_fields),
            },
        }
    return None


__all__ = ["handle_local_preflight"]
