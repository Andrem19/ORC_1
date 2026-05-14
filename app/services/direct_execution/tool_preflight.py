"""
Schema-driven local preflight for direct dev_space1 tool calls.
"""

from __future__ import annotations

from typing import Any

from app.services.direct_execution.backtests_protocol import backtests_start_guard_payload
from app.services.direct_execution.feature_contract_runtime import (
    feature_contract_local_preflight,
)
from app.services.direct_execution.handle_preflight import handle_local_preflight
from app.services.mcp_catalog.models import McpCatalogSnapshot
from app.services.mcp_catalog.validator import ToolValidationResult
from app.services.mcp_catalog.validator import validate_tool_call

ToolPreflightResult = ToolValidationResult


def preflight_direct_tool_call(
    tool_name: str,
    arguments: dict[str, object],
    *,
    catalog_snapshot: McpCatalogSnapshot | None,
    confirmed_handles: dict[str, str] | None = None,
    transcript: list[dict[str, Any]] | None = None,
    runtime_profile: str = "",
    baseline_bootstrap: dict[str, Any] | None = None,
) -> ToolPreflightResult:
    if catalog_snapshot is None:
        return ToolPreflightResult(
            arguments=dict(arguments or {}),
            local_payload={
                "ok": False,
                "error_class": "agent_contract_misuse",
                "summary": "live MCP catalog snapshot is unavailable",
                "details": {
                    "tool_name": str(tool_name or "").strip(),
                    "reason_code": "mcp_catalog_unavailable",
                },
            },
            charge_budget=False,
        )
    pre_schema_handle_payload = handle_local_preflight(
        tool_name=tool_name,
        arguments=dict(arguments or {}),
        catalog_snapshot=catalog_snapshot,
        confirmed_handles=confirmed_handles,
    )
    if pre_schema_handle_payload is not None:
        return ToolPreflightResult(
            arguments=dict(arguments or {}),
            local_payload=pre_schema_handle_payload,
            charge_budget=False,
        )
    result = validate_tool_call(
        snapshot=catalog_snapshot,
        tool_name=tool_name,
        arguments=arguments,
    )
    if result.local_payload is not None:
        return result
    handle_payload = handle_local_preflight(
        tool_name=tool_name,
        arguments=result.arguments,
        catalog_snapshot=catalog_snapshot,
        confirmed_handles=confirmed_handles,
    )
    if handle_payload is not None:
        return ToolPreflightResult(
            arguments=result.arguments,
            local_payload=handle_payload,
            charge_budget=False,
        )
    start_guard_payload = backtests_start_guard_payload(
        tool_name=tool_name,
        arguments=result.arguments,
        transcript=transcript,
        runtime_profile=runtime_profile,
        baseline_bootstrap=baseline_bootstrap,
    )
    if start_guard_payload is not None:
        return ToolPreflightResult(
            arguments=result.arguments,
            local_payload=start_guard_payload,
            charge_budget=False,
        )
    local_payload = feature_contract_local_preflight(
        tool_name=tool_name,
        arguments=result.arguments,
    )
    if local_payload is not None:
        return ToolPreflightResult(
            arguments=result.arguments,
            local_payload=local_payload,
            charge_budget=False,
        )
    return result


__all__ = ["ToolPreflightResult", "preflight_direct_tool_call"]
