"""
Broker-side execution policy for MCP tools.
"""

from __future__ import annotations

from dataclasses import replace

from app.execution_models import ToolPolicy


def infer_tool_policy(tool_name: str) -> ToolPolicy:
    expensive_tools = {
        "backtests_runs",
        "backtests_studies",
        "backtests_walkforward",
        "backtests_analysis",
        "backtests_conditions",
        "features_custom",
        "features_dataset",
        "models_train",
        "models_dataset",
        "experiments_run",
        "datasets_sync",
        "events_sync",
        "research_record",
    }
    async_resumable_tools = {
        "backtests_runs",
        "backtests_studies",
        "backtests_walkforward",
        "backtests_analysis",
        "backtests_conditions",
        "features_custom",
        "features_dataset",
        "models_train",
        "models_dataset",
        "experiments_run",
        "datasets_sync",
        "events_sync",
        "research_record",
    }
    mutating_tools = {
        "backtests_runs",
        "backtests_studies",
        "backtests_walkforward",
        "backtests_analysis",
        "backtests_conditions",
        "features_custom",
        "features_dataset",
        "models_train",
        "models_dataset",
        "datasets_sync",
        "events_sync",
        "research_record",
        "incidents",
    }
    wait_tools = async_resumable_tools
    if tool_name in wait_tools:
        allowed_wait_modes = ["none", "started"]
        default_wait_mode = "started" if tool_name in expensive_tools else ""
    else:
        allowed_wait_modes = []
        default_wait_mode = ""
    return ToolPolicy(
        tool_name=tool_name,
        expensive=tool_name in expensive_tools,
        async_resumable=tool_name in async_resumable_tools,
        mutating=tool_name in mutating_tools,
        autopoll_enabled=tool_name in async_resumable_tools,
        allowed_wait_modes=allowed_wait_modes,
        default_wait_mode=default_wait_mode,
    )


def policy_for_call(*, tool_name: str, arguments: dict[str, object]) -> ToolPolicy:
    base = infer_tool_policy(tool_name)
    action = str(arguments.get("action", "") or "").strip().lower()
    if tool_name == "features_custom" and action in {"inspect", "status"}:
        return replace(
            base,
            expensive=False,
            async_resumable=False,
            mutating=False,
            autopoll_enabled=False,
            allowed_wait_modes=[],
            default_wait_mode="",
        )
    if tool_name == "features_dataset" and action in {"inspect", "status", "build_plan"}:
        return replace(
            base,
            expensive=False,
            async_resumable=False,
            mutating=False,
            autopoll_enabled=False,
            allowed_wait_modes=[],
            default_wait_mode="",
        )
    return base


def normalize_wait_argument(*, tool_name: str, policy: ToolPolicy, arguments: dict[str, object], supports_wait: bool) -> dict[str, object]:
    if not supports_wait:
        return dict(arguments)
    normalized = dict(arguments)
    requested_wait = str(normalized.get("wait", "") or "").strip().lower()
    if requested_wait == "completed" and policy.expensive:
        raise ValueError(
            f"{tool_name}: wait='completed' is blocked by broker policy for expensive async tools; use wait='started' or omit wait."
        )
    if policy.allowed_wait_modes:
        if requested_wait and requested_wait not in set(policy.allowed_wait_modes):
            raise ValueError(f"{tool_name}: unsupported wait mode {requested_wait!r}; allowed={policy.allowed_wait_modes}")
        if not requested_wait and policy.default_wait_mode:
            normalized["wait"] = policy.default_wait_mode
    return normalized


__all__ = ["infer_tool_policy", "normalize_wait_argument", "policy_for_call"]
