"""
Canonical planner-facing MCP contract for plan-mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


INVALID_TOOL_ALIASES: dict[str, str] = {
    "snapshots": "backtests_strategy",
}

TOOL_ACTIONS: dict[str, set[str]] = {
    "datasets": {"catalog", "refresh", "inspect"},
    "datasets_sync": {"add", "sync"},
    "datasets_preview": {"rows", "chart"},
    "features_catalog": {"available", "timeframe"},
    "features_dataset": {"inspect", "build_plan", "build", "refresh", "remove", "verify", "status"},
    "features_custom": {"inspect", "status", "validate", "publish", "delete"},
    "features_analytics": {"build_outcomes", "status", "heatmap", "portability", "render", "analytics", "backfill"},
    "models_dataset": {"contract", "preview", "materialize", "status", "inspect"},
    "models_train": {"start", "status", "cancel"},
    "models_registry": {"inspect", "update_card", "promote_version", "delete_card", "backfill_reproducibility", "delete_version", "compare"},
    "models_to_feature": {"scaffold", "validate", "publish"},
    "models_compare": {"compare"},
    "backtests_strategy": {"inspect", "create", "save_version", "rename", "archive", "delete", "delete_version", "clone"},
    "backtests_strategy_validate": {"signal", "exit"},
    "backtests_plan": {"plan"},
    "backtests_runs": {"start", "inspect", "status", "cancel", "delete", "materialize_analysis", "purge_saved"},
    "backtests_studies": {"preview", "start", "list", "status", "result", "backfill_preview", "cancel"},
    "backtests_walkforward": {"start", "list", "status", "result", "cancel"},
    "backtests_conditions": {"run", "list", "status", "result", "cancel"},
    "backtests_analysis": {"start", "list", "status", "result", "cancel"},
    "research_project": {"list", "create", "open", "set_baseline"},
    "research_map": {"inspect", "define", "record_attempt", "advance_hypothesis"},
    "research_record": {"create", "update", "complete_work_item", "complete_work_items"},
    "research_search": {"search"},
    "events": {"catalog", "align_preview"},
    "events_sync": {"funding", "expiry", "all"},
    "experiments_run": {"describe", "start"},
    "signal_api_binding_apply": {"apply"},
    "gold_collection": {"inspect", "add", "remove"},
}

FORBIDDEN_ARG_PATTERNS: dict[tuple[str, str], set[str]] = {
    ("backtests_runs", "start"): {"strategy", "interval", "snapshot_name", "param_grid"},
    ("backtests_runs", "inspect"): {"strategy", "interval", "snapshot_name"},
}

REQUIRED_ARGS: dict[tuple[str, str], set[str]] = {
    ("backtests_plan", "plan"): {"snapshot_id", "symbol", "anchor_timeframe", "execution_timeframe"},
    ("backtests_runs", "start"): {"snapshot_id", "version", "symbol", "anchor_timeframe", "execution_timeframe"},
    ("backtests_strategy", "clone"): {"source_snapshot_id"},
}

DEFAULT_ACTION_BY_TOOL: dict[str, str] = {
    "backtests_plan": "plan",
    "models_compare": "compare",
    "signal_api_binding_apply": "apply",
}

CANONICAL_TOOL_TEMPLATES = """## Canonical MCP Tool Templates

- Baseline readiness:
  - `backtests_plan(snapshot_id='active-signal-v1', version='1', symbol='BTCUSDT', anchor_timeframe='1h', execution_timeframe='5m')`
- Start backtest:
  - `backtests_runs(action='start', snapshot_id='active-signal-v1', version='1', symbol='BTCUSDT', anchor_timeframe='1h', execution_timeframe='5m', start_at='2023-01-01T00:00:00+00:00', end_at='2024-12-31T23:55:00+00:00')`
- Inspect run:
  - `backtests_runs(action='inspect', run_id='{{stage:0.run_id}}', view='detail')`
- Compare candidate vs baseline:
  - `backtests_runs(action='inspect', view='compare_summary', run_id='{{stage:1.run_id}}', baseline_run_id='{{stage:0.run_id}}')`
- Clone snapshot:
  - `backtests_strategy(action='clone', source_snapshot_id='active-signal-v1', source_version='1', name='candidate-v1')`

Forbidden shortcuts:
- Do NOT use `snapshots(...)`
- Do NOT use `backtests_runs(action='run', ...)`
- Do NOT use `strategy=` or `interval=` in `backtests_runs`
"""


@dataclass(frozen=True)
class ToolContractViolation:
    code: str
    message: str
    suggestion: str = ""


TOOL_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def summarize_templates() -> str:
    return CANONICAL_TOOL_TEMPLATES


def known_tool_names() -> set[str]:
    return set(TOOL_ACTIONS)


def validate_tool_step(*, tool_name: str | None, args: dict[str, Any]) -> list[ToolContractViolation]:
    violations: list[ToolContractViolation] = []
    if not tool_name:
        return [ToolContractViolation("tool_name_missing", "tool_call step requires tool_name")]

    if tool_name in INVALID_TOOL_ALIASES:
        replacement = INVALID_TOOL_ALIASES[tool_name]
        return [ToolContractViolation(
            "tool_alias_invalid",
            f"Tool alias '{tool_name}' is not allowed",
            f"use {replacement}",
        )]

    if tool_name not in TOOL_ACTIONS:
        return [ToolContractViolation(
            "tool_alias_invalid",
            f"Unknown tool '{tool_name}'",
            "use a tool from MCP_TOOL_CATALOG",
        )]

    action = _normalize_action(tool_name, args)
    if action not in TOOL_ACTIONS[tool_name]:
        violations.append(ToolContractViolation(
            "action_invalid",
            f"{tool_name} does not support action '{action}'",
            f"use one of {sorted(TOOL_ACTIONS[tool_name])}",
        ))
        return violations

    forbidden = FORBIDDEN_ARG_PATTERNS.get((tool_name, action), set())
    for arg_name in sorted(set(args) & forbidden):
        violations.append(ToolContractViolation(
            "arg_invalid",
            f"Argument '{arg_name}' is not allowed for {tool_name}(action='{action}')",
            "use the canonical MCP facade arguments",
        ))

    required = REQUIRED_ARGS.get((tool_name, action), set())
    for arg_name in sorted(required):
        if arg_name not in args:
            violations.append(ToolContractViolation(
                "arg_missing",
                f"Missing required argument '{arg_name}' for {tool_name}(action='{action}')",
                "fill all required canonical arguments",
            ))

    return violations


def inspect_legacy_instruction(text: str) -> list[ToolContractViolation]:
    text = text.strip()
    violations: list[ToolContractViolation] = []

    alias_match = TOOL_CALL_RE.match(text)
    tool_name = alias_match.group(1) if alias_match else ""
    if tool_name in INVALID_TOOL_ALIASES:
        replacement = INVALID_TOOL_ALIASES[tool_name]
        violations.append(ToolContractViolation(
            "tool_alias_invalid",
            f"Tool alias '{tool_name}' is not allowed",
            f"use {replacement}",
        ))

    if "backtests_runs(" in text and "action='run'" in text:
        violations.append(ToolContractViolation(
            "action_invalid",
            "backtests_runs(action='run') is not allowed",
            "use backtests_runs(action='start')",
        ))
    if "backtests_runs(" in text and 'action="run"' in text:
        violations.append(ToolContractViolation(
            "action_invalid",
            'backtests_runs(action="run") is not allowed',
            "use backtests_runs(action='start')",
        ))

    if "backtests_runs(" in text:
        for arg_name in ("strategy=", "interval=", "snapshot_name=", "param_grid="):
            if arg_name in text:
                violations.append(ToolContractViolation(
                    "arg_invalid",
                    f"Legacy argument '{arg_name[:-1]}' is not allowed in backtests_runs",
                    "use canonical start/inspect arguments",
                ))

    if "<returned_run_id>" in text or "<winner_snapshot_ref>" in text:
        violations.append(ToolContractViolation(
            "step_ref_invalid",
            "Narrative pseudo variables are not allowed",
            "use {{step:step_id.field}}",
        ))

    return violations


def format_step_as_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    if not args:
        return f"{tool_name}()"
    rendered_args = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in args.items()
    )
    return f"{tool_name}({rendered_args})"


def _normalize_action(tool_name: str, args: dict[str, Any]) -> str:
    action = args.get("action")
    if action is None:
        return DEFAULT_ACTION_BY_TOOL.get(tool_name, "")
    return str(action)
