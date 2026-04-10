"""
Prompt builder for planner-driven semantic plan creation.

The planner produces a SemanticRawPlan (same schema as converter's LLM
extraction step) instead of a direct ExecutionPlan.  This lets the
deterministic compiler (compile_semantic_raw_plan) handle tool expansion,
budget assignment, batching, and reconciliation — identical to the
converter path.
"""

from __future__ import annotations

from typing import Any

from app.raw_plan_prompts import SEMANTIC_RAW_PLAN_SCHEMA

_TOOL_FAMILY_NAMES = (
    "research_memory",
    "data_readiness",
    "feature_contract",
    "modeling",
    "backtesting",
    "analysis",
    "events",
    "experiments",
    "finalization",
)


def build_planner_semantic_prompt(
    *,
    goal: str,
    operator_directives: str = "",
    baseline_bootstrap: dict[str, Any] | None = None,
    plan_version: int = 1,
    worker_count: int = 1,
    available_tools: list[str] | None = None,
    previous_state_summary: str = "",
    previous_blockers: list[str] | None = None,
) -> str:
    snapshot_id = "active-signal-v1"
    baseline_version = 1
    symbol = "BTCUSDT"
    anchor = "1h"
    execution = "5m"
    if baseline_bootstrap:
        snapshot_id = baseline_bootstrap.get("baseline_snapshot_id", snapshot_id)
        baseline_version = baseline_bootstrap.get("baseline_version", baseline_version)
        symbol = baseline_bootstrap.get("symbol", symbol)
        anchor = baseline_bootstrap.get("anchor_timeframe", anchor)
        execution = baseline_bootstrap.get("execution_timeframe", execution)
    parts = [
        "You are a planner producing a semantic research plan for a direct model runtime.",
        "Do not call tools. Do not inspect files. Do not gather extra context.",
        "The worker model owns approved direct dev_space1 tool execution.",
        "",
        f"Plan version: {plan_version}",
        f"Goal: {goal}",
        f"Baseline: {snapshot_id}@{baseline_version}",
        f"Symbol/timeframes: {symbol}, anchor={anchor}, execution={execution}",
        f"Parallel worker slots available: {min(worker_count, 3)}",
        "",
        "Rules:",
        "- Return JSON only. No markdown, no commentary, no code fences.",
        "- Produce 1 to 6 stages. The system will batch them into execution plans of 3 slices each.",
        "- Each stage must have a non-empty objective and at least one action.",
        "- tool_hints must use semantic family names or individual tool names.",
        f"  Semantic families: {', '.join(_TOOL_FAMILY_NAMES)}.",
        f"  Individual tools: {', '.join(sorted(available_tools or []))}.",
        "- Do NOT set budget numbers (max_turns, max_tool_calls). The compiler assigns them automatically.",
        "- depends_on must reference earlier stage IDs only.",
        "- Mark optional stages with required=false.",
        "- Mark parallelizable stages with parallelizable=true.",
        "- Favor cheap validation before expensive studies or backtests.",
        "- Keep the baseline fixed. New work must seek orthogonal evidence, new trades, or missing regimes.",
        "",
    ]
    if operator_directives.strip():
        parts.extend(["Operator directives:", operator_directives.strip(), ""])
    if previous_state_summary.strip():
        parts.extend(["Previous execution summary:", previous_state_summary.strip(), ""])
    if previous_blockers:
        parts.extend(["Known blockers and capability gaps:"])
        parts.extend(f"- {item}" for item in previous_blockers[:8] if str(item).strip())
        parts.append("")
    parts.extend(
        [
            "Output schema:",
            SEMANTIC_RAW_PLAN_SCHEMA,
            "",
            "Validation constraints:",
            "- stage_id must be unique across all stages.",
            "- depends_on must reference existing earlier stage IDs.",
            "- Every stage must have at least one action and one success criterion.",
            "- If a target domain appears unavailable, frame the stage as discovery/availability validation first.",
        ]
    )
    return "\n".join(parts).strip() + "\n"
