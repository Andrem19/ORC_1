"""
Prompt builder for direct slice execution.
"""

from __future__ import annotations

from typing import Any


def build_direct_slice_prompt(
    *,
    plan_id: str,
    slice_payload: dict[str, Any],
    baseline_bootstrap: dict[str, Any],
    known_facts: dict[str, Any],
    recent_turn_summaries: list[str],
    checkpoint_summary: str,
    allowed_tools: list[str],
    max_tool_calls: int,
    max_expensive_tool_calls: int,
    worker_system_prompt: str = "",
    required_output_facts: list[str] | None = None,
) -> str:
    lines: list[str] = []
    if worker_system_prompt.strip():
        lines.append(worker_system_prompt.strip())
        lines.append("")
    lines.extend(
        [
            "You are a direct execution worker for trading research.",
            "You may use only the approved dev_space1 tools listed below.",
            "Do not use shell, filesystem, workspace, web, or unrelated local tools.",
            "Complete this slice end-to-end when possible, then return exactly one terminal JSON object.",
            "Never return an intermediate tool_call action to the orchestrator.",
            "You MUST call at least one tool before returning a checkpoint. Do not return a checkpoint without using tools.",
            "",
            "Critical dev_space1 incident rule:",
            "- If a dev_space1 tool has an infrastructure issue, unclear contract, surprising required field, broken path, or ambiguous behavior, record an incident with the incidents tool and include precise debugging details in reportable_issues.",
            "",
            "Backtest/model safety:",
            "- Do not restart heavy operations when a run_id/job_id/operation_id is already known.",
            "- Start expensive backtests, studies, syncs, or model training only when directly required by the slice.",
            "- Prefer status/result polling for known operation ids over duplicate starts.",
            f"- Tool-call budget for this direct session: {max_tool_calls}.",
            f"- Expensive-tool budget for this direct session: {max_expensive_tool_calls}.",
            "",
            "Output contract:",
            "- Return exactly one JSON object wrapped in ```json``` fences.",
            "- Allowed final types: final_report, checkpoint, abort.",
            "- final_report requires summary, verdict, findings, facts, evidence_refs, confidence.",
            "- checkpoint requires status=partial|complete|blocked and summary.",
            "- abort requires reason_code and summary; use abort only for true impossibility or hard failure.",
            "- Do not output prose outside the JSON object.",
            "",
            f"Plan id: {plan_id}",
            f"Slice id: {slice_payload.get('slice_id', '')}",
            f"Title: {slice_payload.get('title', '')}",
            f"Hypothesis: {slice_payload.get('hypothesis', '')}",
            f"Objective: {slice_payload.get('objective', '')}",
            (
                "Baseline: "
                f"{baseline_bootstrap.get('baseline_snapshot_id', 'active-signal-v1')}@"
                f"{baseline_bootstrap.get('baseline_version', 1)} "
                f"symbol={baseline_bootstrap.get('symbol', 'BTCUSDT')} "
                f"anchor={baseline_bootstrap.get('anchor_timeframe', '1h')} "
                f"execution={baseline_bootstrap.get('execution_timeframe', '5m')}"
            ),
            "",
            "Success criteria:",
        ]
    )
    lines.extend(f"- {item}" for item in slice_payload.get("success_criteria", []) or [])
    lines.extend(["", "Evidence requirements:"])
    lines.extend(f"- {item}" for item in slice_payload.get("evidence_requirements", []) or [])
    lines.extend(["", "Allowed dev_space1 tools:"])
    lines.extend(f"- {item}" for item in allowed_tools)
    contract_hints = _tool_contract_hints(allowed_tools=allowed_tools, known_facts=known_facts)
    if contract_hints:
        lines.extend(["", "Tool contract hints:"])
        lines.extend(f"- {item}" for item in contract_hints)
    if checkpoint_summary.strip():
        lines.extend(["", "Previous checkpoint:", checkpoint_summary.strip()])
    if recent_turn_summaries:
        lines.extend(["", "Recent direct attempts:"])
        lines.extend(f"- {item}" for item in recent_turn_summaries[-6:])
    if known_facts:
        lines.extend(["", "Known facts:"])
        for key in sorted(known_facts)[:100]:
            value = str(known_facts[key])
            if len(value) > 260:
                value = value[:257] + "..."
            lines.append(f"- {key} = {value}")
    if required_output_facts:
        lines.extend(["", "Required downstream facts:"])
        lines.extend(f"- Include `{item}` in final_report.facts or create tool evidence that deterministically yields it." for item in required_output_facts)
        lines.extend(
            [
                "- Do not return final_report if required downstream facts are still missing.",
                "- For research short-list slices, final_report.facts must carry active research project id, short-list family names, and created hypothesis node refs.",
            ]
        )
    lines.extend(
        [
            "",
            "Example final_report:",
            "```json",
            '{"type":"final_report","summary":"...","verdict":"WATCHLIST","findings":[],"facts":{},"evidence_refs":[],"confidence":0.7}',
            "```",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _tool_contract_hints(*, allowed_tools: list[str], known_facts: dict[str, Any]) -> list[str]:
    allowed = {str(item).strip() for item in allowed_tools if str(item).strip()}
    hints: list[str] = [
        "If a tool error gives an exact remediation, rewrite the same call accordingly instead of probing unrelated tools.",
    ]
    atlas_dimensions = _first_fact_value(known_facts, suffix="atlas_dimensions")
    if "research_record" in allowed:
        hints.append(
            "For research_record(action='create', kind='hypothesis') in atlas-enabled projects, pass top-level "
            "atlas={statement, expected_outcome, falsification_criteria, coordinates}."
        )
        hints.append("Do not put atlas inside record, and do not serialize coordinates as a string.")
        if atlas_dimensions:
            hints.append(f"Use atlas.coordinates as an object keyed by the active dimensions: {atlas_dimensions}.")
        hints.append(
            "If you are only documenting a short-list or observation, prefer research_record(kind='note' or 'milestone') "
            "instead of hypothesis creation."
        )
    if "research_search" in allowed:
        hints.append(
            "research_search requires a concrete non-empty query; do not use it as a generic project listing call."
        )
    if "research_map" in allowed:
        hints.append("Use research_map.inspect to inspect atlas state; use research_map.define only when dimensions are still missing.")
        hints.append(
            "research_map always requires project_id. Never call research_map without project_id. "
            "Get it from research_project(action='list') or from confirmed known_facts."
        )
    if "research_project" in allowed:
        hints.append(
            "research_project(action='create') returns project_id in the ids field of the tool result. "
            "Extract and reuse it for subsequent tool calls immediately — do not call list or current to find it."
        )
        hints.append(
            "Before research_project(action='create'), call research_project(action='list') "
            "to check if a project with the same name already exists."
        )
    if "experiments_read" in allowed:
        hints.append(
            "experiments_read is NOT a listing tool. It always requires a concrete job_id. "
            "To list available jobs, call experiments_inspect(view='list') first."
        )
    if "experiments_inspect" in allowed:
        hints.append(
            "experiments_inspect(view='status') requires job_id. "
            "Call experiments_inspect(view='list') first to see available jobs, then retry with a specific job_id."
        )
    if any(t in allowed for t in ("backtests_conditions", "backtests_analysis", "backtests_runs")):
        hints.append(
            "backtests_conditions and backtests_analysis require snapshot_id resolved from "
            "backtests_strategy(action='inspect', view='detail', snapshot_id='active-signal-v1', version='1'). "
            "Always look up snapshot_id before calling these tools."
        )
    return hints


def _first_fact_value(known_facts: dict[str, Any], *, suffix: str) -> str:
    for key in sorted(known_facts):
        if key.endswith(f".{suffix}") or key == suffix:
            value = known_facts[key]
            if isinstance(value, (list, tuple)):
                return ", ".join(str(item) for item in value)
            return str(value)
    return ""


__all__ = ["build_direct_slice_prompt"]
