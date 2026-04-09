"""
Canonical prompt builders for the markdown wave orchestrator.
"""

from __future__ import annotations

from string import Template
from textwrap import dedent
from typing import Any


WORKER_REPORT_SCHEMA = """{
  "status": "success|error|partial",
  "what_was_requested": "brief summary of what was asked",
  "what_was_done": "detailed summary of what was actually done",
  "results_table": [
    {"run_id": "...", "net_pnl": 0, "trades": 0, "PF": 0.0, "WR": 0.0, "verdict": "PROMOTE|REJECT|WATCHLIST"}
  ],
  "key_metrics": {"metric_name": value},
  "artifacts": ["list of produced files or items"],
  "verdict": "PROMOTE|WATCHLIST|REJECT|PENDING",
  "confidence": 0.9,
  "error": "error message if any, empty otherwise",
  "mcp_problems": [{"tool_name": "...", "description": "...", "suggestion": "...", "severity": "low|medium|high"}]
}"""

EXECUTION_PLAN_SCHEMA = """{
  "plan_id": "plan_...",
  "goal": "one concise sentence",
  "baseline_ref": {
    "snapshot_id": "active-signal-v1",
    "version": 1,
    "symbol": "BTCUSDT",
    "anchor_timeframe": "1h",
    "execution_timeframe": "5m"
  },
  "global_constraints": [
    "keep 1h/5m timeframes fixed",
    "do not retune baseline thresholds"
  ],
  "slices": [
    {
      "slice_id": "slice_1",
      "title": "Funding signal readiness",
      "hypothesis": "funding events can add orthogonal information",
      "objective": "validate data availability and cheap signal evidence before heavy work",
      "success_criteria": ["one", "two"],
      "allowed_tools": ["events", "events_sync", "features_custom"],
      "evidence_requirements": ["data freshness", "cheap validation output"],
      "policy_tags": ["cheap_first", "orthogonal_signal"],
      "max_turns": 8,
      "max_tool_calls": 5,
      "max_expensive_calls": 1,
      "parallel_slot": 1
    }
  ]
}"""

WORKER_ACTION_SCHEMA = """{
  "type": "tool_call|checkpoint|final_report|abort",
  "tool": "required only for tool_call",
  "arguments": {"tool": "args"},
  "reason": "why this is the next best move",
  "expected_evidence": ["what this call should confirm"],
  "status": "required only for checkpoint: partial|complete|blocked",
  "summary": "required for checkpoint/final_report/abort",
  "facts": {"new_fact": "value"},
  "artifacts": ["ids or file refs"],
  "pending_questions": ["open issue"],
  "reportable_issues": [
    {
      "summary": "runtime or contract issue",
      "severity": "low|medium|high|critical",
      "details": "exact problem",
      "affected_tool": "tool_name",
      "category": "runtime|contract"
    }
  ],
  "verdict": "required only for final_report",
  "key_metrics": {"metric": 1.23},
  "findings": ["confirmed finding"],
  "rejected_findings": ["thing that failed"],
  "next_actions": ["what should happen next"],
  "risks": ["important unresolved risk"],
  "evidence_refs": ["artifact id or report path"],
  "confidence": 0.0,
  "reason_code": "required only for abort",
  "retryable": false
}"""


STRICT_RETRY_PREFIX = """Write the final markdown plan immediately.
Do not use tools. Do not inspect files. Do not gather more context. Do not spawn subagents.
If you previously started thinking or tried tool use, stop and write the plan now.
Be concise. Do not use fenced code blocks, pseudocode, or multi-line snippets.
Keep the plan short enough to fit comfortably in one response.
Do not name MCP tools or embed raw Python/API syntax; describe the work in plain English.
$retry_reason_section"""


def build_plan_creation_prompt(
    *,
    goal: str,
    operator_directives: str = "",
    research_context: str | None = None,
    cumulative_findings: str = "",
    anti_patterns: list[str] | None = None,
    research_history: list[str] | None = None,
    previous_plan_markdown: str | None = None,
    plan_version: int = 1,
    worker_count: int = 1,
    baseline_bootstrap: dict[str, Any] | None = None,
    prompt_profile: str = "compact",
    wave_summary: str = "",
    wave_context: str = "",
    prompt_template: str = "",
    retry_reason: str = "",
) -> str:
    from app.config import DEFAULT_PLANNER_PROMPT_TEMPLATE

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

    def _section(title: str, body: str, *, limit: int = 0) -> str:
        text = (body or "").strip()
        if not text:
            return ""
        if limit > 0:
            text = text[:limit]
        return f"## {title}\n{text}\n\n"

    def _list_section(title: str, items: list[str] | None, *, limit: int, item_limit: int) -> str:
        lines = [f"- {item[:item_limit]}" for item in (items or [])[:limit] if item]
        if not lines:
            return ""
        return f"## {title}\n" + "\n".join(lines) + "\n\n"

    substitutions = {
        "goal": goal[:320],
        "plan_version": str(plan_version),
        "worker_count": str(worker_count),
        "wave_context": wave_context or f"starting a new wave, slot 1 of up to {min(worker_count, 3)}",
        "baseline_snapshot_id": str(snapshot_id),
        "baseline_version": str(baseline_version),
        "symbol": str(symbol),
        "anchor_timeframe": str(anchor),
        "execution_timeframe": str(execution),
        "previous_wave_summary_section": _section("Previous Wave Summary", wave_summary, limit=1000),
        "previous_findings_section": _section("Previous Findings", cumulative_findings, limit=1000),
        "anti_patterns_section": _list_section("Anti-Patterns", anti_patterns, limit=6, item_limit=200),
        "research_history_section": _list_section("Research History", research_history, limit=8, item_limit=200),
        "previous_plan_excerpt_section": _section("Previous Plan Excerpt", previous_plan_markdown or "", limit=800),
        "research_context_section": _section("Research Context", research_context or "", limit=1200),
        "retry_reason_section": _section("Retry Reason", retry_reason.replace("_", " "), limit=200),
    }

    chosen_template = prompt_template or DEFAULT_PLANNER_PROMPT_TEMPLATE
    rendered = Template(dedent(chosen_template).strip()).safe_substitute(substitutions).strip()
    parts: list[str] = []
    if prompt_profile == "strict":
        parts.append(Template(STRICT_RETRY_PREFIX).safe_substitute(substitutions).strip())
    if operator_directives.strip():
        parts.append(f"## Operator Directives\n{operator_directives.strip()}")
    parts.append(rendered)
    return "\n\n".join(part for part in parts if part).strip() + "\n"


def build_worker_prompt(
    *,
    plan_markdown: str,
    plan_version: int,
    worker_system_prompt: str = "",
    previous_reports_summary: str = "",
    current_etap_markdown: str | None = None,
    checkpoint_summary: str = "",
    continuation_note: str = "",
) -> str:
    parts: list[str] = []
    if worker_system_prompt:
        parts.append(worker_system_prompt)
        parts.append("")
    parts.append(f"# Execute Research Plan v{plan_version}")
    if current_etap_markdown:
        parts.append(
            "Execute ONLY the current ETAP for this worker session. "
            "Treat previous ETAP summaries as immutable context. "
            "Do not repeat completed work unless verification is strictly required."
        )
    else:
        parts.append(
            "Execute ALL ETAPs in this plan sequentially. "
            "Follow each step precisely. Use MCP dev_space1 tools as instructed."
        )
    parts.append("")
    if previous_reports_summary:
        parts.append("## Previous Plan Results (for context)")
        parts.append(previous_reports_summary)
        parts.append("")
    if checkpoint_summary:
        parts.append("## Execution Checkpoint")
        parts.append(checkpoint_summary)
        parts.append("")
    if continuation_note:
        parts.append("## Continuation")
        parts.append(continuation_note)
        parts.append("")
    if current_etap_markdown:
        parts.append("## Current ETAP")
        parts.append(current_etap_markdown.strip())
        parts.append("")
        parts.append("## Full Plan Context")
    parts.append(plan_markdown)
    parts.append("")
    parts.append("## Required Output Format")
    parts.append("When finished, respond with a JSON object matching this schema:")
    parts.append("```json")
    parts.append(WORKER_REPORT_SCHEMA)
    parts.append("```")
    parts.append("")
    parts.append(
        "CRITICAL:\n"
        "- Respond with JSON ONLY. No preamble text, markdown commentary, or code fences.\n"
        "- First non-whitespace character must be `{` and last must be `}`.\n"
        "- `what_was_done` must detail EVERYTHING you did across ALL ETAPs.\n"
        "- `results_table` must contain one row per experiment/run.\n"
        "- `key_metrics` must include the most important numbers.\n"
        "- `verdict` must be honest: PROMOTE if results are good, REJECT if bad, WATCHLIST if uncertain.\n"
        "- If a stage fails, note it in `error` but continue with remaining stages.\n"
        "- If any MCP tool call reports `tool not found in registry`, stop further tool use, summarize the checkpoint, and return terminal JSON immediately.\n"
    )
    return "\n".join(parts)


def build_worker_resume_prompt(
    *,
    plan_markdown: str,
    current_etap_markdown: str,
    plan_version: int,
    worker_system_prompt: str = "",
    previous_reports_summary: str = "",
    checkpoint_summary: str = "",
) -> str:
    return build_worker_prompt(
        plan_markdown=plan_markdown,
        plan_version=plan_version,
        worker_system_prompt=worker_system_prompt,
        previous_reports_summary=previous_reports_summary,
        current_etap_markdown=current_etap_markdown,
        checkpoint_summary=checkpoint_summary,
        continuation_note=(
            "Previous worker session lost MCP tool surface. Continue from the checkpoint only. "
            "Treat checkpoint artifacts and IDs as authoritative handoff context. "
            "Do not recreate confirmed artifacts, rerun completed jobs, or repeat expensive work unless verification is strictly required. "
            "Resume from the earliest unfinished step in the current ETAP. "
            "If dev_space1 tools are still unavailable, return terminal JSON immediately and stop further tool use."
        ),
    )


def build_findings_summary(reports: list[dict[str, Any]], max_entries: int = 10) -> str:
    if not reports:
        return ""
    lines: list[str] = []
    for report in reports[-max_entries:]:
        line = f"Plan v{report.get('plan_version', '?')} ({report.get('worker_id', '?')}): {report.get('status', '?')}"
        verdict = report.get("verdict", "PENDING")
        if verdict != "PENDING":
            line += f" -> {verdict}"
        metrics = report.get("key_metrics", {}) or {}
        metric_parts = [f"{k}={v}" for k, v in list(metrics.items())[:5]]
        if metric_parts:
            line += f" | {', '.join(metric_parts)}"
        done = str(report.get("what_was_done", "") or "")[:120]
        if done:
            line += f" | {done}"
        lines.append(line)
    return "\n".join(lines)


def build_brokered_plan_creation_prompt(
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
    tools = ", ".join(sorted(available_tools or []))
    parts = [
        "You are a planner that writes a structured JSON execution plan for a brokered runtime.",
        "Do not call tools. Do not inspect files. Do not gather extra context.",
        "The worker model will not own tools. The broker owns tool execution.",
        "",
        f"Plan version: {plan_version}",
        f"Goal: {goal}",
        f"Baseline: {snapshot_id}@{baseline_version}",
        f"Symbol/timeframes: {symbol}, anchor={anchor}, execution={execution}",
        f"Parallel worker slots available: {min(worker_count, 3)}",
        "",
        "Rules:",
        "- Return JSON only. No markdown, no commentary, no code fences.",
        "- Produce 1 to 3 slices.",
        "- Each slice must be independently executable and bounded.",
        "- Do not embed concrete tool call arguments beyond allowed tool names.",
        "- Favor cheap validation before expensive studies or backtests.",
        "- Keep the baseline fixed. New work must seek orthogonal evidence, new trades, or missing regimes.",
        f"- Allowed tool names for slices: {tools}",
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
            EXECUTION_PLAN_SCHEMA,
            "",
            "Validation constraints:",
            "- `parallel_slot` must be unique and between 1 and 3.",
            "- `max_turns` and `max_tool_calls` must be positive integers.",
            "- `allowed_tools` must be a non-empty subset of the allowed tool names.",
            "- Do not assume a feature family, event family, or dataset exists unless it is directly supported by the allowed public tool surface or explicitly confirmed by previous blockers/context.",
            "- If a target domain appears unavailable, frame the slice as discovery/availability validation first, not direct feature engineering.",
            "- If previous blockers mention missing liquidation-family coverage, do not plan liquidation_* feature engineering as already feasible.",
        ]
    )
    return "\n".join(parts).strip() + "\n"


def build_brokered_worker_prompt(
    *,
    plan_id: str,
    slice_payload: dict[str, Any],
    worker_system_prompt: str = "",
    baseline_bootstrap: dict[str, Any] | None = None,
    known_facts: dict[str, Any] | None = None,
    recent_turn_summaries: list[str] | None = None,
    latest_tool_summary: str = "",
    remaining_budget: dict[str, int] | None = None,
    checkpoint_summary: str = "",
    active_operation: dict[str, Any] | None = None,
) -> str:
    baseline_bootstrap = baseline_bootstrap or {}
    known_facts = known_facts or {}
    recent_turn_summaries = recent_turn_summaries or []
    remaining_budget = remaining_budget or {}
    active_operation = active_operation or {}
    parts: list[str] = []
    if worker_system_prompt.strip():
        parts.append(worker_system_prompt.strip())
        parts.append("")
    parts.extend(
        [
            "You are a worker decision model in a brokered execution runtime.",
            "Do not call tools directly. Choose only the next best action.",
            "Return JSON only. Exactly one action object. No markdown, no commentary, no code fences.",
            "",
            f"Plan: {plan_id}",
            f"Slice: {slice_payload.get('slice_id', '')} | {_compact_text(slice_payload.get('title', ''), 120)}",
            f"Hypothesis: {_compact_text(slice_payload.get('hypothesis', ''), 320)}",
            f"Objective: {_compact_text(slice_payload.get('objective', ''), 320)}",
            f"Success criteria: {', '.join(_compact_list(slice_payload.get('success_criteria', []) or [], item_limit=3, char_limit=160))}",
            f"Allowed tools: {', '.join(slice_payload.get('allowed_tools', []) or [])}",
            "Tool naming examples:",
            "- valid: features_catalog",
            "- valid: events",
            "- valid: research_record",
            "- invalid: mcp__dev_space1__features_catalog",
            (
                "Budgets: "
                f"turns={slice_payload.get('max_turns')} "
                f"tool_calls={slice_payload.get('max_tool_calls')} "
                f"expensive_calls={slice_payload.get('max_expensive_calls')}"
            ),
            (
                "Remaining budget: "
                f"turns_used={remaining_budget.get('turns_used', 0)} "
                f"turns_remaining={remaining_budget.get('turns_remaining', slice_payload.get('max_turns', 0))} "
                f"tool_calls_used={remaining_budget.get('tool_calls_used', 0)} "
                f"tool_calls_remaining={remaining_budget.get('tool_calls_remaining', slice_payload.get('max_tool_calls', 0))} "
                f"expensive_calls_used={remaining_budget.get('expensive_calls_used', 0)} "
                f"expensive_calls_remaining={remaining_budget.get('expensive_calls_remaining', slice_payload.get('max_expensive_calls', 0))}"
            ),
            "",
            "Context:",
            (
                "Baseline: "
                f"{baseline_bootstrap.get('baseline_snapshot_id', 'active-signal-v1')}@"
                f"{baseline_bootstrap.get('baseline_version', 1)} "
                f"symbol={baseline_bootstrap.get('symbol', 'BTCUSDT')} "
                f"anchor={baseline_bootstrap.get('anchor_timeframe', '1h')} "
                f"execution={baseline_bootstrap.get('execution_timeframe', '5m')}"
            ),
            f"Known facts: {_compact_mapping(known_facts)}",
        ]
    )
    if recent_turn_summaries:
        parts.append("Recent turn summaries:")
        parts.extend(f"- {_compact_text(item, 180)}" for item in recent_turn_summaries[-4:])
    if checkpoint_summary.strip():
        parts.extend(["Last checkpoint summary:", _compact_text(checkpoint_summary.strip(), 220)])
    if latest_tool_summary.strip():
        parts.extend(["Latest broker tool result:", _compact_text(latest_tool_summary.strip(), 220)])
    if active_operation.get("ref"):
        parts.extend(
            [
                "Active broker-owned operation:",
                f"- tool={active_operation.get('tool', '')}",
                f"- ref={active_operation.get('ref', '')}",
                f"- status={active_operation.get('status', '')}",
            ]
        )
    constraints = _broker_enforced_constraints(slice_payload.get("allowed_tools", []) or [])
    if constraints:
        parts.extend(["Broker-enforced tool constraints:"])
        parts.extend(f"- {item}" for item in constraints)
    parts.extend(
        [
            "",
            "Allowed action types:",
            "- `tool_call`: choose exactly one next tool and arguments.",
            "- `checkpoint`: summarize progress when enough evidence was gathered for a checkpoint.",
            "- `final_report`: finish the slice with a verdict.",
            "- `abort`: stop when the slice is blocked or the budget is exhausted.",
            "",
            "Output schema:",
            WORKER_ACTION_SCHEMA,
            "",
            "Important rules:",
            "- Never emit more than one tool call.",
            "- Do not repeat a completed tool call without a clear reason.",
            "- If an active broker operation is still running, prefer waiting for its status rather than starting duplicate work.",
            "- Report contract/runtime issues in `reportable_issues` when needed.",
            "- Use `checkpoint` or `final_report` as soon as the current objective is satisfied.",
            "- Use only the exact literal tool names shown in `Allowed tools`.",
            "- Never add MCP namespace, provider prefix, or decoration to a tool name.",
            "- Do not treat the worker CLI session's own tool registry as relevant. The broker executes the `Allowed tools` externally even if your local session shows no MCP tools.",
            "- If you are unsure which exact literal tool name to use, return `abort` with `reason_code=\"tool_name_ambiguous\"`.",
        ]
    )
    return "\n".join(parts).strip() + "\n"


def _broker_enforced_constraints(allowed_tools: list[str]) -> list[str]:
    allowed = set(str(item) for item in allowed_tools if str(item).strip())
    constraints: list[str] = []
    if "research_search" in allowed:
        constraints.append("For `research_search`, use `level=\"normal\"`. Do not use `compact`.")
        constraints.append("For `research_search`, do not use the baseline snapshot id as `project_id`.")
        constraints.append("For `research_search`, omit `project_id` unless a real research project id was explicitly confirmed by earlier tool results or known facts.")
    if {"events_sync", "datasets_sync"} & allowed:
        constraints.append("For expensive async sync tools, never use `wait=\"completed\"`; use `wait=\"started\"` or omit `wait`.")
    if "events_sync" in allowed:
        constraints.append("For `events_sync`, always include both `family` and `scope` in the arguments.")
        constraints.append("For `events_sync`, use `family` from the public contract such as `funding`, `expiry`, or `all`, and `scope` such as `incremental` or `backfill`.")
        constraints.append("If the broker already reports an active `events_sync` operation as running, do not start another `events_sync` call with the same intent; wait for resume/status or choose another cheap discovery step.")
    if "features_catalog" in allowed:
        constraints.append("For `features_catalog`, prefer `scope=\"available\"`; do not use deprecated alias values like `all` unless a prior tool result explicitly requires it.")
    if "features_custom" in allowed:
        constraints.append("For `features_custom`, never use `action=\"create\"`. Valid public actions are inspect, status, validate, publish, and delete.")
        constraints.append("Before proposing a new custom feature, prefer `features_custom(action=\"inspect\", view=\"contract\")` to confirm the exact public authoring contract.")
    if "features_analytics" in allowed:
        constraints.append("For `features_analytics`, always specify one concrete feature via `feature_name`, `feature`, `column_name`, or `column`.")
        constraints.append("Do not call `features_analytics(action=\"heatmap\"|\"analytics\"|\"render\"|\"portability\")` without a specific feature selector.")
        constraints.append("If a prior broker result says analytics are not ready for a feature, do not immediately retry the same `features_analytics` call; switch to discovery, checkpoint, or abort.")
    if "datasets_preview" in allowed:
        constraints.append("For `datasets_preview`, always include both `dataset_id` and `view`.")
        constraints.append("For `datasets_preview`, `view` must be `rows` or `chart`.")
        constraints.append("Do not pass `symbol` or `timeframes` directly to `datasets_preview`; first discover a canonical dataset id using `datasets`.")
    return constraints


def _compact_text(value: Any, char_limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= char_limit:
        return text
    return text[: max(0, char_limit - 3)].rstrip() + "..."


def _compact_list(values: list[Any], *, item_limit: int, char_limit: int) -> list[str]:
    result: list[str] = []
    for item in list(values)[:item_limit]:
        text = _compact_text(item, char_limit)
        if text:
            result.append(text)
    return result


def _compact_mapping(mapping: dict[str, Any], *, item_limit: int = 8, value_limit: int = 120) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for index, (key, value) in enumerate((mapping or {}).items()):
        if index >= item_limit:
            compact["..."] = f"+{len(mapping) - item_limit} more"
            break
        if isinstance(value, dict):
            compact[str(key)] = _compact_mapping(value, item_limit=4, value_limit=value_limit)
        elif isinstance(value, list):
            compact[str(key)] = [_compact_text(item, value_limit) for item in value[:4]]
            if len(value) > 4:
                compact[str(key)].append(f"+{len(value) - 4} more")
        else:
            compact[str(key)] = _compact_text(value, value_limit)
    return compact


def build_brokered_worker_correction_prompt(
    *,
    previous_prompt: str,
    raw_output: str,
    allowed_tools: list[str],
    parse_error: str,
) -> str:
    return (
        f"{previous_prompt.rstrip()}\n\n"
        "## Correction Required\n"
        "Your previous response violated the worker tool-name contract.\n"
        f"Parse error: {parse_error}\n"
        f"Allowed literal tool names: {', '.join(allowed_tools)}\n"
        f"Previous invalid raw output:\n{raw_output}\n\n"
        "Rewrite the response now.\n"
        "Return JSON only.\n"
        "If the issue concerns `features_custom`, do not invent legacy actions like `create`; use only the public contract.\n"
        "If you choose `tool_call`, the `tool` field must be one exact literal name from Allowed tools.\n"
        "Do not use MCP-prefixed names like `mcp__dev_space1__...`.\n"
        "Do not claim that the allowed tools are unavailable just because your local CLI session has no MCP tools; the broker executes Allowed tools outside your session.\n"
        "If you are unsure, return `abort` with `reason_code=\"tool_name_ambiguous\"`.\n"
    )


def build_wave_summary(*, wave_id: int, reports: list[dict[str, Any]], target_size: int, slots: list[dict[str, Any]] | None = None) -> str:
    slot_items = slots or []
    if not slot_items:
        slot_items = [
            {
                "slot": report.get("wave_position", "?"),
                "kind": "terminal_report",
                "status": report.get("status", "?"),
                "verdict": report.get("verdict", "PENDING"),
                "plan_version": report.get("plan_version", "?"),
                "what_was_done": report.get("what_was_done", ""),
                "key_metrics": report.get("key_metrics", {}),
            }
            for report in reports
        ]
    if not slot_items:
        return f"Wave {wave_id}: no slot outcomes."
    promoted = sum(1 for item in slot_items if str(item.get("verdict", "PENDING")) == "PROMOTE")
    rejected = sum(1 for item in slot_items if str(item.get("verdict", "PENDING")) == "REJECT")
    planner_failures = sum(1 for item in slot_items if str(item.get("kind", "")) == "planner_failure")
    launch_failures = sum(1 for item in slot_items if str(item.get("kind", "")) == "launch_failure")
    open_slots = sum(1 for item in slot_items if str(item.get("kind", "")) == "open_slot")
    lines = [f"Wave {wave_id} summary ({len(slot_items)}/{target_size} slots accounted):"]
    for item in sorted(slot_items, key=lambda entry: int(entry.get("slot", 0) or 0)):
        metrics = item.get("key_metrics", {}) or {}
        metric_parts = [f"{k}={v}" for k, v in list(metrics.items())[:4]]
        line = f"- Slot {item.get('slot', '?')}: {item.get('kind', 'terminal_report')} | {item.get('status', '?')} -> {item.get('verdict', 'PENDING')}"
        if item.get("plan_version"):
            line += f" | plan_v{item.get('plan_version')}"
        if metric_parts:
            line += f" | {', '.join(metric_parts)}"
        done = str(item.get("what_was_done", "") or item.get("failure_detail", "") or "")[:140]
        if done:
            line += f" | {done}"
        lines.append(line)
    lines.append(
        f"Outcome: promoted={promoted}, rejected={rejected}, planner_failures={planner_failures}, launch_failures={launch_failures}, open_slots={open_slots}."
    )
    return "\n".join(lines)
