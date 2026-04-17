"""
Backtests-specific prompt guidance and local safety guards.
"""

from __future__ import annotations

import json
from typing import Any

from app.services.mcp_catalog.models import McpCatalogSnapshot

STRICT_BACKTESTS_PROFILES = frozenset(
    {
        "backtests_stability_analysis",
        "backtests_integration_analysis",
        "backtests_cannibalization_analysis",
    }
)

_BACKTESTS_CONTEXT_MARKERS = (
    "backtest",
    "standalone",
    "stability",
    "condition analysis",
    "integration",
    "cannibal",
    "ownership",
    "new-entry proof",
    "walk-forward",
    "walkforward",
    "forensic",
    "reconstruct",
    "reproduce",
    "reproducible",
    "review",
    "audit",
)


def is_backtests_context(
    *,
    runtime_profile: str = "",
    title: str = "",
    objective: str = "",
    success_criteria: list[str] | tuple[str, ...] | None = None,
    policy_tags: list[str] | tuple[str, ...] | None = None,
    allowed_tools: set[str] | list[str] | tuple[str, ...] | None = None,
) -> bool:
    profile = str(runtime_profile or "").strip()
    if profile.startswith("backtests_"):
        return True
    tool_set = {str(item).strip() for item in (allowed_tools or []) if str(item).strip()}
    if not (tool_set & {"backtests_plan", "backtests_runs", "backtests_conditions", "backtests_analysis", "backtests_walkforward", "backtests_strategy"}):
        return False
    haystack = " ".join(
        str(item or "").strip().lower()
        for item in (title, objective, *(success_criteria or []), *(policy_tags or []))
        if str(item or "").strip()
    )
    return any(marker in haystack for marker in _BACKTESTS_CONTEXT_MARKERS)


def augment_allowed_tools_for_backtests(
    *,
    allowed_tools: set[str],
    catalog_snapshot: McpCatalogSnapshot | None,
    runtime_profile: str = "",
    title: str = "",
    objective: str = "",
    success_criteria: list[str] | tuple[str, ...] | None = None,
    policy_tags: list[str] | tuple[str, ...] | None = None,
) -> set[str]:
    """Add cheap read-only tools when a backtests slice can use them.

    1. For any backtests context that already has ``backtests_runs``, add
       ``backtests_plan`` so the worker can do a cheap preflight before
       starting a run.

    2. For analysis-only profiles (stability, integration, cannibalization)
       that have analysis tools but lack ``backtests_runs``, add
       ``backtests_runs`` so the worker can discover saved runs and pass
       their run_ids to the analysis tools.
    """

    tools = {str(item).strip() for item in allowed_tools if str(item).strip()}
    if not is_backtests_context(
        runtime_profile=runtime_profile,
        title=title,
        objective=objective,
        success_criteria=success_criteria,
        policy_tags=policy_tags,
        allowed_tools=tools,
    ):
        return tools

    has_catalog = catalog_snapshot is not None

    # --- Branch A: already has backtests_runs -> add backtests_plan ----------
    if "backtests_runs" in tools and "backtests_plan" not in tools:
        if has_catalog and catalog_snapshot.has_tool_name("backtests_plan"):
            tools.add("backtests_plan")

    # --- Branch B: analysis profile without backtests_runs -> add it ---------
    profile = str(runtime_profile or "").strip()
    if (
        profile in STRICT_BACKTESTS_PROFILES
        and "backtests_runs" not in tools
        and (
            "backtests_analysis" in tools
            or "backtests_conditions" in tools
            or "backtests_studies" in tools
        )
    ):
        if has_catalog and catalog_snapshot.has_tool_name("backtests_runs"):
            tools.add("backtests_runs")

    return tools


def format_backtests_plan_call(*, baseline_bootstrap: dict[str, Any]) -> str:
    snapshot_id = str(
        baseline_bootstrap.get("baseline_snapshot_id")
        or baseline_bootstrap.get("snapshot_id")
        or "active-signal-v1"
    )
    version = baseline_bootstrap.get("baseline_version", 1)
    symbol = str(baseline_bootstrap.get("symbol", "BTCUSDT") or "BTCUSDT")
    anchor_timeframe = str(baseline_bootstrap.get("anchor_timeframe", "1h") or "1h")
    execution_timeframe = str(baseline_bootstrap.get("execution_timeframe", "5m") or "5m")
    return (
        "backtests_plan("
        f"snapshot_id='{snapshot_id}', version={version}, symbol='{symbol}', "
        f"anchor_timeframe='{anchor_timeframe}', execution_timeframe='{execution_timeframe}')"
    )


def _is_standalone_backtest_slice(slice_payload: dict[str, Any]) -> bool:
    """Detect slices that test new signals independently (not re-running baseline).

    Only matches when "standalone" is explicitly present in title/objective/criteria.
    Broader terms like "candidate" alone are not enough — integration slices also
    mention candidates but should use the standard protocol.
    """
    haystack = " ".join(
        str(item or "").strip().lower()
        for item in (
            slice_payload.get("title"),
            slice_payload.get("objective"),
            *(slice_payload.get("success_criteria") or []),
        )
        if str(item or "").strip()
    )
    return "standalone" in haystack


def build_backtests_protocol_lines(
    *,
    slice_payload: dict[str, Any],
    allowed_tools: list[str],
    baseline_bootstrap: dict[str, Any],
) -> list[str]:
    tool_set = {str(item).strip() for item in allowed_tools if str(item).strip()}
    if not is_backtests_context(
        runtime_profile=str(slice_payload.get("runtime_profile") or ""),
        title=str(slice_payload.get("title") or ""),
        objective=str(slice_payload.get("objective") or ""),
        success_criteria=list(slice_payload.get("success_criteria") or []),
        policy_tags=list(slice_payload.get("policy_tags") or []),
        allowed_tools=tool_set,
    ):
        return []
    lines = ["", "Backtests protocol:"]
    is_standalone = _is_standalone_backtest_slice(slice_payload)
    runtime_profile = str(slice_payload.get("runtime_profile") or "").strip()
    is_analysis_profile = runtime_profile in STRICT_BACKTESTS_PROFILES
    if "backtests_plan" in tool_set:
        plan_call = format_backtests_plan_call(baseline_bootstrap=baseline_bootstrap)
        if is_standalone:
            lines.extend(
                [
                    "- First, inspect existing snapshots via `backtests_strategy(action='inspect', view='list')` to find candidate-specific snapshots.",
                    f"- If candidate snapshots exist, plan and run backtests on those. If not, call {plan_call} to check baseline readiness.",
                    "- If the baseline already has completed runs and no new candidate snapshots can be created, create a `research_memory(action='create', kind='result')` entry recording this finding, then return COMPLETE (NOT WATCHLIST). Recording the finding in research_memory IS the deliverable.",
                ]
            )
        else:
            lines.extend(
                [
                    f"- First live action: call {plan_call}.",
                    "- Start `backtests_runs(action='start', ...)` only after a successful `backtests_plan(...)` or when reusing a known durable run_id.",
                ]
            )
    elif "backtests_runs" in tool_set:
        lines.extend(
            [
                "- First, list saved runs with `backtests_runs(action='inspect', view='list', scope='saved')` to discover completed run_ids.",
                "- Do NOT start new runs in an analysis slice; use only saved/completed run_ids.",
            ]
        )
    else:
        lines.extend(
            [
                "- Use only the approved backtests tools listed above; do not call unavailable readiness tools.",
                "- Do not start a new run unless a successful readiness/preflight result already exists in this attempt.",
            ]
        )
    lines.extend(
        [
            "- Use one targeted `research_memory.search` only if shortlist/project context is genuinely missing, then switch back to backtests immediately.",
            "- Do not loop on research_memory once the candidate/backtest context is clear.",
            "- After `backtests_runs(... view='list' ...)`, treat `active_runs` and `saved_runs` as the only valid `run_id` sources.",
            "- `request_id`, `correlation_id`, and `server_session_id` are transport metadata only. They are never `run_id` values.",
            "- If the run list has no reusable run_id, do not guess and do not switch to detail/status/events/trades.",
            "- If no reusable run_id or allowed readiness path exists, return a blocked checkpoint with the missing durable handle reason instead of starting a duplicate baseline run.",
            "- Do not return final_report until the transcript contains live backtest planning or run evidence.",
            (
                "- IMPORTANT: If backtests_runs(action='start') is blocked as a duplicate, "
                "pick a completed run_id from the saved_runs list and call "
                "backtests_conditions(action='run', run_id=<saved_run_id>) or "
                "backtests_analysis(action='start', run_id=<saved_run_id>) instead. "
                "Do NOT just create a research note and return without running analysis."
                if is_analysis_profile
                else "- IMPORTANT: If backtests_runs(action='start') is blocked as a duplicate, do NOT retry with the same arguments. Instead, create a research_memory(action='create', kind='result') note recording your findings, then return COMPLETE (NOT WATCHLIST). A valid research_memory node IS sufficient evidence even when no new runs were started."
            ),
            "- When returning evidence_refs, use real IDs from tool responses (snapshot@version, transcript:N:tool_name, node-ids, run-ids). NEVER fabricate refs like 'tool-response-empty'.",
        ]
    )
    if "backtests_analysis" in tool_set:
        lines.append(
            "- layer_compare requires compatible runs (matching execution profiles). "
            "If layer_compare fails with a compatibility error, fall back to individual diagnostics per run."
        )
    return lines


def build_backtests_first_action_guide(
    *,
    allowed_tools: set[str],
    baseline_bootstrap: dict[str, Any],
    first_criterion: str = "",
) -> str:
    if "backtests_plan" in allowed_tools:
        plan_call = format_backtests_plan_call(baseline_bootstrap=baseline_bootstrap)
        parts = [
            "PROTOCOL:",
            f"Step 1: Call {plan_call}.",
            "Step 2: Read the planning result and shortlist/run readiness details.",
            "Step 3: If needed, start or inspect backtests_runs using the planned inputs.",
            "Step 4: Return exactly one final_report JSON with evidence_refs referencing the live backtest results.",
        ]
        if first_criterion:
            parts.append(f"Prioritize evidence for: {first_criterion}.")
        parts.append(f"IMPORTANT: You MUST call {plan_call} or another allowed backtests tool before returning text.")
        return " ".join(parts)
    preferred = "backtests_runs" if "backtests_runs" in allowed_tools else sorted(allowed_tools)[0]
    parts = [
        "PROTOCOL:",
        f"Step 1: Use {preferred} only within the approved tool contract.",
        "Step 2: If listing runs, use run_id only from active_runs or saved_runs.",
        "Step 3: Do not guess run_id from request_id, correlation_id, or server_session_id.",
        "Step 4: If no durable run_id is available, return a blocked checkpoint instead of starting a duplicate baseline run.",
    ]
    if first_criterion:
        parts.append(f"Prioritize evidence for: {first_criterion}.")
    return " ".join(parts)


def build_backtests_zero_tool_nudge(*, allowed_tools: set[str], baseline_bootstrap: dict[str, Any]) -> str:
    if "backtests_plan" in allowed_tools:
        return (
            "You did not call any tool. "
            f"You MUST call {format_backtests_plan_call(baseline_bootstrap=baseline_bootstrap)} now, unless you already know a reusable run_id. "
            "If shortlist context is missing, do at most one targeted research_memory.search and then immediately call a backtests tool. "
            "Call a tool first, then return your final_report JSON."
        )
    tools = ", ".join(sorted(allowed_tools))
    return (
        "You did not call any tool. Use only approved tools "
        f"[{tools}]. If no durable run_id is available, return a blocked checkpoint; "
        "do not guess from transport ids and do not start a duplicate baseline run."
    )


def backtests_start_guard_payload(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    transcript: list[dict[str, Any]] | None,
    runtime_profile: str,
    baseline_bootstrap: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if str(tool_name or "").strip() != "backtests_runs":
        return None
    if str(arguments.get("action") or "").strip().lower() != "start":
        return None
    transcript_items = list(transcript or [])
    is_analysis = str(runtime_profile or "").strip() in STRICT_BACKTESTS_PROFILES
    if _has_prior_handle_misuse(transcript_items):
        return _start_guard_error("backtests_start_blocked_after_handle_misuse", arguments, is_analysis=is_analysis)
    if _has_duplicate_saved_run(transcript_items, arguments):
        saved_ids = _saved_run_ids_from_transcript(transcript_items, arguments)
        return _start_guard_error("duplicate_baseline_start_blocked", arguments, is_analysis=is_analysis, saved_run_ids=saved_ids)
    if _is_baseline_rerun_in_strict_profile(arguments=arguments, runtime_profile=runtime_profile, baseline_bootstrap=baseline_bootstrap or {}):
        return _start_guard_error("duplicate_baseline_start_blocked", arguments, is_analysis=is_analysis)
    if not _has_successful_backtests_plan(transcript_items):
        return _start_guard_error("backtests_plan_required_before_start", arguments, is_analysis=is_analysis)
    return None


def _start_guard_error(
    reason_code: str,
    arguments: dict[str, Any],
    *,
    is_analysis: bool = False,
    saved_run_ids: list[str] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "ok": False,
        "error_class": "agent_contract_misuse",
        "summary": "backtests_runs(action='start') blocked by local backtests start guard.",
        "details": {
            "tool_name": "backtests_runs",
            "reason_code": reason_code,
            "arguments": dict(arguments or {}),
        },
    }
    if is_analysis and reason_code == "duplicate_baseline_start_blocked":
        ids_hint = ""
        if saved_run_ids:
            sample = saved_run_ids[:3]
            ids_hint = f" Available saved run_ids (first 3): {sample}."
        error["remediation"] = (
            "This is an analysis slice — do NOT start new runs. "
            "Use an existing saved run_id with backtests_conditions(action='run', run_id=<saved_run_id>) "
            "or backtests_analysis(action='start', run_id=<saved_run_id>)."
            + ids_hint
        )
    return error


def _saved_run_ids_from_transcript(transcript: list[dict[str, Any]], arguments: dict[str, Any]) -> list[str]:
    """Extract saved run_ids matching the target snapshot/symbol from prior list calls."""
    target_symbol = str(arguments.get("symbol") or "").strip()
    target_snapshot = str(arguments.get("snapshot_id") or arguments.get("strategy_snapshot_id") or "").strip()
    target_version = str(arguments.get("version") or arguments.get("snapshot_version") or "").strip()
    if not target_symbol or not target_snapshot:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for item in transcript:
        if item.get("kind") != "tool_result" or str(item.get("tool") or "") != "backtests_runs":
            continue
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        if str(args.get("view") or "").strip().lower() != "list":
            continue
        data = _structured_data(item.get("payload"))
        saved_runs = data.get("saved_runs") if isinstance(data.get("saved_runs"), list) else []
        for run in saved_runs:
            if not isinstance(run, dict):
                continue
            if str(run.get("status") or "").strip().lower() != "completed":
                continue
            if str(run.get("symbol") or "").strip() != target_symbol:
                continue
            if str(run.get("strategy_snapshot_id") or "").strip() != target_snapshot:
                continue
            version = str(run.get("strategy_snapshot_version") or "").strip()
            if target_version and version and target_version != version:
                continue
            run_id = str(run.get("run_id") or "").strip()
            if run_id and run_id not in seen:
                seen.add(run_id)
                ids.append(run_id)
    return ids


def _has_successful_backtests_plan(transcript: list[dict[str, Any]]) -> bool:
    for item in transcript:
        if item.get("kind") != "tool_result" or str(item.get("tool") or "") != "backtests_plan":
            continue
        if _is_success_payload(item.get("payload")):
            return True
    return False


def _has_prior_handle_misuse(transcript: list[dict[str, Any]]) -> bool:
    for item in transcript:
        if item.get("kind") != "tool_result":
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        if str(payload.get("error_class") or "") != "agent_contract_misuse":
            continue
        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        reason = str(details.get("reason_code") or "").strip()
        json_path = str(details.get("json_path") or "").strip()
        if reason == "suspicious_durable_handle" or json_path == "run_id":
            return True
    return False


def _has_duplicate_saved_run(transcript: list[dict[str, Any]], arguments: dict[str, Any]) -> bool:
    target_symbol = str(arguments.get("symbol") or "").strip()
    target_snapshot = str(arguments.get("snapshot_id") or arguments.get("strategy_snapshot_id") or "").strip()
    target_version = str(arguments.get("version") or arguments.get("snapshot_version") or arguments.get("strategy_snapshot_version") or "").strip()
    if not target_symbol or not target_snapshot:
        return False
    for item in transcript:
        if item.get("kind") != "tool_result" or str(item.get("tool") or "") != "backtests_runs":
            continue
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        if str(args.get("view") or "").strip().lower() != "list":
            continue
        data = _structured_data(item.get("payload"))
        saved_runs = data.get("saved_runs") if isinstance(data.get("saved_runs"), list) else []
        for run in saved_runs:
            if not isinstance(run, dict):
                continue
            if str(run.get("status") or "").strip().lower() != "completed":
                continue
            if str(run.get("symbol") or "").strip() != target_symbol:
                continue
            if str(run.get("strategy_snapshot_id") or "").strip() != target_snapshot:
                continue
            version = str(run.get("strategy_snapshot_version") or "").strip()
            if target_version and version and target_version != version:
                continue
            return True
    return False


def _is_baseline_rerun_in_strict_profile(
    *,
    arguments: dict[str, Any],
    runtime_profile: str,
    baseline_bootstrap: dict[str, Any],
) -> bool:
    if str(runtime_profile or "").strip() not in {"backtests_integration_analysis", "backtests_cannibalization_analysis"}:
        return False
    snapshot = str(arguments.get("snapshot_id") or "").strip()
    version = str(arguments.get("version") or "").strip()
    symbol = str(arguments.get("symbol") or "").strip()
    baseline_snapshot = str(baseline_bootstrap.get("baseline_snapshot_id") or baseline_bootstrap.get("snapshot_id") or "active-signal-v1").strip()
    baseline_version = str(baseline_bootstrap.get("baseline_version") or baseline_bootstrap.get("version") or 1).strip()
    baseline_symbol = str(baseline_bootstrap.get("symbol") or "BTCUSDT").strip()
    return snapshot == baseline_snapshot and version == baseline_version and symbol == baseline_symbol


def _structured_data(payload: Any) -> dict[str, Any]:
    structured = _structured_payload(payload)
    data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
    return data if isinstance(data, dict) else {}


def _structured_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    structured = nested.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    content = nested.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or str(item.get("type") or "") != "text":
                continue
            try:
                parsed = json.loads(str(item.get("text") or ""))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _is_success_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("error") or payload.get("ok") is False:
        return False
    structured = _structured_payload(payload)
    status = str(structured.get("status") or "").strip().lower()
    return status not in {"error", "failed"}


def coerce_analysis_start_to_existing_run(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    transcript: list[dict[str, Any]],
    runtime_profile: str,
    allowed_tools: set[str] | list[str] | tuple[str, ...],
) -> tuple[str, dict[str, Any], str] | None:
    """Coerce a duplicate backtests_runs(start) on an analysis profile to use a saved run.

    When the agent is in an analysis profile (stability, integration, cannibalization)
    and tries to start a new run against a snapshot that already has completed saved runs,
    automatically redirect the call to the appropriate analysis tool with the saved run_id.

    Returns (new_tool_name, new_arguments, repair_note) or None if coercion doesn't apply.
    """
    if str(tool_name or "").strip() != "backtests_runs":
        return None
    if str(arguments.get("action") or "").strip().lower() != "start":
        return None
    if str(runtime_profile or "").strip() not in STRICT_BACKTESTS_PROFILES:
        return None
    saved_ids = _saved_run_ids_from_transcript(transcript, arguments)
    if not saved_ids:
        return None
    tool_set = {str(t).strip() for t in allowed_tools if str(t).strip()}
    target_run_id = saved_ids[0]
    repair_note = (
        f"Coerced backtests_runs(action='start') to use existing saved run_id={target_run_id} "
        f"(analysis profile={runtime_profile}). The agent should analyze existing runs, not start duplicates."
    )
    if "backtests_conditions" in tool_set:
        new_args: dict[str, Any] = {
            "action": "run",
            "run_id": target_run_id,
        }
        for key in ("symbol", "anchor_timeframe", "execution_timeframe", "snapshot_id", "version", "project_id"):
            if key in arguments:
                new_args[key] = arguments[key]
        return ("backtests_conditions", new_args, repair_note)
    if "backtests_analysis" in tool_set:
        new_args = {
            "action": "start",
            "run_id": target_run_id,
        }
        for key in ("project_id",):
            if key in arguments:
                new_args[key] = arguments[key]
        return ("backtests_analysis", new_args, repair_note)
    return None


__all__ = [
    "STRICT_BACKTESTS_PROFILES",
    "augment_allowed_tools_for_backtests",
    "backtests_start_guard_payload",
    "build_backtests_first_action_guide",
    "build_backtests_protocol_lines",
    "build_backtests_zero_tool_nudge",
    "coerce_analysis_start_to_existing_run",
    "format_backtests_plan_call",
    "is_backtests_context",
]
