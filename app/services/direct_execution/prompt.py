"""
Prompt builder for direct slice execution.
"""

from __future__ import annotations

from typing import Any

from app.services.direct_execution.backtests_protocol import build_backtests_protocol_lines, is_backtests_context
from app.services.direct_execution.prompt_facts import compact_known_facts_for_prompt
from app.services.mcp_catalog.hints import build_tool_contract_hints
from app.services.mcp_catalog.models import McpCatalogSnapshot


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
    required_prerequisite_facts: list[str] | None = None,
    required_output_facts: list[str] | None = None,
    provider: str = "",
    catalog_snapshot: McpCatalogSnapshot | None = None,
) -> str:
    weak = _is_weak_provider(provider)
    strict_acceptance = _requires_strict_terminal_acceptance(slice_payload)
    lines: list[str] = []
    if worker_system_prompt.strip():
        lines.append(worker_system_prompt.strip())
        lines.append("")
    if weak:
        lines.extend(_build_weak_provider_header(max_tool_calls, max_expensive_tool_calls))
    else:
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
            ]
        )
    lines.append("")
    lines.extend(
        [
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
    contract_hints = (
        build_tool_contract_hints(
            snapshot=catalog_snapshot,
            allowed_tools=allowed_tools,
            known_facts=known_facts,
            provider=provider,
        )
        if catalog_snapshot is not None
        else []
    )
    if contract_hints:
        hint_limit = 3 if weak else len(contract_hints)
        lines.extend(["", "Tool contract hints:"])
        lines.extend(f"- {item}" for item in contract_hints[:hint_limit])
    if checkpoint_summary.strip():
        lines.extend(["", "Previous checkpoint:", checkpoint_summary.strip()])
    recent_limit = 3 if weak else 6
    if recent_turn_summaries:
        lines.extend(["", "Recent direct attempts:"])
        lines.extend(f"- {item}" for item in recent_turn_summaries[-recent_limit:])
    facts_limit = 12 if weak else 24
    compact_known_facts = compact_known_facts_for_prompt(
        known_facts=known_facts,
        dependency_ids=[str(item).strip() for item in slice_payload.get("depends_on", []) or [] if str(item).strip()],
        required_facts=[
            *[str(item).strip() for item in (required_prerequisite_facts or []) if str(item).strip()],
            *[str(item).strip() for item in (required_output_facts or []) if str(item).strip()],
        ],
        limit=facts_limit,
    )
    if compact_known_facts:
        lines.extend(["", "Known facts:"])
        for key, raw_value in compact_known_facts:
            value = str(raw_value)
            if len(value) > 260:
                value = value[:257] + "..."
            lines.append(f"- {key} = {value}")
    if _looks_like_research_setup_slice(allowed_tools):
        baseline_snapshot_id = str(baseline_bootstrap.get("baseline_snapshot_id", "active-signal-v1") or "active-signal-v1")
        baseline_version = baseline_bootstrap.get("baseline_version", 1)
        lines.extend(
            [
                "",
                "Research setup protocol:",
                (
                    "- Lock the baseline with "
                    f"research_project(action='set_baseline', snapshot_id='{baseline_snapshot_id}', version={baseline_version})."
                ),
                "- On a clean run, create or explicitly open the cycle project before any set_baseline / research_map / research_memory write.",
                "- Define atlas dimensions with research_map(action='define', ...).",
                "- Record invariants and naming_convention via research_memory(action='create', kind='note', record={content: {title: '...', text: '...'}, metadata: {invariants: '<your invariants>', naming_convention: '<your convention>'}}).",
                "  The metadata.invariants field sets research.invariants_recorded=True. The metadata.naming_convention field sets research.naming_recorded=True.",
                "- Verify research.baseline_configured=True, research.atlas_defined=True, research.invariants_recorded=True, research.naming_recorded=True before claiming completion.",
                "- Do not claim completion until all four are backed by tool evidence.",
            ]
        )
    if _looks_like_research_shortlist_slice(
        slice_payload=slice_payload,
        allowed_tools=allowed_tools,
        required_output_facts=required_output_facts or [],
    ):
        lines.extend(
            [
                "",
                "Research shortlist protocol:",
                "- Read/map calls do not complete this slice on their own.",
                "- Terminal step: persist one shortlist milestone with research_memory(action='create', kind='milestone', ...).",
                "- Canonical shortlist payload fields:",
                "- record.metadata.stage='hypothesis_formation'",
                "- record.metadata.outcome='shortlist_recorded'",
                "- record.metadata.shortlist_families=[...]",
                "- record.metadata.novelty_justification_present=true",
                "- record.content.candidates=[{family, why_new, relative_to}]",
                "- Use relative_to=['base', 'v1-v12'] when the novelty claim is against the base space and history v1-v12.",
                "- If you can describe the milestone payload or next_call_example, execute the milestone write instead of returning a checkpoint.",
                "- Do not return final_report before the shortlist milestone write succeeds and downstream facts are complete.",
                "- Once the shortlist milestone write succeeds and required facts are present, the terminal verdict MUST be COMPLETE.",
            ]
        )
    if _looks_like_mixed_domain_context_slice(allowed_tools=allowed_tools):
        lines.extend(
            [
                "",
                "Mixed-domain protocol:",
                "- `research_memory.search` is context recovery only. Do not repeat it more than twice in a row.",
                "- Once project or shortlist context is clear, switch to a non-research tool and gather live domain evidence.",
                "- After 2 successful non-research probes, either do one last targeted domain check or return final_report.",
                "- Do not bounce back to `research_memory.search` only to refine wording or restate the same context.",
            ]
        )
    if _looks_like_feature_contract_slice(
        slice_payload=slice_payload,
        allowed_tools=allowed_tools,
    ):
        tool_set = {str(item).strip() for item in allowed_tools if str(item).strip()}
        baseline_symbol = str(baseline_bootstrap.get("symbol", "BTCUSDT") or "BTCUSDT")
        anchor_timeframe = str(baseline_bootstrap.get("anchor_timeframe", "1h") or "1h")
        lines.extend(
            [
                "",
                "Feature contract protocol:",
                "- `research_memory.search` is context recovery only. Do not repeat it more than twice.",
                "- After shortlist context is clear, switch to a non-research tool and build contract evidence.",
                "- Mixed outcomes are allowed: some families may be VIABLE while others are DATA_BLOCKED.",
                "- This slice is COMPLETE when every shortlisted family is explicitly classified, leakage boundaries are checked, and each blocker is recorded as a finding rather than left ambiguous.",
                "- Do not use WATCHLIST just because some families are blocked; use WATCHLIST only when the contract exploration itself is still incomplete or ambiguous.",
                "- Do not guess `feature_name` for `features_analytics` or `name` for `features_custom` detail/source views.",
                "- If several candidate features remain after live discovery, stop and return final_report with the contract findings instead of inventing identifiers.",
            ]
        )
        if "features_catalog" in tool_set:
            lines.append(
                f"- Inspect managed feature coverage with features_catalog(scope='timeframe', timeframe='{anchor_timeframe}') or features_catalog(scope='available')."
            )
        if "events" in tool_set:
            lines.append("- Inspect normalized event coverage with events(view='catalog') before claiming event alignment.")
        if "datasets" in tool_set:
            lines.append("- Inspect local dataset coverage with datasets(view='catalog') or datasets(view='instruments').")
        if "features_dataset" in tool_set:
            lines.append(
                "- Inspect available anchor-timeframe columns with "
                f"features_dataset(action='inspect', view='columns', symbol='{baseline_symbol}', timeframe='{anchor_timeframe}')."
            )
        if "features_custom" in tool_set:
            lines.append("- Inspect publish/validate requirements with features_custom(action='inspect', view='contract').")
            lines.append("- `features_custom(action='inspect', view='detail'|'source')` requires one explicit `name`; inspect `view='list'` first when needed.")
        if "features_analytics" in tool_set:
            lines.append("- `features_analytics(action='analytics'|'heatmap'|'render'|'portability')` requires one explicit `feature_name`.")
        if tool_set & {"features_custom", "features_analytics", "models_dataset"}:
            lines.append(
                "- Use features_custom(validate/publish), features_analytics, or models_dataset only when they add concrete contract or leakage evidence."
            )
        lines.extend(
            [
                "- Read-only research search never satisfies data-contract or feature-contract evidence by itself.",
                "- Do not return final_report until the transcript includes non-research evidence for the contract work.",
            ]
        )
    lines.extend(
        build_backtests_protocol_lines(
            slice_payload=slice_payload,
            allowed_tools=allowed_tools,
            baseline_bootstrap=baseline_bootstrap,
        )
    )
    if _looks_like_backtests_analysis_slice(
        slice_payload=slice_payload,
        allowed_tools=allowed_tools,
    ):
        analysis_tools = [item for item in allowed_tools if item in {"backtests_conditions", "backtests_analysis", "backtests_runs"}]
        preferred_tool = analysis_tools[0] if analysis_tools else "backtests_analysis"
        has_conditions = "backtests_conditions" in {str(item).strip() for item in allowed_tools}
        conditions_lines = [
            "",
            "Backtests analysis protocol:",
            "- `research_memory.search` is context recovery only. Do not repeat it more than twice.",
            f"- Once candidate context is clear, switch immediately to `{preferred_tool}` and gather live stability/integration/cannibalization evidence.",
            "- Do not spend the tool budget on repeated shortlist wording searches.",
            "- For integration/cannibalization slices, use dependency-provided candidate handles first; do not rediscover them from transport metadata.",
            "- Return final_report only after the transcript includes live non-research backtest evidence.",
        ]
        if has_conditions:
            conditions_lines.extend(
                [
                    "- CRITICAL: `backtests_conditions(action='run')` requires an explicit `run_id` parameter. Get one from `backtests_runs(action='inspect', view='list')` → saved_runs[].run_id. Without run_id the server returns `selection_required` and no analysis is produced.",
                    "- Do NOT call `backtests_runs(action='start')` for stability/condition analysis. Use existing completed runs from the saved_runs list.",
                    "- If `backtests_conditions(action='list')` shows completed jobs for the target snapshot, read their results via `backtests_conditions(action='result', job_id=<id>)` before starting new condition runs.",
                ]
            )
        conditions_lines.extend(
            [
                "",
                "Layer compare failure recovery:",
                "- `backtests_analysis(action='start', analysis='layer_compare', ...)` can fail when baseline and candidate runs have incompatible execution profiles (different notional_usd, take_profit_pct, stop_loss_pct, etc.).",
                "- If layer_compare fails with a compatibility error, do NOT retry it. Instead: run diagnostics individually on each run using `backtests_runs(action='detail', run_id=...)` or `backtests_analysis(action='start', analysis='diagnostics', ...)` and report findings separately.",
                "- Report the incompatibility as a finding in your final_report, with verdict WATCHLIST. Include the specific execution profile differences in facts.",
                "- Never spend more than one attempt on layer_compare for a given pair of runs.",
            ]
        )
        lines.extend(conditions_lines)
    if _looks_like_model_training_slice(
        slice_payload=slice_payload,
        allowed_tools=allowed_tools,
    ):
        lines.extend(
            [
                "",
                "Model-training pipeline protocol:",
                "- This slice requires chaining ML pipeline tools in the correct order.",
                "- Pipeline order: models_dataset(preview) -> models_dataset(materialize) -> models_train(start) -> models_train(status) -> models_to_feature(validate) -> models_to_feature(publish).",
                "- Step 1: Prepare training data with models_dataset(action='preview', spec=...) to validate the spec, then models_dataset(action='materialize', spec=...) to persist it.",
                "- Step 2: Train the model with models_train(action='start', model_id=..., dataset_id=..., task_type=..., library=..., primary_metric=...). This returns a job_id.",
                "- Step 3: Poll completion with models_train(action='status', job_id=...). Wait for status='completed'.",
                "- Step 4: Validate feature integration with models_to_feature(action='validate', model_id=..., version=...).",
                "- Step 5: Publish the model-backed feature with models_to_feature(action='publish', model_id=..., version=...).",
                "- Use models_registry(action='inspect', view='list') to check existing model cards before creating duplicates.",
                "- Use models_compare(model_id=...) to compare model versions before promoting.",
                "- Do NOT skip steps. Do NOT call models_train without a materialized dataset. Do NOT call models_to_feature without a trained version.",
                "- Record the model_id, version, dataset_id, and resulting feature_name in final_report.facts.",
            ]
        )
    if required_prerequisite_facts:
        lines.extend(["", "Required prerequisite facts:"])
        lines.extend(
            f"- Expect `{item}` to already exist in dependency facts before using expensive tools."
            for item in required_prerequisite_facts
            if str(item).strip()
        )
    if required_output_facts:
        lines.extend(["", "Required downstream facts:"])
        if weak:
            lines.extend(f"- `{item}`" for item in required_output_facts)
            lines.append("- Include these facts in final_report.facts.")
        else:
            lines.extend(f"- Include `{item}` in final_report.facts or create tool evidence that deterministically yields it." for item in required_output_facts)
            lines.append("- Do not return final_report if required downstream facts are still missing.")
    missing_downstream = known_facts.get("direct.missing_downstream_prerequisites")
    if missing_downstream and isinstance(missing_downstream, list):
        lines.extend(["", "Downstream slices need these facts from you:"])
        lines.extend(f"- `{item}`" for item in missing_downstream if str(item).strip())
        lines.append("Your final_report.facts MUST include these before completion.")
    lines.extend(
        [
            "",
            "Contract footer:",
        ]
    )
    lines.extend(f"- {item}" for item in _model_contract_footer(provider=provider))
    if strict_acceptance:
        lines.extend(
            [
                "- This slice is acceptance-strict: downstream work will not unblock on WATCHLIST.",
                "- Use COMPLETE when the success criteria are proven by tool evidence, even if some findings are blocked, negative, or mixed.",
                "- Use WATCHLIST only when the slice itself remains incomplete, ambiguous, or unsupported by evidence.",
            ]
        )
    lines.extend(_build_confidence_and_evidence_instructions())
    if weak:
        lines.extend(
            [
                "",
                "CRITICAL RULES:",
                "1. ALWAYS return final_report, not checkpoint.",
                "2. NEVER use verdict INCOMPLETE, PARTIAL, or FAILED.",
                "3. Use WATCHLIST if unsure, COMPLETE if confident.",
            ]
        )
    lines.extend(
        [
            "",
            "Example final_report:",
            "```json",
            _example_final_report_json(strict_acceptance=strict_acceptance),
            "```",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _is_weak_provider(provider: str) -> bool:
    return str(provider or "").strip().lower() in {"lmstudio"}  # minimax is NOT weak


def _build_weak_provider_header(max_tool_calls: int, max_expensive_tool_calls: int) -> list[str]:
    return [
        "You are a direct execution worker for trading research.",
        "Use only the approved dev_space1 tools listed below.",
        "Return exactly one JSON object wrapped in ```json``` fences.",
        "The JSON must have type=final_report with: summary, verdict, findings, facts, evidence_refs, confidence.",
        "NEVER use verdict INCOMPLETE or PARTIAL. Use WATCHLIST if unsure, COMPLETE if confident.",
        f"Tool budget: {max_tool_calls} calls, {max_expensive_tool_calls} expensive.",
        "If a tool fails, fix the call per the error message.",
        "Do not restart known operations.",
        "CRITICAL: You MUST call at least one tool BEFORE returning any JSON.",
        "DO NOT return final_report until you have called a tool and received its result.",
        "Returning final_report with zero tool calls is a failure condition.",
        "Use evidence_refs from successful tool results only: transcript refs like transcript:1:tool_name or concrete ids like node_* and note_*.",
    ]


def _requires_strict_terminal_acceptance(slice_payload: dict[str, Any]) -> bool:
    unblock_mode = str(slice_payload.get("dependency_unblock_mode") or "").strip().lower()
    watchlist_ok = bool(slice_payload.get("watchlist_allows_unblock", False))
    return unblock_mode != "advisory_only" and not watchlist_ok


def _example_final_report_json(*, strict_acceptance: bool) -> str:
    verdict = "COMPLETE" if strict_acceptance else "WATCHLIST"
    return (
        '{"type":"final_report","summary":"...","verdict":"'
        + verdict
        + '","findings":[],"facts":{},"evidence_refs":[],"confidence":0.7}'
    )
def _looks_like_research_setup_slice(allowed_tools: list[str]) -> bool:
    tool_set = {str(item).strip() for item in allowed_tools if str(item).strip()}
    return {"research_project", "research_map", "research_memory"}.issubset(tool_set)


def _looks_like_research_shortlist_slice(
    *,
    slice_payload: dict[str, Any],
    allowed_tools: list[str],
    required_output_facts: list[str],
) -> bool:
    runtime_profile = str(slice_payload.get("runtime_profile") or "").strip()
    if runtime_profile == "research_shortlist":
        return True
    normalized_required = {str(item or "").strip() for item in required_output_facts if str(item or "").strip()}
    if "research.novelty_justification_present" in normalized_required:
        return True
    tool_set = {str(item).strip() for item in allowed_tools if str(item).strip()}
    if not ({"research_map", "research_project"} & tool_set):
        return False
    if not ({"research_memory", "research_record"} & tool_set):
        return False
    haystack = " ".join(
        str(item or "").strip().lower()
        for item in (
            slice_payload.get("title"),
            slice_payload.get("objective"),
            *(slice_payload.get("success_criteria") or []),
            *(slice_payload.get("policy_tags") or []),
        )
        if str(item or "").strip()
    )
    return any(marker in haystack for marker in ("shortlist", "short-list", "novelty", "first wave", "v1-v12"))


def _looks_like_feature_contract_slice(
    *,
    slice_payload: dict[str, Any],
    allowed_tools: list[str],
) -> bool:
    if not _looks_like_mixed_domain_context_slice(allowed_tools=allowed_tools):
        return False
    haystack = " ".join(
        str(item or "").strip().lower()
        for item in (
            slice_payload.get("title"),
            slice_payload.get("objective"),
            *(slice_payload.get("success_criteria") or []),
            *(slice_payload.get("policy_tags") or []),
        )
        if str(item or "").strip()
    )
    markers = (
        "feature contract",
        "data contract",
        "leakage",
        "custom feature",
        "validated and published",
        "event alignment",
        "feature_contract",
        "data_readiness",
    )
    return any(marker in haystack for marker in markers)


def _looks_like_mixed_domain_context_slice(*, allowed_tools: list[str]) -> bool:
    tool_set = {str(item).strip() for item in allowed_tools if str(item).strip()}
    has_research = "research_memory" in tool_set
    has_domain_contract_tool = bool(
        tool_set & {
            "features_catalog",
            "events",
            "datasets",
            "features_custom",
            "features_dataset",
            "features_analytics",
            "models_dataset",
        }
    )
    return has_research and has_domain_contract_tool


def _looks_like_model_training_slice(
    *,
    slice_payload: dict[str, Any],
    allowed_tools: list[str],
) -> bool:
    tool_set = {str(item).strip() for item in allowed_tools if str(item).strip()}
    model_pipeline_tools = {"models_dataset", "models_train", "models_to_feature"}
    if not (tool_set & model_pipeline_tools):
        return False
    haystack = " ".join(
        str(item or "").strip().lower()
        for item in (
            slice_payload.get("title"),
            slice_payload.get("objective"),
            *(slice_payload.get("success_criteria") or []),
            *(slice_payload.get("policy_tags") or []),
        )
        if str(item or "").strip()
    )
    return any(
        marker in haystack
        for marker in ("model", "train", "classification", "routing", "model-backed", "wave b")
    )


def _looks_like_backtests_slice(
    *,
    slice_payload: dict[str, Any],
    allowed_tools: list[str],
) -> bool:
    return is_backtests_context(
        runtime_profile=str(slice_payload.get("runtime_profile") or ""),
        title=str(slice_payload.get("title") or ""),
        objective=str(slice_payload.get("objective") or ""),
        success_criteria=list(slice_payload.get("success_criteria") or []),
        policy_tags=list(slice_payload.get("policy_tags") or []),
        allowed_tools=allowed_tools,
    )


def _looks_like_backtests_analysis_slice(
    *,
    slice_payload: dict[str, Any],
    allowed_tools: list[str],
) -> bool:
    tool_set = {str(item).strip() for item in allowed_tools if str(item).strip()}
    if "research_memory" not in tool_set:
        return False
    if not (tool_set & {"backtests_conditions", "backtests_analysis", "backtests_runs"}):
        return False
    haystack = " ".join(
        str(item or "").strip().lower()
        for item in (
            slice_payload.get("title"),
            slice_payload.get("objective"),
            *(slice_payload.get("success_criteria") or []),
            *(slice_payload.get("policy_tags") or []),
        )
        if str(item or "").strip()
    )
    markers = (
        "stability",
        "condition analysis",
        "integration",
        "cannibal",
        "ownership",
        "new-entry proof",
        "analysis",
    )
    return any(marker in haystack for marker in markers)


def _model_contract_footer(*, provider: str) -> list[str]:
    provider_name = str(provider or "").strip().lower()
    if provider_name in {"lmstudio", "minimax", "qwen_cli"}:
        return [
            "Success means `final_report` only.",
            "A `checkpoint` with `status=complete` is fallback-compatible only when evidence is complete and downstream facts are satisfied.",
            "A partial checkpoint triggers one bounded repair or fallback; it is not terminal success.",
        ]
    return [
        "Success means `final_report` only.",
        "Do not treat partial progress as terminal success.",
    ]


def _build_confidence_and_evidence_instructions() -> list[str]:
    """Instructions for confidence calibration and evidence_ref formatting.

    Models that are not classified as weak (MiniMax, Claude) still sometimes
    produce confidence=0.30 as a default. These instructions explicitly tell
    the model what confidence to use and what evidence_ref formats are valid.
    """
    return [
        "Confidence and evidence rules:",
        "- confidence MUST be >= 0.70 when your final_report is backed by real tool call results.",
        "- Use confidence=0.85-0.95 for clear-cut findings, 0.70-0.80 for mixed or partial evidence.",
        "- NEVER use confidence below 0.50 when you have made successful tool calls.",
        "- evidence_refs must use concrete identifiers from tool results. Valid formats:",
        "  * Transcript refs: 'transcript:N:tool_name' where N is the 1-based transcript index.",
        "  * Run ids: 'run_abc123' or concrete ids like '20260414-101141-61cb0d64'.",
        "  * Analysis ids: 'analysis-f97d51219322' or job ids like 'cond-f07199b451c1'.",
        "  * Snapshot refs: 'active-signal-v1@1' or 'snapshot_id@version'.",
        "  * Research nodes: 'node-e9ea1d9574354c5892a481f4805e1e2e' or 'note_*'.",
        "  * Project/branch ids: 'postmortem-v1-v2-invariants-ef8c5e4b', 'branch-xxx'.",
        "- Do NOT fabricate evidence_refs. Only use ids returned by actual tool responses.",
        "- If backtests_runs(action='start') is blocked, create research_memory(action='create', kind='result') with your findings BEFORE returning final_report. This provides a mutating tool call and a research node for acceptance.",
    ]


__all__ = ["build_direct_slice_prompt"]
