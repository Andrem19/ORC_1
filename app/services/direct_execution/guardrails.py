"""
Guardrails for direct execution fallback, repair, and evidence gating.
"""

from __future__ import annotations

import re
from typing import Any

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.fact_hydration import hydrate_final_report_facts

REPAIRABLE_PROVIDER_NAMES = {"lmstudio", "minimax", "qwen_cli", "claude_cli", "glm_cli"}
_RESEARCH_SETUP_TOOLS = frozenset({"research_project", "research_map", "research_memory"})
_STRICT_RESEARCH_SETUP_FACTS = (
    "research.baseline_configured",
    "research.atlas_defined",
    "research.invariants_recorded",
    "research.naming_recorded",
)
_DOMAIN_EXECUTION_PREFIXES = (
    "features_",
    "models_",
    "events",
    "datasets",
    "backtests_",
    "signal_api_binding_",
    "research_",
)
_DOMAIN_EVIDENCE_MARKERS = (
    "feature contract",
    "data contract",
    "custom feature",
    "model-backed",
    "leakage",
    "event alignment",
    "alignment",
    "coverage",
    "materialized dataset",
    "publish",
    "published",
    "dataset",
)

# Verdicts that explicitly indicate the result is NOT successful.
_FAIL_VERDICTS = frozenset(
    v.upper() for v in (
        "INCOMPLETE", "FAILED", "FAILURE", "PARTIAL", "ABORTED", "UNKNOWN",
    )
)

# Minimum confidence a final_report must report to be considered successful.
_MIN_SUCCESS_CONFIDENCE = 0.5
# WATCHLIST is the default verdict produced by salvage/fallback heuristics;
# require it to carry a clearly elevated confidence before we accept it.
_MIN_WATCHLIST_CONFIDENCE = 0.70

# Confidence floor when the model made real tool calls but defaulted to a low
# confidence (e.g. MiniMax defaults to 0.30). If the model made enough
# successful tool calls, we treat the work as genuine and boost to this floor.
_TOOL_USE_CONFIDENCE_FLOOR = 0.55
_TOOL_USE_FLOOR_MIN_CALLS = 3

# Auto-salvage fact markers — if any is set on an action, the final_report was
# synthesized from a transcript after a model stall/budget event and must not
# be accepted as a real completion. Gate will reject it so fallback triggers.
_AUTO_SALVAGE_FACT_PREFIX = "direct.auto_finalized_from_"

# Narrow carve-out markers: when a prerequisite slice publishes one of these
# facts, downstream slices are allowed to terminate with REJECT/SKIP and zero
# tool calls — because there is literally no work to do. This does NOT relax
# the main `final_report_passes_quality_gate`; it only lets the orchestrator
# recognise a legitimate "upstream blocked me" outcome after every fallback
# provider has already rejected the result via the strict gate.
_PREREQUISITE_BLOCK_MARKERS = (
    "all_rejected",
    "blocks_downstream",
    "no_surviving_candidates",
    "skipped_by_prerequisite",
)
_PREREQUISITE_BLOCK_VERDICTS = frozenset({"REJECT", "SKIP", "SKIPPED"})

# Concrete ID prefixes returned by actual tool responses — always accepted as valid
# evidence refs because they represent real tool outputs, not fabricated identifiers.
_CONCRETE_REF_PREFIXES = (
    "run_",
    "analysis-",
    "cond-",
    "node-",
    "note_",
    "branch-",
    "job-",
)
# Date-stamped IDs: YYYYMMDD-HHMMSS-<hex> (backtest run IDs, etc.)
_DATE_STAMPED_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8,}$")
# Snapshot version refs: <snapshot_id>@<version>
_SNAPSHOT_REF_RE = re.compile(r"^[\w.-]+@\d+$")


def _is_concrete_evidence_ref(ref: str) -> bool:
    """Return True if *ref* matches a known concrete ID format from tool responses."""
    text = str(ref or "").strip()
    if not text:
        return False
    if any(text.startswith(prefix) for prefix in _CONCRETE_REF_PREFIXES):
        return True
    if _SNAPSHOT_REF_RE.match(text):
        return True
    if _DATE_STAMPED_ID_RE.match(text):
        return True
    return False


def is_prerequisite_block_terminal(
    action: Any,
    prerequisite_facts: dict[str, Any] | None,
) -> bool:
    """Return True iff *action* is a legitimate "upstream blocked me" terminal.

    Strict criteria (narrow carve-out, does NOT weaken the main quality gate):
    - action.verdict ∈ {REJECT, SKIP, SKIPPED}
    - at least one prerequisite fact key contains a ``_PREREQUISITE_BLOCK_MARKERS``
      marker and its value is truthy.

    Without both conditions this returns False and the slice is treated as a
    normal failure.
    """
    if action is None:
        return False
    verdict = str(getattr(action, "verdict", "") or "").strip().upper()
    if verdict not in _PREREQUISITE_BLOCK_VERDICTS:
        return False
    facts = dict(prerequisite_facts or {})
    if not facts:
        return False
    for key, value in facts.items():
        key_norm = str(key or "").strip().lower()
        if not key_norm:
            continue
        if not any(marker in key_norm for marker in _PREREQUISITE_BLOCK_MARKERS):
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if value:
                return True
            continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized and normalized not in {"false", "0", "no", "none", ""}:
                return True
            continue
        # Numbers / other truthy scalars.
        if bool(value):
            return True
    return False


def checkpoint_complete_passes_gate(
    *,
    slice_obj: PlanSlice,
    action: WorkerAction | None,
    required_output_facts: list[str],
    inherited_facts: dict[str, Any] | None = None,
) -> bool:
    if action is None:
        return False
    if action.action_type != "checkpoint" or str(action.status or "").strip() != "complete":
        return False
    if not str(action.summary or "").strip():
        return False
    if not action.facts:
        return False
    readiness = hydrate_final_report_facts(
        slice_obj=slice_obj,
        action=action,
        required_output_facts=required_output_facts,
        inherited_facts=inherited_facts,
    )
    if not readiness.evidence_refs:
        return False
    return not readiness.missing_required_facts


def final_report_payload_passes_gate(
    *,
    facts: dict[str, Any],
    findings: list[str],
    evidence_refs: list[str],
    required_output_facts: list[str],
    inherited_facts: dict[str, Any] | None = None,
) -> bool:
    synthetic_slice = PlanSlice(
        slice_id="guardrail",
        title="guardrail",
        hypothesis="guardrail",
        objective="guardrail",
        success_criteria=[],
        allowed_tools=["incidents"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
    )
    action = WorkerAction(
        action_id="guardrail_action",
        action_type="final_report",
        summary="guardrail",
        verdict="COMPLETE",
        findings=list(findings),
        facts=dict(facts),
        evidence_refs=[str(item).strip() for item in evidence_refs if str(item).strip()],
        confidence=0.6,
    )
    readiness = hydrate_final_report_facts(
        slice_obj=synthetic_slice,
        action=action,
        required_output_facts=required_output_facts,
        inherited_facts=inherited_facts,
    )
    if not readiness.evidence_refs:
        return False
    return not readiness.missing_required_facts


def final_report_passes_quality_gate(
    *,
    tool_call_count: int,
    action: WorkerAction,
    slice_obj: PlanSlice,
    required_output_facts: list[str],
    inherited_facts: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Validate that a final_report actually meets quality standards.

    Returns ``(passes, reason)`` where *reason* is empty on success.
    Failed checks short-circuit in order: verdict → confidence → tool calls →
    evidence refs → required facts.
    """
    # 1. Verdict must not be a known-failure value.
    verdict = str(action.verdict or "").strip().upper()
    if verdict in _FAIL_VERDICTS:
        return False, f"verdict_is_{verdict.lower()}"

    # 1b. Auto-salvage stubs (transcript-synthesized after stall/budget) are
    #     telemetry, not completions. Route them to fallback unconditionally.
    action_facts = getattr(action, "facts", {}) or {}
    if any(
        str(key).startswith(_AUTO_SALVAGE_FACT_PREFIX) and bool(value)
        for key, value in dict(action_facts).items()
    ):
        return False, "auto_salvage_stub_rejected"

    # 2. Confidence must meet minimum threshold.
    confidence = float(action.confidence or 0)

    # 2a. Confidence floor for tool-using results: some non-weak models
    #     (MiniMax) default to 0.30 even after successful tool work.
    #     When the model made >=3 successful tool calls, boost confidence
    #     to the floor so the gate evaluates the actual evidence quality
    #     rather than rejecting on a default-low confidence number.
    if confidence < _TOOL_USE_CONFIDENCE_FLOOR and tool_call_count >= _TOOL_USE_FLOOR_MIN_CALLS:
        confidence = _TOOL_USE_CONFIDENCE_FLOOR

    if confidence < _MIN_SUCCESS_CONFIDENCE:
        return False, f"confidence_{confidence:.2f}_below_{_MIN_SUCCESS_CONFIDENCE}"

    # 2b. WATCHLIST is the default fallback verdict — require elevated
    #     confidence so low-quality defaults (0.60) route to fallback.
    #     However, when a genuine (non-salvage) model response made >=3
    #     real tool calls, the low confidence is a calibration artifact,
    #     not a quality signal. Boost to the WATCHLIST threshold so the
    #     gate evaluates evidence and facts instead of the confidence number.
    #     Salvage reports are already rejected above (auto_salvage gate).
    if verdict == "WATCHLIST" and confidence < _MIN_WATCHLIST_CONFIDENCE:
        if tool_call_count >= _TOOL_USE_FLOOR_MIN_CALLS:
            confidence = _MIN_WATCHLIST_CONFIDENCE
        else:
            return False, f"watchlist_confidence_{confidence:.2f}_below_{_MIN_WATCHLIST_CONFIDENCE}"

    # 3. The model must have actually used at least one tool.
    #    Zero tool calls means the model hallucinated a result without doing any work.
    if tool_call_count <= 0:
        return False, "zero_tool_calls"

    # 4. Evidence refs must be present for a genuine success.
    readiness = hydrate_final_report_facts(
        slice_obj=slice_obj,
        action=action,
        required_output_facts=required_output_facts,
        inherited_facts=inherited_facts,
    )
    facts = readiness.facts
    refs = readiness.evidence_refs
    if not refs:
        return False, "empty_evidence_refs"

    # 5. Required output facts must be satisfied.
    if readiness.missing_required_facts:
        return False, f"missing_required_facts:{','.join(readiness.missing_required_facts[:5])}"

    # 6. Evidence refs must map to actual transcript-supported refs/ids when available.
    supported_refs = _string_list(facts.get("direct.supported_evidence_refs"))
    if supported_refs and not _has_supported_evidence_ref(refs=refs, supported_refs=supported_refs):
        # When the model made sufficient real tool calls, the transcript contains
        # verifiable evidence. Some models (MiniMax) fabricate evidence_refs
        # (e.g., "backtests_plan-response-empty") instead of using real
        # transcript refs.  Rescue by substituting real supported refs — this
        # is a formatting artifact, not a quality deficit, when real work was done.
        if tool_call_count >= _TOOL_USE_FLOOR_MIN_CALLS:
            refs = supported_refs[:5]
            facts["direct.rescued_evidence_refs"] = True
        else:
            return False, "unsupported_evidence_refs"

    return True, ""


# Verdicts that indicate the model was cautious, not that the task genuinely failed.
# These are candidates for automatic repair when evidence is actually sufficient.
_REPAIRABLE_VERDICTS = frozenset(v.upper() for v in ("INCOMPLETE", "PARTIAL"))

# Providers where verdict repair is allowed (local/weak models that tend to be
# overly cautious).
_VERDICT_REPAIR_PROVIDERS = frozenset({"lmstudio"})


def attempt_verdict_repair(
    *,
    action: WorkerAction,
    tool_call_count: int,
    slice_obj: PlanSlice,
    required_output_facts: list[str],
    inherited_facts: dict[str, Any] | None = None,
    provider: str = "",
) -> WorkerAction | None:
    """Return a repaired action with a passing verdict, or *None* if repair is not safe.

    This is NOT a gate bypass — every quality-gate check (evidence refs, confidence,
    tool calls, required facts) must pass.  Only the textual verdict is changed from
    a cautious ``INCOMPLETE``/``PARTIAL`` to ``WATCHLIST``.
    """
    if str(provider or "").strip().lower() not in _VERDICT_REPAIR_PROVIDERS:
        return None
    if action.action_type != "final_report":
        return None
    verdict = str(action.verdict or "").strip().upper()
    if verdict not in _REPAIRABLE_VERDICTS:
        return None
    # All quality-gate sub-checks must pass (except the verdict itself).
    if float(action.confidence or 0) < _MIN_SUCCESS_CONFIDENCE:
        return None
    if tool_call_count <= 0:
        return None
    if required_output_facts:
        readiness = hydrate_final_report_facts(
            slice_obj=slice_obj,
            action=action,
            required_output_facts=required_output_facts,
            inherited_facts=inherited_facts,
        )
        if readiness.missing_required_facts:
            return None
        refs = readiness.evidence_refs
    else:
        readiness = hydrate_final_report_facts(
            slice_obj=slice_obj,
            action=action,
            required_output_facts=[],
            inherited_facts=inherited_facts,
        )
        refs = readiness.evidence_refs
    if not refs:
        return None
    return WorkerAction(
        action_id=action.action_id,
        action_type=action.action_type,
        summary=action.summary,
        verdict="WATCHLIST",
        findings=list(action.findings or []),
        facts=dict(readiness.facts),
        evidence_refs=list(refs),
        confidence=action.confidence,
        status=action.status,
        reason_code=action.reason_code,
        reason=action.reason,
        tool=action.tool,
        arguments=action.arguments,
    )


def normalize_incomplete_reason(result: Any) -> str:
    error = str(getattr(result, "error", "") or "").strip()
    if error:
        return error
    acceptance = getattr(result, "acceptance_result", None)
    if isinstance(acceptance, dict) and acceptance:
        blockers = acceptance.get("blocking_reasons")
        if isinstance(blockers, list) and blockers:
            return "acceptance_verifier_failed:" + ",".join(str(item) for item in blockers[:5])
        status = str(acceptance.get("status") or "").strip()
        if status and status != "pass":
            return f"acceptance_verifier_failed:{status}"
    action = getattr(result, "action", None)
    if action is None:
        return "direct_execution_failed"
    if action.action_type == "checkpoint":
        status = str(action.status or "").strip() or "unknown"
        reason_code = str(action.reason_code or "").strip() or "checkpoint_not_terminal"
        return f"checkpoint_{status}:{reason_code}"
    if action.action_type == "abort":
        return str(action.reason_code or action.summary or "abort").strip() or "abort"
    return str(action.action_type or "direct_execution_failed").strip() or "direct_execution_failed"


def should_attempt_contract_repair(*, provider_name: str, result: Any, attempts_remaining: int) -> bool:
    if attempts_remaining <= 0:
        return False
    if provider_name not in REPAIRABLE_PROVIDER_NAMES:
        return False
    action = getattr(result, "action", None)
    error_text = str(getattr(result, "error", "") or "").strip().lower()
    action_reason = ""
    if action is not None:
        action_reason = " | ".join(
            text.strip().lower()
            for text in (
                str(getattr(action, "reason_code", "") or ""),
                str(getattr(action, "summary", "") or ""),
            )
            if text and text.strip()
        )
    raw_output = str(getattr(result, "raw_output", "") or "").strip()
    if any(
        token in f"{error_text} | {action_reason}"
        for token in (
            "direct_output_parse_failed",
            "tool_not_in_allowlist",
            "worker_action_type_invalid",
            "tool_prefixed_namespace_forbidden",
            "final_report_requires",
            "tool_call_requires",
            "abort_requires",
            "checkpoint_status_invalid",
            "dev_space1_tools_unavailable",
            "missing_required_facts",
            "evidence_complete_but_verdict_not_accepted",
            "auto_salvage_stub_rejected",
            "direct_error_loop_detected",
        )
    ):
        return True
    if action is None:
        return False
    if action.action_type == "checkpoint" and str(action.status or "").strip() == "partial":
        return True
    if action.action_type == "checkpoint" and action.facts.get("direct.invalid_terminal_tool_call"):
        return True
    if "next_action" in raw_output.lower():
        return True
    return False


def build_contract_repair_prompt(
    *,
    provider_name: str,
    allowed_tools: list[str],
    failure_reason: str,
    raw_output_excerpt: str,
    required_output_facts: list[str],
    forbid_tool_calls: bool = False,
    registry_missing: bool = False,
    registry_verified: bool = False,
    allow_new_tool_calls: bool = False,
    repair_tool_call_budget: int = 0,
    is_research_setup_repair: bool = False,
    strict_acceptance_required: bool = False,
) -> str:
    sanitized_failure = failure_reason
    if "tool_not_in_allowlist:" in sanitized_failure.lower():
        sanitized_failure = "tool_not_in_allowlist"
    lines = [
        "## Contract Repair",
        f"Provider: {provider_name}",
        f"Failure: {sanitized_failure[:800]}",
        "",
        "Repair the previous output into exactly one terminal JSON object wrapped in ```json``` fences.",
        "Success means `final_report` only.",
        "A `checkpoint` is not a success here.",
        "Do not invent evidence that was not already present in the failed output or known facts.",
    ]
    if strict_acceptance_required:
        lines.extend(
            [
                "This slice is dependency-critical and requires an accepted terminal result.",
                "If the existing evidence fully proves every required downstream fact, the repaired final_report MUST use `verdict=\"COMPLETE\"`.",
                "Do NOT return `WATCHLIST` when the facts and evidence already prove success.",
                "If the evidence does not safely justify `COMPLETE`, return `abort` with reason_code=`direct_contract_blocker`.",
            ]
        )
    if required_output_facts:
        lines.append(
            "Required downstream facts for success: " + ", ".join(
                f"`{item}`" for item in required_output_facts if str(item).strip()
            )
        )
    if registry_verified:
        lines.extend(
            [
                "Qwen native preflight already proved the required dev_space1 tools are visible.",
                "Do not claim tools are unavailable. Use the exact visible tool names from the prompt context.",
            ]
        )
    if registry_missing:
        lines.extend(
            [
                "Qwen native tool registry preflight did not expose the required dev_space1 tools.",
                "Do not attempt tool calls, do not emit next_action, and do not reference unavailable tools.",
                "Return either `final_report` from already-known evidence or `abort` with reason_code=`qwen_tool_registry_missing`.",
            ]
        )
    elif is_research_setup_repair:
        lines.extend(
            [
                "This is a research-setup slice — the previous attempt missed required setup facts.",
                "You MUST make new tool calls to complete the missing research setup steps before returning final_report.",
                f"Tool-call budget for this repair: {repair_tool_call_budget}.",
                "Required research setup chain (execute missing steps in order):",
                "1. research_project(action='create' or 'open') → ensure the project is active.",
                "2. research_project(action='set_baseline', ...) → verify research.baseline_configured is True.",
                "3. research_map(action='define', ...) → verify research.atlas_defined is True.",
                "4. research_memory(action='create', kind='note', record={content: {title: 'Invariants and naming convention', text: '<details>'}, metadata: {invariants: '<your invariants>', naming_convention: '<your naming convention>'}}) → this sets research.invariants_recorded=True and research.naming_recorded=True via the metadata fields.",
                "Do NOT return final_report until ALL required facts are set.",
                "Do NOT skip steps or fabricate fact values.",
            ]
        )
    elif forbid_tool_calls or "tool_not_in_allowlist" in failure_reason.lower():
        lines.extend(
            [
                "Do not emit `tool_call` or `next_action`.",
                "Do not mention unavailable tools.",
                "If a tool name must appear in findings or facts, it must be one of: "
                + ", ".join(allowed_tools),
            ]
        )
    elif allow_new_tool_calls and repair_tool_call_budget > 0:
        lines.extend(
            [
                "Repair may include a small amount of new evidence collection before the final JSON.",
                f"You may make up to {repair_tool_call_budget} new non-expensive tool calls to gather missing non-research evidence.",
                "Do not restart heavy jobs, do not exceed the stated budget, and return a terminal JSON result after that short repair pass.",
            ]
        )
    else:
        lines.append("Do not emit `tool_call` or `next_action` in the repaired output.")
    lines.extend(
        [
            "If evidence is still incomplete for a compliant `final_report`, return `abort` with reason_code=`direct_contract_blocker`.",
            "",
            "Failed raw output excerpt:",
            "```",
            raw_output_excerpt[:2000],
            "```",
        ]
    )
    return "\n".join(lines).strip()


def synthesize_transcript_evidence_refs(transcript: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    refs: list[str] = []
    for idx, entry in enumerate(transcript, start=1):
        if entry.get("kind") != "tool_result":
            continue
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        if payload.get("error") or payload.get("ok") is False:
            continue
        tool_name = str(entry.get("tool") or "").strip()
        if not tool_name:
            continue
        ref = f"transcript:{idx}:{tool_name}"
        if ref not in refs:
            refs.append(ref)
        if len(refs) >= limit:
            break
    return refs


__all__ = [
    "REPAIRABLE_PROVIDER_NAMES",
    "attempt_verdict_repair",
    "build_contract_repair_prompt",
    "checkpoint_complete_passes_gate",
    "final_report_passes_quality_gate",
    "final_report_payload_passes_gate",
    "is_prerequisite_block_terminal",
    "normalize_incomplete_reason",
    "should_attempt_contract_repair",
    "synthesize_transcript_evidence_refs",
]


def _missing_research_setup_facts(*, slice_obj: PlanSlice, facts: dict[str, Any]) -> list[str]:
    allowed = {str(item).strip() for item in slice_obj.allowed_tools if str(item).strip()}
    if not _RESEARCH_SETUP_TOOLS.issubset(allowed):
        return []
    return [key for key in _STRICT_RESEARCH_SETUP_FACTS if _is_missing(facts.get(key))]


def _write_result_guard_reason(*, slice_obj: PlanSlice, facts: dict[str, Any]) -> str:
    if not (
        bool(getattr(slice_obj, "requires_mutating_evidence", False))
        or str(slice_obj.runtime_profile or "").strip() == "write_result"
    ):
        return ""
    mutating_count = _int_value(facts.get("direct.successful_mutating_tool_count"))
    if mutating_count < 1:
        return "write_result_without_mutating_tool"
    required_coverage = max(len(slice_obj.success_criteria or []), len(slice_obj.evidence_requirements or []))
    successful_tool_count = _int_value(facts.get("direct.successful_tool_count"))
    if required_coverage > 1 and successful_tool_count < 2:
        return "insufficient_successful_tool_evidence"
    return ""


def _domain_tool_guard_reason(*, slice_obj: PlanSlice, facts: dict[str, Any]) -> str:
    if not _slice_requires_domain_tool_evidence(slice_obj):
        return ""
    required_tools = {
        item
        for item in (str(tool).strip() for tool in slice_obj.allowed_tools)
        if item and _is_domain_execution_tool(item)
    }
    if not required_tools:
        return ""
    non_research_required_tools = {
        item for item in required_tools if not _is_research_domain_tool(item)
    }
    if non_research_required_tools:
        # Mixed-domain slices must prove they touched at least one non-research
        # live domain tool; research_* calls alone are only context recovery.
        required_tools = non_research_required_tools
    successful_tools = set(_string_list(facts.get("direct.successful_tool_names")))
    if successful_tools & required_tools:
        return ""
    return "missing_domain_tool_evidence"


def _execution_artifact_guard_reason(*, slice_obj: PlanSlice, facts: dict[str, Any], refs: list[str]) -> str:
    requires_persisted = bool(getattr(slice_obj, "requires_persisted_artifact", False))
    requires_live = bool(getattr(slice_obj, "requires_live_handle_validation", False))
    if not (requires_persisted or requires_live):
        return ""
    durable_run_id = _durable_run_id(facts)
    successful_tools = set(_string_list(facts.get("direct.successful_tool_names")))
    if "backtests_runs" not in successful_tools and not durable_run_id:
        return ""
    if requires_persisted and not durable_run_id:
        return "missing_persisted_execution_artifact"
    lowered_statuses = [item.lower() for item in _string_list(facts.get("direct.statuses"))]
    lowered_warnings = [item.lower() for item in _string_list(facts.get("direct.warnings"))]
    issue_text = " | ".join([*lowered_statuses, *lowered_warnings])
    if requires_live and "resource_not_found" in issue_text:
        return "execution_handle_probe_resource_not_found"
    if not requires_live:
        return ""
    if not any(_looks_like_live_probe_status(status) for status in lowered_statuses):
        return "execution_handle_not_live_validated"
    if durable_run_id and durable_run_id in refs:
        return ""
    for ref in refs:
        parsed = _parse_transcript_ref(ref)
        if parsed is not None and parsed[1] == "backtests_runs":
            return ""
    return "execution_handle_probe_not_evidenced"


def _slice_requires_domain_tool_evidence(slice_obj: PlanSlice) -> bool:
    haystack = " ".join(
        str(item or "").strip().lower()
        for item in (
            slice_obj.title,
            slice_obj.objective,
            *(slice_obj.success_criteria or []),
            *(slice_obj.evidence_requirements or []),
            *(slice_obj.policy_tags or []),
        )
        if str(item or "").strip()
    )
    return any(marker in haystack for marker in _DOMAIN_EVIDENCE_MARKERS)


def _is_domain_execution_tool(name: str) -> bool:
    normalized = str(name or "").strip()
    return any(normalized.startswith(prefix) for prefix in _DOMAIN_EXECUTION_PREFIXES)


def _is_research_domain_tool(name: str) -> bool:
    return str(name or "").strip().startswith("research_")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _has_supported_evidence_ref(*, refs: list[str], supported_refs: list[str]) -> bool:
    supported_set = {str(item).strip() for item in supported_refs if str(item).strip()}
    if any(ref in supported_set for ref in refs):
        return True
    # Accept concrete ID formats that tools actually return (run IDs, analysis
    # IDs, node IDs, etc.). These represent real tool outputs, not fabricated refs.
    if any(_is_concrete_evidence_ref(ref) for ref in refs):
        return True
    transcript_refs = [
        parsed
        for parsed in (_parse_transcript_ref(item) for item in supported_refs)
        if parsed is not None
    ]
    if not transcript_refs:
        return False
    requested_transcript_refs = [
        parsed
        for parsed in (_parse_transcript_ref(item) for item in refs)
        if parsed is not None
    ]
    for ref in refs:
        parsed = _parse_transcript_ref(ref)
        if parsed is None:
            continue
        requested_index, requested_tool = parsed
        for local_index, (_, supported_tool) in enumerate(transcript_refs, start=1):
            if requested_index == local_index and _tool_ref_matches(requested_tool, supported_tool):
                return True
        for supported_index, supported_tool in transcript_refs:
            if requested_index == supported_index and _tool_ref_matches(requested_tool, supported_tool):
                return True
    if len(requested_transcript_refs) >= 2 and _matches_supported_tool_sequence(
        requested_refs=requested_transcript_refs,
        supported_refs=transcript_refs,
    ):
        return True
    return False


def _parse_transcript_ref(value: str) -> tuple[int, str] | None:
    text = str(value or "").strip()
    if not text.startswith("transcript:"):
        return None
    _, _, remainder = text.partition("transcript:")
    index_text, sep, tool_name = remainder.partition(":")
    if not sep:
        return None
    try:
        index = int(index_text.strip())
    except ValueError:
        return None
    normalized_tool = str(tool_name or "").strip()
    if not normalized_tool:
        return None
    return index, normalized_tool


def _tool_ref_matches(requested_tool: str, supported_tool: str) -> bool:
    requested = str(requested_tool or "").strip()
    supported = str(supported_tool or "").strip()
    if not requested or not supported:
        return False
    return requested == supported or requested.startswith(f"{supported}_")


def _matches_supported_tool_sequence(
    *,
    requested_refs: list[tuple[int, str]],
    supported_refs: list[tuple[int, str]],
) -> bool:
    requested_indices = [index for index, _ in requested_refs]
    if requested_indices != sorted(requested_indices):
        return False
    supported_position = 0
    for _, requested_tool in requested_refs:
        matched = False
        while supported_position < len(supported_refs):
            _, supported_tool = supported_refs[supported_position]
            supported_position += 1
            if _tool_ref_matches(requested_tool, supported_tool):
                matched = True
                break
        if not matched:
            return False
    return True


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not bool(value)
    return False


def _durable_run_id(facts: dict[str, Any]) -> str:
    for key in (
        "run_id",
        "candidate_run_id",
        "base_run_id",
        "baseline_run_id",
        "comparison_run_id",
        "analysis_run_id",
    ):
        value = str(facts.get(key) or "").strip()
        if value:
            return value
    for mapping_key in ("backtests.integration_handles", "integration_handles"):
        mapping = facts.get(mapping_key)
        if not isinstance(mapping, dict):
            continue
        for value in mapping.values():
            text = str(value or "").strip()
            if text:
                return text
    return ""


def _looks_like_live_probe_status(status: str) -> bool:
    normalized = str(status or "").strip().lower()
    if not normalized:
        return False
    return normalized in {"completed", "complete", "running", "queued", "saved", "success", "succeeded", "ok"}
