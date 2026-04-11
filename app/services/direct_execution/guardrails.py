"""
Guardrails for direct execution fallback, repair, and evidence gating.
"""

from __future__ import annotations

from typing import Any

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.fact_hydration import hydrate_final_report_facts

REPAIRABLE_PROVIDER_NAMES = {"lmstudio", "qwen_cli", "claude_cli"}

# Verdicts that explicitly indicate the result is NOT successful.
_FAIL_VERDICTS = frozenset(
    v.upper() for v in (
        "INCOMPLETE", "FAILED", "FAILURE", "PARTIAL", "ABORTED", "UNKNOWN",
    )
)

# Minimum confidence a final_report must report to be considered successful.
_MIN_SUCCESS_CONFIDENCE = 0.5


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
    if not action.evidence_refs:
        return False
    readiness = hydrate_final_report_facts(
        slice_obj=slice_obj,
        action=action,
        required_output_facts=required_output_facts,
        inherited_facts=inherited_facts,
    )
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
    if not action.evidence_refs:
        return False
    readiness = hydrate_final_report_facts(
        slice_obj=synthetic_slice,
        action=action,
        required_output_facts=required_output_facts,
        inherited_facts=inherited_facts,
    )
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

    # 2. Confidence must meet minimum threshold.
    confidence = float(action.confidence or 0)
    if confidence < _MIN_SUCCESS_CONFIDENCE:
        return False, f"confidence_{confidence:.2f}_below_{_MIN_SUCCESS_CONFIDENCE}"

    # 3. The model must have actually used at least one tool.
    #    Zero tool calls means the model hallucinated a result without doing any work.
    if tool_call_count <= 0:
        return False, "zero_tool_calls"

    # 4. Evidence refs must be present for a genuine success.
    refs = [str(r).strip() for r in (action.evidence_refs or []) if str(r).strip()]
    if not refs:
        return False, "empty_evidence_refs"

    # 5. Required output facts must be satisfied.
    if required_output_facts:
        readiness = hydrate_final_report_facts(
            slice_obj=slice_obj,
            action=action,
            required_output_facts=required_output_facts,
            inherited_facts=inherited_facts,
        )
        if readiness.missing_required_facts:
            return False, f"missing_required_facts:{','.join(readiness.missing_required_facts[:5])}"

    return True, ""


def normalize_incomplete_reason(result: Any) -> str:
    error = str(getattr(result, "error", "") or "").strip()
    if error:
        return error
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
    raw_output = str(getattr(result, "raw_output", "") or "").strip()
    if any(
        token in error_text
        for token in (
            "direct_output_parse_failed",
            "tool_not_in_allowlist",
            "worker_action_type_invalid",
            "tool_prefixed_namespace_forbidden",
            "final_report_requires",
            "tool_call_requires",
            "abort_requires",
            "checkpoint_status_invalid",
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
    if required_output_facts:
        lines.append(
            "Required downstream facts for success: " + ", ".join(
                f"`{item}`" for item in required_output_facts if str(item).strip()
            )
        )
    if registry_missing:
        lines.extend(
            [
                "Qwen native tool registry preflight did not expose the required dev_space1 tools.",
                "Do not attempt tool calls, do not emit next_action, and do not reference unavailable tools.",
                "Return either `final_report` from already-known evidence or `abort` with reason_code=`qwen_tool_registry_missing`.",
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
    "build_contract_repair_prompt",
    "checkpoint_complete_passes_gate",
    "final_report_passes_quality_gate",
    "final_report_payload_passes_gate",
    "normalize_incomplete_reason",
    "should_attempt_contract_repair",
    "synthesize_transcript_evidence_refs",
]
