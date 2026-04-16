"""
Failure provenance helpers for direct fallback chains.
"""

from __future__ import annotations

from typing import Any

from app.execution_models import PlanSlice, WorkerAction, make_id
from app.services.direct_execution.executor import DirectExecutionResult
from app.services.direct_execution.guardrails import final_report_passes_quality_gate, normalize_incomplete_reason


def is_provider_rate_limit(error_text: str) -> bool:
    normalized = str(error_text or "").strip().lower()
    return any(token in normalized for token in ("rate limit", "rate_limit", "hit your limit", "too many requests"))


def build_actionable_failure_checkpoint(
    *,
    last_result: DirectExecutionResult,
    attempts: list[Any],
    slice_obj: PlanSlice,
    required_output_facts: list[str],
    inherited_facts: dict[str, Any],
) -> DirectExecutionResult:
    """Preserve the best evidence-bearing failure when the tail is infra noise."""

    if last_result.action is not None:
        return last_result
    best_attempt = _best_failed_attempt(
        attempts=attempts,
        slice_obj=slice_obj,
        required_output_facts=required_output_facts,
        inherited_facts=inherited_facts,
    )
    if best_attempt is None:
        return last_result
    provider_limit_seen = is_provider_rate_limit(normalize_incomplete_reason(last_result))
    root_reason = best_attempt["reason"]
    facts = {
        "direct.root_failure_reason": root_reason,
        "direct.root_failure_artifact": best_attempt["artifact_path"],
        "direct.best_failed_attempt_provider": best_attempt["provider"],
        "direct.best_failed_tool_call_count": best_attempt["tool_call_count"],
        "direct.last_provider_failure": normalize_incomplete_reason(last_result),
    }
    if provider_limit_seen:
        facts["direct.provider_limit_seen"] = True
    artifacts = [
        item
        for item in (best_attempt["artifact_path"], last_result.artifact_path)
        if str(item or "").strip()
    ]
    action = WorkerAction(
        action_id=make_id("action"),
        action_type="checkpoint",
        status="blocked",
        summary=f"Fallback chain exhausted; actionable root failure: {root_reason}.",
        facts=facts,
        artifacts=artifacts,
        reason_code="direct_output_parse_failed",
    )
    return DirectExecutionResult(
        action=action,
        artifact_path=best_attempt["artifact_path"] or last_result.artifact_path,
        raw_output=last_result.raw_output,
        error=normalize_incomplete_reason(last_result),
        provider=last_result.provider,
        duration_ms=last_result.duration_ms,
        tool_call_count=max(int(last_result.tool_call_count or 0), int(best_attempt["tool_call_count"] or 0)),
        expensive_tool_call_count=max(
            int(last_result.expensive_tool_call_count or 0),
            int(best_attempt["expensive_tool_call_count"] or 0),
        ),
        parse_retry_count=last_result.parse_retry_count,
        fallback_provider_index=last_result.fallback_provider_index,
        transcript=list(last_result.transcript or []),
        acceptance_result=dict(last_result.acceptance_result or {}),
    )


def _best_failed_attempt(
    *,
    attempts: list[Any],
    slice_obj: PlanSlice,
    required_output_facts: list[str],
    inherited_facts: dict[str, Any],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for attempt in attempts:
        result = attempt.result
        if int(result.tool_call_count or 0) <= 0:
            continue
        reason = normalize_incomplete_reason(result)
        if result.action is not None and result.action.action_type == "final_report":
            passes, gate_reason = final_report_passes_quality_gate(
                tool_call_count=result.tool_call_count,
                action=result.action,
                slice_obj=slice_obj,
                required_output_facts=required_output_facts,
                inherited_facts=inherited_facts,
            )
            if not passes:
                reason = gate_reason
        candidate = {
            "provider": attempt.provider,
            "artifact_path": result.artifact_path or attempt.artifact_path,
            "reason": reason,
            "tool_call_count": int(result.tool_call_count or 0),
            "expensive_tool_call_count": int(result.expensive_tool_call_count or 0),
        }
        if best is None or candidate["tool_call_count"] > int(best["tool_call_count"]):
            best = candidate
    return best


__all__ = ["build_actionable_failure_checkpoint", "is_provider_rate_limit"]
