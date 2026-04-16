"""
Classify slice and plan blockers into semantic vs recoverable classes.
"""

from __future__ import annotations

from typing import Any

_SEMANTIC_REASON_MARKERS = (
    "semantic_abort",
    "user_stop",
    "goal_impossible",
    "plan_contract_violation",
    "explicit_abort",
)
_CONTRACT_REASON_MARKERS = (
    "agent_contract_misuse",
    "branch_project_contract_blocker",
    "dev_space1_tools_unavailable",
    "direct_contract_blocker",
    "direct_error_loop_detected",
    "direct_slice_missing_prerequisite_facts",
    "schema_validation_failed",
    "tool_selection_ambiguous",
    "unknown research project",
    "worker_contract_recovery_exhausted",
)
_INFRA_REASON_MARKERS = (
    "adapter_invoke_timeout",
    "adapter_timeout",
    "auto_salvage_stub_rejected",
    "checkpoint_blocked",
    "claude_cli_timeout",
    "direct_expensive_tool_budget_exhausted",
    "direct_model_stalled_before_first_action",
    "direct_model_stalled_between_actions",
    "direct_output_parse_failed",
    "direct_tool_budget_exhausted",
    "empty_evidence_refs",
    "lmstudio__chat_timeout",
    "lmstudio_http_timeout",
    "lmstudio_model_crash",
    "mcp_catalog_tool_missing",
    "missing_domain_tool_evidence",
    "qwen_cli_timeout",
    "qwen_tool_registry_missing",
    "watchlist_confidence",
    "zero_tool_calls",
)
_PROVIDER_LIMIT_MARKERS = (
    "hit your limit",
    "provider_rate_limit",
    "rate limit",
    "rate_limit",
    "too many requests",
)


def classify_blocker(
    *,
    reason_code: str = "",
    summary: str = "",
    blocker_class: str = "",
) -> str:
    explicit = str(blocker_class or "").strip().lower()
    if explicit in {"semantic", "contract", "infra", "provider_limit"}:
        return explicit
    haystack = " | ".join(
        text.strip().lower()
        for text in (str(reason_code or ""), str(summary or ""))
        if text and text.strip()
    )
    if not haystack:
        return "unknown"
    if any(marker in haystack for marker in _PROVIDER_LIMIT_MARKERS):
        return "provider_limit"
    if any(marker in haystack for marker in _SEMANTIC_REASON_MARKERS):
        return "semantic"
    if any(marker in haystack for marker in _CONTRACT_REASON_MARKERS):
        return "contract"
    if any(marker in haystack for marker in _INFRA_REASON_MARKERS):
        return "infra"
    return "unknown"


def blocker_class_from_slice(slice_obj: Any) -> str:
    reason_code = str(getattr(slice_obj, "last_error", "") or "")
    summary = str(getattr(slice_obj, "last_summary", "") or getattr(slice_obj, "last_checkpoint_summary", "") or "")
    if reason_code == "dependency_blocked":
        return classify_blocker(
            reason_code=str(getattr(slice_obj, "dependency_blocker_reason_code", "") or ""),
            summary=summary,
            blocker_class=str(getattr(slice_obj, "dependency_blocker_class", "") or ""),
        )
    return classify_blocker(reason_code=reason_code, summary=summary)


__all__ = ["blocker_class_from_slice", "classify_blocker"]
