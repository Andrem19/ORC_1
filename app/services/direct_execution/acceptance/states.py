"""
Acceptance-state helpers for slice completion and dependency unblocking.
"""

from __future__ import annotations

from app.execution_models import PlanSlice

_DEPENDENCY_READY_STATES = frozenset({"accepted_ready", "advisory_only_done"})


def normalize_dependency_unblock_mode(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "advisory_only":
        return "advisory_only"
    return "accepted_only"


def slice_allows_watchlist_unblock(slice_obj: PlanSlice) -> bool:
    return bool(getattr(slice_obj, "watchlist_allows_unblock", False))


def slice_requires_strict_acceptance(slice_obj: PlanSlice) -> bool:
    return (
        normalize_dependency_unblock_mode(getattr(slice_obj, "dependency_unblock_mode", ""))
        == "accepted_only"
        and not slice_allows_watchlist_unblock(slice_obj)
    )


def verdict_is_accepted(slice_obj: PlanSlice, verdict: str) -> bool:
    verdict_norm = str(verdict or "").strip().upper()
    if verdict_norm == "WATCHLIST" and slice_requires_strict_acceptance(slice_obj):
        return False
    return True


def verdict_acceptance_blocker_reason(slice_obj: PlanSlice, verdict: str) -> str:
    verdict_norm = str(verdict or "").strip().upper()
    if verdict_norm == "WATCHLIST" and slice_requires_strict_acceptance(slice_obj):
        return "watchlist_not_accepted"
    if verdict_norm in {"SKIP", "SKIPPED", "REJECT"}:
        return "terminal_report_not_accepted"
    if verdict_norm:
        return "terminal_report_not_accepted"
    return ""


def proof_is_accepted(slice_obj: PlanSlice) -> bool:
    proof = getattr(slice_obj, "acceptance_proof", {}) or {}
    if not isinstance(proof, dict):
        return False
    return str(proof.get("status") or "").strip().lower() == "pass"


def accepted_completion_state(*, slice_obj: PlanSlice, verdict: str) -> str:
    # Acceptance proof is the canonical strict-acceptance signal; when it passes
    # the slice is accepted regardless of the agent's self-reported verdict.
    # Mirrors the override in fallback_executor._is_success that treats a passing
    # final_report acceptance proof as overriding a non-accepted verdict (e.g.
    # WATCHLIST), so downstream dependents are unblocked consistently.
    del verdict  # retained in signature for API stability; proof drives the state
    if not proof_is_accepted(slice_obj):
        return "reported_terminal"
    if normalize_dependency_unblock_mode(getattr(slice_obj, "dependency_unblock_mode", "")) == "advisory_only":
        return "advisory_only_done"
    return "accepted_ready"


def dependency_unblocked_by(slice_obj: PlanSlice) -> bool:
    acceptance_state = str(getattr(slice_obj, "acceptance_state", "") or "").strip().lower()
    if acceptance_state in _DEPENDENCY_READY_STATES:
        return True
    if acceptance_state and acceptance_state not in {"pending", "accepted_ready", "advisory_only_done"}:
        return False
    return False


def acceptance_blocker_reason(slice_obj: PlanSlice) -> str:
    acceptance_state = str(getattr(slice_obj, "acceptance_state", "") or "").strip().lower()
    if acceptance_state == "blocked":
        return str(getattr(slice_obj, "last_error", "") or "slice_blocked")
    if acceptance_state not in {"reported_terminal"}:
        return ""
    blockers = getattr(slice_obj, "acceptance_blockers", None)
    if isinstance(blockers, list) and blockers:
        return str(blockers[0] or "acceptance_verifier_failed")
    proof = getattr(slice_obj, "acceptance_proof", {}) or {}
    if isinstance(proof, dict) and proof.get("status") and str(proof.get("status")).lower() != "pass":
        return "acceptance_verifier_failed"
    return verdict_acceptance_blocker_reason(slice_obj, getattr(slice_obj, "verdict", ""))


__all__ = [
    "acceptance_blocker_reason",
    "accepted_completion_state",
    "dependency_unblocked_by",
    "normalize_dependency_unblock_mode",
    "proof_is_accepted",
    "slice_requires_strict_acceptance",
    "slice_allows_watchlist_unblock",
    "verdict_acceptance_blocker_reason",
    "verdict_is_accepted",
]
