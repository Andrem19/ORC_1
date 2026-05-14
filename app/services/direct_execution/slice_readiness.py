"""
Semantic readiness checks for dependent direct slices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.execution_models import ExecutionPlan, PlanSlice
from app.services.direct_execution.backtests_facts import normalize_backtests_facts


@dataclass(frozen=True)
class ReadinessBlocker:
    summary: str
    reason_code: str
    missing_facts: list[str]
    blocking_slice_ids: list[str]


def required_prerequisite_facts_for_slice(plan: ExecutionPlan, slice_obj: PlanSlice) -> list[str]:
    del plan
    return [
        str(item).strip()
        for item in slice_obj.required_prerequisite_facts
        if str(item).strip()
    ]


def required_output_facts_for_slice(plan: ExecutionPlan, slice_obj: PlanSlice) -> list[str]:
    del plan
    return [str(item).strip() for item in slice_obj.required_output_facts if str(item).strip()]


def dependency_readiness_blocker(
    plan: ExecutionPlan,
    slice_obj: PlanSlice,
    *,
    resolve_dependency: Callable[[str], PlanSlice | None] | None = None,
    cross_plan_slices: dict[str, PlanSlice] | None = None,
) -> ReadinessBlocker | None:
    required = required_prerequisite_facts_for_slice(plan, slice_obj)
    if not required or not slice_obj.depends_on:
        return None
    by_id = {item.slice_id: item for item in plan.slices}
    cross = dict(cross_plan_slices or {})
    def _default_resolver(sid: str) -> PlanSlice | None:
        return by_id.get(sid) or cross.get(sid)
    resolver = resolve_dependency or _default_resolver
    missing: list[str] = []
    blockers: list[str] = []
    merged = {}
    for dep_id in slice_obj.depends_on:
        dep = resolver(dep_id)
        if dep is None:
            continue
        blockers.append(dep_id)
        for key, value in dep.facts.items():
            merged.setdefault(key, value)
    merged = normalize_backtests_facts(merged)
    for key in required:
        value = merged.get(key)
        if _is_missing(value):
            missing.append(key)
    if not missing:
        return None
    return ReadinessBlocker(
        summary=(
            f"Missing prerequisite facts for slice '{slice_obj.slice_id}': "
            f"{', '.join(missing)}. Upstream output is incomplete, so direct execution would be underdefined."
        ),
        reason_code="direct_slice_missing_prerequisite_facts",
        missing_facts=missing,
        blocking_slice_ids=blockers,
    )


def downstream_prerequisites_blocker(
    plan: ExecutionPlan,
    completed_slice: PlanSlice,
    *,
    resolve_dependency: Callable[[str], PlanSlice | None] | None = None,
    hydrated_facts: dict | None = None,
) -> ReadinessBlocker | None:
    """Check whether completing *completed_slice* would still leave downstream
    dependents unsatisfied.  Returns a ``ReadinessBlocker`` when any downstream
    slice has ``required_prerequisite_facts`` that are missing even after
    merging facts from **all** of its dependencies (using *hydrated_facts* for
    the completing slice).
    """
    slice_id = completed_slice.slice_id
    by_id = {item.slice_id: item for item in plan.slices}
    resolver = resolve_dependency or by_id.get

    all_missing: list[str] = []
    all_blocking_ids: list[str] = []

    for candidate in plan.slices:
        if slice_id not in (candidate.depends_on or []):
            continue
        required = required_prerequisite_facts_for_slice(plan, candidate)
        if not required:
            continue
        merged: dict = {}
        for dep_id in candidate.depends_on:
            dep = resolver(dep_id)
            if dep is None:
                continue
            source_facts = hydrated_facts if dep_id == slice_id else dep.facts
            for key, value in source_facts.items():
                merged.setdefault(key, value)
        merged = normalize_backtests_facts(merged)
        for key in required:
            if key not in all_missing and _is_missing(merged.get(key)):
                all_missing.append(key)
        if all_missing:
            all_blocking_ids.append(candidate.slice_id)

    if not all_missing:
        return None
    return ReadinessBlocker(
        summary=(
            f"Downstream slices need facts not yet available from slice '{slice_id}': "
            f"{', '.join(all_missing)}."
        ),
        reason_code="direct_slice_missing_downstream_prerequisites",
        missing_facts=all_missing,
        blocking_slice_ids=all_blocking_ids,
    )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not bool(value)
    return False


def optional_slice_gate_blocker(
    plan: ExecutionPlan,
    slice_obj: PlanSlice,
) -> ReadinessBlocker | None:
    """Check whether an optional/gated slice should be skipped.

    Returns a ``ReadinessBlocker`` when the slice has ``optional_candidate`` in
    its ``policy_tags`` AND its gate condition is clearly NOT met based on
    upstream dependency facts.

    This does NOT weaken quality gates -- it prevents running a slice whose
    prerequisite condition (defined in the plan itself) was not satisfied.
    """
    from typing import Any as _Any

    policy_tags = [str(t).strip().lower() for t in (slice_obj.policy_tags or [])]
    if "optional_candidate" not in policy_tags:
        return None

    gate_hint = str(slice_obj.gate_hint or "").strip()
    if not gate_hint:
        return None

    # Gather all upstream facts from dependencies.
    by_id = {item.slice_id: item for item in plan.slices}
    upstream_facts: dict[str, _Any] = {}
    visited: set[str] = set()
    stack = list(slice_obj.depends_on or [])
    while stack:
        dep_id = stack.pop()
        if dep_id in visited or dep_id not in by_id:
            continue
        visited.add(dep_id)
        dep = by_id[dep_id]
        for key, value in dep.facts.items():
            upstream_facts.setdefault(key, value)
        stack.extend(dep.depends_on or [])

    if _gate_condition_not_met(gate_hint, upstream_facts):
        return ReadinessBlocker(
            summary=(
                f"Optional slice '{slice_obj.slice_id}' skipped: "
                f"gate condition not met -- {gate_hint[:120]}"
            ),
            reason_code="optional_gate_condition_not_met",
            missing_facts=[],
            blocking_slice_ids=[],
        )

    return None


def _gate_condition_not_met(gate_hint: str, upstream_facts: dict) -> bool:
    """Evaluate whether an optional branch's gate condition is NOT met.

    Returns True when the gate says the branch should NOT run.
    """
    hint_lower = gate_hint.lower()

    # Pattern: "Wave A inconclusive" means the branch runs ONLY if Wave A failed.
    if "inconclusive" in hint_lower:
        # If upstream facts contain accepted/promoted candidates, Wave A was
        # conclusive and the optional branch should NOT run.
        acceptance_markers = [
            "research.shortlist_families",
            "research.wave_a_accepted",
        ]
        for marker in acceptance_markers:
            value = upstream_facts.get(marker)
            if value and not _is_missing(value):
                return True  # Wave A produced results -> optional branch should NOT run
        return False  # No accepted candidates -> optional branch SHOULD run

    # Default: gate condition is met (don't block).
    return False


def upstream_artifact_gate_blocker(
    plan: ExecutionPlan,
    slice_obj: PlanSlice,
    *,
    resolve_dependency: Callable[[str], PlanSlice | None] | None = None,
    cross_plan_slices: dict[str, PlanSlice] | None = None,
) -> ReadinessBlocker | None:
    """Block a slice whose acceptance contract requires ``run_set_non_empty``
    when all upstream dependencies completed with zero backtest runs.

    This prevents the orchestrator from burning budget on a slice that can
    never pass acceptance because its predecessor produced no analysable
    artifacts.  The slice is auto-completed as SKIP instead.

    This does **not** weaken any acceptance criterion — the slice would have
    failed ``run_set_non_empty`` regardless.
    """

    contract = slice_obj.acceptance_contract or {}
    predicates = contract.get("required_predicates") or []
    if "run_set_non_empty" not in predicates:
        return None

    by_id = {item.slice_id: item for item in plan.slices}
    cross = dict(cross_plan_slices or {})

    def _default_resolver(sid: str) -> PlanSlice | None:
        return by_id.get(sid) or cross.get(sid)

    resolver = resolve_dependency or _default_resolver

    deps_with_zero_runs: list[str] = []
    for dep_id in slice_obj.depends_on or []:
        dep = resolver(dep_id)
        if dep is None:
            continue
        if dep.status not in ("completed",):
            continue

        if _upstream_has_zero_artifacts(dep.facts):
            deps_with_zero_runs.append(dep_id)

    if not deps_with_zero_runs:
        return None

    dep_labels = ", ".join(deps_with_zero_runs)
    return ReadinessBlocker(
        summary=(
            f"Slice '{slice_obj.slice_id}' requires runs but upstream "
            f"dependencies ({dep_labels}) completed with zero backtest runs. "
            "Acceptance predicate run_set_non_empty cannot pass."
        ),
        reason_code="upstream_zero_artifacts_gate",
        missing_facts=[],
        blocking_slice_ids=deps_with_zero_runs,
    )


def _upstream_has_zero_artifacts(facts: dict) -> bool:
    """Return True when upstream dependency facts indicate zero analysable
    backtest artifacts — regardless of which exact fact keys the worker used.

    Workers (minimax, claude, etc.) may express "nothing produced" via different
    fact keys, so we check a broad set of signals.
    """
    # Explicit counter facts.
    runs_found = facts.get("backtest_runs_found")
    if isinstance(runs_found, int) and runs_found == 0:
        return True
    if isinstance(runs_found, str) and runs_found.strip() == "0":
        return True

    snapshots_found = facts.get("strategy_snapshots_found")
    if isinstance(snapshots_found, int) and snapshots_found == 0:
        return True
    if isinstance(snapshots_found, str) and snapshots_found.strip() == "0":
        return True

    # Worker-reported reason strings.
    reason = str(facts.get("reason", "")).lower()
    if "no_candidate_snapshots" in reason or "no_candidate_runs" in reason:
        return True

    # Candidate/signal count facts produced by various worker versions.
    candidates_count = facts.get("standalone_candidates_count")
    if isinstance(candidates_count, int) and candidates_count == 0:
        return True
    if isinstance(candidates_count, str) and candidates_count.strip() == "0":
        return True

    # Shortlist status — "empty" means no candidates were produced.
    shortlist = str(facts.get("shortlist_status", "")).strip().lower()
    if shortlist == "empty":
        return True

    # Normalized backtests handles — empty dict/list means no candidate runs.
    for key in ("backtests.candidate_handles", "backtests.integration_handles"):
        val = facts.get(key)
        if val is not None and not val:
            return True

    return False


__all__ = [
    "ReadinessBlocker",
    "dependency_readiness_blocker",
    "downstream_prerequisites_blocker",
    "optional_slice_gate_blocker",
    "required_output_facts_for_slice",
    "required_prerequisite_facts_for_slice",
    "upstream_artifact_gate_blocker",
]
