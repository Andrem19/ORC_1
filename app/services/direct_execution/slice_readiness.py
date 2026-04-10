"""
Semantic readiness checks for dependent direct slices.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.execution_models import ExecutionPlan, PlanSlice


@dataclass(frozen=True)
class ReadinessBlocker:
    summary: str
    reason_code: str
    missing_facts: list[str]
    blocking_slice_ids: list[str]


def required_output_facts_for_slice(plan: ExecutionPlan, slice_obj: PlanSlice) -> list[str]:
    del plan
    tools = {str(item or "").strip() for item in slice_obj.allowed_tools}
    if tools == {"features_catalog"}:
        return ["research.project_id", "research.shortlist_families"]
    return []


def dependency_readiness_blocker(plan: ExecutionPlan, slice_obj: PlanSlice) -> ReadinessBlocker | None:
    required = required_output_facts_for_slice(plan, slice_obj)
    if not required or not slice_obj.depends_on:
        return None
    by_id = {item.slice_id: item for item in plan.slices}
    missing: list[str] = []
    blockers: list[str] = []
    merged = {}
    for dep_id in slice_obj.depends_on:
        dep = by_id.get(dep_id)
        if dep is None:
            continue
        blockers.append(dep_id)
        for key, value in dep.facts.items():
            merged.setdefault(key, value)
    for key in required:
        value = merged.get(key)
        if _is_missing(value):
            missing.append(key)
    if not missing:
        return None
    return ReadinessBlocker(
        summary=(
            f"Missing prerequisite research facts for slice '{slice_obj.slice_id}': "
            f"{', '.join(missing)}. Upstream output is incomplete, so direct execution would be underdefined."
        ),
        reason_code="direct_slice_missing_prerequisite_facts",
        missing_facts=missing,
        blocking_slice_ids=blockers,
    )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not bool(value)
    return False


__all__ = ["ReadinessBlocker", "dependency_readiness_blocker", "required_output_facts_for_slice"]
