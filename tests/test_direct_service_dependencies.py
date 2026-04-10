from __future__ import annotations

from app.execution_models import BaselineRef, ExecutionPlan, ExecutionStateV2, PlanSlice
from app.services.direct_execution.service import DirectExecutionService


def _slice(slice_id: str, *, status: str = "pending", depends_on: list[str] | None = None, facts: dict | None = None) -> PlanSlice:
    return PlanSlice(
        slice_id=slice_id,
        title=slice_id,
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_map"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status=status,
        depends_on=list(depends_on or []),
        facts=dict(facts or {}),
    )


def _plan(plan_id: str, slices: list[PlanSlice], *, status: str = "running") -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        goal="g",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=[],
        slices=slices,
        status=status,
    )


def test_dependencies_satisfied_resolves_cross_batch_slice_ids() -> None:
    stage3 = _slice("compiled_plan_v1_stage_3", status="completed")
    plan1 = _plan("compiled_plan_v1_batch_1", [stage3], status="completed")
    stage4 = _slice("compiled_plan_v1_stage_4", status="pending", depends_on=["compiled_plan_v1_stage_3"])
    plan2 = _plan("compiled_plan_v1_batch_2", [stage4], status="running")

    service = DirectExecutionService.__new__(DirectExecutionService)
    service.state = ExecutionStateV2(goal="g", plans=[plan1, plan2])

    assert service._dependencies_satisfied(plan2, stage4) is True


def test_known_facts_for_slice_merges_cross_batch_dependency_facts() -> None:
    stage3 = _slice(
        "compiled_plan_v1_stage_3",
        status="completed",
        facts={"research.project_id": "proj_123", "research.shortlist_families": ["alpha"]},
    )
    plan1 = _plan("compiled_plan_v1_batch_1", [stage3], status="completed")
    stage4 = _slice("compiled_plan_v1_stage_4", status="pending", depends_on=["compiled_plan_v1_stage_3"])
    plan2 = _plan("compiled_plan_v1_batch_2", [stage4], status="running")

    service = DirectExecutionService.__new__(DirectExecutionService)
    service.state = ExecutionStateV2(goal="g", plans=[plan1, plan2])

    facts = service._known_facts_for_slice(plan2, stage4)
    assert facts["compiled_plan_v1_stage_3.research.project_id"] == "proj_123"
    assert facts["compiled_plan_v1_stage_3.research.shortlist_families"] == ["alpha"]
