from __future__ import annotations

from app.execution_models import BaselineRef, ExecutionPlan, PlanSlice
from app.services.direct_execution.slice_readiness import dependency_readiness_blocker, required_output_facts_for_slice


def _plan(*slices: PlanSlice) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_1",
        goal="goal",
        baseline_ref=BaselineRef(snapshot_id="s", version=1),
        global_constraints=[],
        slices=list(slices),
    )


def test_required_output_facts_for_features_catalog_slice() -> None:
    slice_obj = PlanSlice(
        slice_id="stage_3",
        title="Feature contract",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["features_catalog"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        depends_on=["stage_2"],
    )

    assert required_output_facts_for_slice(_plan(slice_obj), slice_obj) == [
        "research.project_id",
        "research.shortlist_families",
    ]


def test_dependency_readiness_blocker_when_upstream_facts_missing() -> None:
    upstream = PlanSlice(
        slice_id="stage_2",
        title="Shortlist",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_record"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={"research.project_id": "proj_1"},
    )
    downstream = PlanSlice(
        slice_id="stage_3",
        title="Feature contract",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["features_catalog"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        depends_on=["stage_2"],
    )

    blocker = dependency_readiness_blocker(_plan(upstream, downstream), downstream)

    assert blocker is not None
    assert blocker.reason_code == "direct_slice_missing_prerequisite_facts"
    assert "research.shortlist_families" in blocker.missing_facts


def test_dependency_readiness_allows_slice_when_upstream_facts_present() -> None:
    upstream = PlanSlice(
        slice_id="stage_2",
        title="Shortlist",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_record"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={
            "research.project_id": "proj_1",
            "research.shortlist_families": ["funding divergence"],
            "research.hypothesis_refs": ["node_1"],
        },
    )
    downstream = PlanSlice(
        slice_id="stage_3",
        title="Feature contract",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["features_catalog"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        depends_on=["stage_2"],
    )

    assert dependency_readiness_blocker(_plan(upstream, downstream), downstream) is None
