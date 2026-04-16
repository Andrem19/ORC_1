"""Tests for optional slice gate condition checking (Fix 2)."""
from __future__ import annotations

from app.execution_models import BaselineRef, ExecutionPlan, PlanSlice
from app.services.direct_execution.slice_readiness import optional_slice_gate_blocker


def _plan_with_slices(*slices: PlanSlice) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="test_plan",
        goal="test",
        baseline_ref=BaselineRef(snapshot_id="snap", version=1),
        global_constraints=[],
        slices=list(slices),
    )


def _optional_slice(
    *,
    slice_id: str = "seq_stage_b",
    gate_hint: str = "Wave A inconclusive: no candidates accepted",
    depends_on: list[str] | None = None,
    policy_tags: list[str] | None = None,
) -> PlanSlice:
    return PlanSlice(
        slice_id=slice_id,
        title="Wave B construction",
        hypothesis="h",
        objective="model-backed routing",
        success_criteria=["at least one Wave B candidate"],
        allowed_tools=["features_custom", "models_train", "models_to_feature", "research_memory"],
        evidence_requirements=["at least one Wave B candidate"],
        policy_tags=policy_tags or ["optional_candidate", "modeling"],
        max_turns=36,
        max_tool_calls=30,
        max_expensive_calls=6,
        parallel_slot=1,
        gate_hint=gate_hint,
        depends_on=depends_on or ["seq_stage_a"],
    )


def _upstream_slice(
    *,
    slice_id: str = "seq_stage_a",
    facts: dict | None = None,
) -> PlanSlice:
    return PlanSlice(
        slice_id=slice_id,
        title="Wave A exploration",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=10,
        max_tool_calls=10,
        max_expensive_calls=0,
        parallel_slot=1,
        facts=facts or {},
        status="completed",
    )


def test_optional_slice_blocked_when_wave_a_conclusive() -> None:
    """Optional slice should be blocked when upstream facts indicate Wave A
    was conclusive (has shortlist families)."""
    upstream = _upstream_slice(
        facts={"research.shortlist_families": ["momentum", "funding"]},
    )
    optional = _optional_slice()
    plan = _plan_with_slices(upstream, optional)

    blocker = optional_slice_gate_blocker(plan, optional)
    assert blocker is not None
    assert blocker.reason_code == "optional_gate_condition_not_met"
    assert "gate condition not met" in blocker.summary.lower() or "skipped" in blocker.summary.lower()


def test_optional_slice_allowed_when_wave_a_inconclusive() -> None:
    """Optional slice should NOT be blocked when upstream facts show no accepted
    candidates (Wave A was inconclusive)."""
    upstream = _upstream_slice(facts={})
    optional = _optional_slice()
    plan = _plan_with_slices(upstream, optional)

    blocker = optional_slice_gate_blocker(plan, optional)
    assert blocker is None


def test_optional_slice_allowed_when_wave_a_empty_candidates() -> None:
    """Optional slice should NOT be blocked when candidate handles is empty list."""
    upstream = _upstream_slice(
        facts={"backtests.candidate_handles": []},
    )
    optional = _optional_slice()
    plan = _plan_with_slices(upstream, optional)

    blocker = optional_slice_gate_blocker(plan, optional)
    assert blocker is None


def test_required_slice_not_blocked() -> None:
    """A non-optional slice (no 'optional_candidate' tag) should never be blocked."""
    upstream = _upstream_slice(
        facts={"backtests.candidate_handles": ["snap_v1_run_abc"]},
    )
    required = PlanSlice(
        slice_id="seq_stage_b",
        title="Required stage",
        hypothesis="h",
        objective="model-backed routing",
        success_criteria=[],
        allowed_tools=["models_train"],
        evidence_requirements=[],
        policy_tags=["modeling"],  # no 'optional_candidate'
        max_turns=10,
        max_tool_calls=10,
        max_expensive_calls=0,
        parallel_slot=1,
        gate_hint="Wave A inconclusive",
        depends_on=["seq_stage_a"],
    )
    plan = _plan_with_slices(upstream, required)

    blocker = optional_slice_gate_blocker(plan, required)
    assert blocker is None


def test_no_gate_hint_returns_none() -> None:
    """Optional slice with no gate_hint should not be blocked."""
    upstream = _upstream_slice(facts={})
    optional = _optional_slice(gate_hint="")
    plan = _plan_with_slices(upstream, optional)

    blocker = optional_slice_gate_blocker(plan, optional)
    assert blocker is None


def test_research_shortlist_families_triggers_block() -> None:
    """research.shortlist_families in upstream facts should also trigger block."""
    upstream = _upstream_slice(
        facts={"research.shortlist_families": ["family_alpha", "family_beta"]},
    )
    optional = _optional_slice()
    plan = _plan_with_slices(upstream, optional)

    blocker = optional_slice_gate_blocker(plan, optional)
    assert blocker is not None
    assert blocker.reason_code == "optional_gate_condition_not_met"


def test_transitive_upstream_facts_checked() -> None:
    """Gate check should gather facts from transitive dependencies."""
    # Three-level chain: grandparent -> parent -> optional
    grandparent = PlanSlice(
        slice_id="seq_gp",
        title="GP",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=[],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=5,
        max_tool_calls=5,
        max_expensive_calls=0,
        parallel_slot=1,
        facts={"research.shortlist_families": ["momentum", "funding"]},
        status="completed",
    )
    parent = PlanSlice(
        slice_id="seq_parent",
        title="Parent",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=[],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=5,
        max_tool_calls=5,
        max_expensive_calls=0,
        parallel_slot=1,
        depends_on=["seq_gp"],
        status="completed",
    )
    optional = _optional_slice(depends_on=["seq_parent"])
    plan = _plan_with_slices(grandparent, parent, optional)

    blocker = optional_slice_gate_blocker(plan, optional)
    assert blocker is not None
    assert blocker.reason_code == "optional_gate_condition_not_met"
