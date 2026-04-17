from __future__ import annotations

from app.execution_models import BaselineRef, ExecutionPlan, PlanSlice
from app.services.direct_execution.slice_readiness import (
    dependency_readiness_blocker,
    downstream_prerequisites_blocker,
    required_output_facts_for_slice,
    required_prerequisite_facts_for_slice,
    upstream_artifact_gate_blocker,
)


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
        runtime_profile="catalog_contract_probe",
        required_output_facts=["features_catalog.scopes"],
        finalization_mode="fact_based",
        depends_on=["stage_2"],
    )
    result = required_output_facts_for_slice(_plan(slice_obj), slice_obj)
    assert result == ["features_catalog.scopes"]


def test_required_prerequisite_facts_for_research_shortlist_slice() -> None:
    slice_obj = PlanSlice(
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
        runtime_profile="research_shortlist",
        required_prerequisite_facts=[],
        required_output_facts=[
            "research.shortlist_families",
            "research.novelty_justification_present",
        ],
        finalization_mode="fact_based",
        depends_on=["stage_1"],
    )

    result = required_prerequisite_facts_for_slice(_plan(slice_obj), slice_obj)
    assert result == []


def test_dependency_readiness_blocker_when_prerequisite_facts_missing() -> None:
    upstream = PlanSlice(
        slice_id="stage_1",
        title="Setup",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_project"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={},
    )
    downstream = PlanSlice(
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
        runtime_profile="research_shortlist",
        required_prerequisite_facts=[],
        required_output_facts=[
            "research.shortlist_families",
            "research.novelty_justification_present",
        ],
        finalization_mode="fact_based",
        depends_on=["stage_1"],
    )

    blocker = dependency_readiness_blocker(_plan(upstream, downstream), downstream)

    # No ID prerequisites → no blocker
    assert blocker is None


def test_dependency_readiness_allows_slice_when_prerequisite_facts_present() -> None:
    upstream = PlanSlice(
        slice_id="stage_1",
        title="Setup",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_project"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={
            "research.project_id": "proj_1",
            "research.atlas_defined": True,
        },
    )
    downstream = PlanSlice(
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
        runtime_profile="research_shortlist",
        required_prerequisite_facts=[],
        required_output_facts=[
            "research.shortlist_families",
            "research.novelty_justification_present",
        ],
        finalization_mode="fact_based",
        depends_on=["stage_1"],
    )

    assert dependency_readiness_blocker(_plan(upstream, downstream), downstream) is None


def test_dependency_readiness_does_not_treat_output_facts_as_prerequisites() -> None:
    upstream = PlanSlice(
        slice_id="stage_2",
        title="Setup",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_project"],
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
        runtime_profile="catalog_contract_probe",
        required_prerequisite_facts=[],
        required_output_facts=["features_catalog.scopes"],
        finalization_mode="fact_based",
        depends_on=["stage_2"],
    )

    assert dependency_readiness_blocker(_plan(upstream, downstream), downstream) is None


def test_dependency_readiness_uses_cross_batch_dependency_resolver() -> None:
    prior_batch = PlanSlice(
        slice_id="compiled_plan_v1_stage_3_part1",
        title="Exploration",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={"research.project_id": "proj_cross_batch"},
    )
    current_batch = PlanSlice(
        slice_id="compiled_plan_v1_stage_3_part2",
        title="Construction",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["features_custom"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        runtime_profile="write_result",
        required_prerequisite_facts=[],
        required_output_facts=[],
        finalization_mode="none",
        depends_on=["compiled_plan_v1_stage_3_part1"],
    )

    blocker = dependency_readiness_blocker(
        _plan(current_batch),
        current_batch,
        resolve_dependency=lambda dep_id: prior_batch if dep_id == prior_batch.slice_id else None,
    )

    assert blocker is None


def test_dependency_readiness_accepts_canonical_backtests_prerequisites_from_legacy_dependency_facts() -> None:
    upstream = PlanSlice(
        slice_id="stage_6",
        title="Stability",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_conditions"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={
            "research.project_id": "proj_1",
            "feature_long_job": "cond-f07199b451c1",
            "feature_short_job": "cond-8972f7822bb2",
            "diagnostics_run": "20260411-193208-40dbd831",
        },
    )
    downstream = PlanSlice(
        slice_id="stage_7",
        title="Integration",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs", "backtests_analysis", "research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        runtime_profile="backtests_integration_analysis",
        required_prerequisite_facts=[],
        required_output_facts=[],
        finalization_mode="none",
        depends_on=["stage_6"],
    )

    assert dependency_readiness_blocker(_plan(upstream, downstream), downstream) is None


# --- downstream_prerequisites_blocker tests ---


def test_downstream_blocker_detects_missing_prerequisites_in_downstream() -> None:
    completed = PlanSlice(
        slice_id="stage_2",
        title="Generic mutation",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={},
        required_output_facts=[],
    )
    downstream = PlanSlice(
        slice_id="stage_3",
        title="Integration",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_analysis"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        runtime_profile="backtests_integration_analysis",
        required_prerequisite_facts=["research.shortlist_families"],
        required_output_facts=[],
        finalization_mode="none",
        depends_on=["stage_2"],
    )
    blocker = downstream_prerequisites_blocker(
        _plan(completed, downstream),
        completed,
        hydrated_facts={},
    )
    assert blocker is not None
    assert blocker.reason_code == "direct_slice_missing_downstream_prerequisites"
    assert "research.shortlist_families" in blocker.missing_facts
    assert "stage_3" in blocker.blocking_slice_ids


def test_downstream_blocker_returns_none_when_downstream_satisfied() -> None:
    completed = PlanSlice(
        slice_id="stage_2",
        title="Generic mutation",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={},
        required_output_facts=[],
    )
    downstream = PlanSlice(
        slice_id="stage_3",
        title="Integration",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_analysis"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        runtime_profile="backtests_integration_analysis",
        required_prerequisite_facts=["research.shortlist_families"],
        required_output_facts=[],
        finalization_mode="none",
        depends_on=["stage_2"],
    )
    blocker = downstream_prerequisites_blocker(
        _plan(completed, downstream),
        completed,
        hydrated_facts={"research.shortlist_families": ["momentum", "funding"]},
    )
    assert blocker is None


def test_downstream_blocker_returns_none_when_no_downstream() -> None:
    completed = PlanSlice(
        slice_id="stage_final",
        title="Final report",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={"research.project_id": "proj_1"},
        required_output_facts=[],
    )
    blocker = downstream_prerequisites_blocker(
        _plan(completed),
        completed,
        hydrated_facts={},
    )
    assert blocker is None


def test_downstream_blocker_returns_none_when_downstream_has_no_prerequisites() -> None:
    completed = PlanSlice(
        slice_id="stage_2",
        title="Generic mutation",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={},
        required_output_facts=[],
    )
    downstream = PlanSlice(
        slice_id="stage_3",
        title="Summary",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        required_prerequisite_facts=[],
        depends_on=["stage_2"],
    )
    blocker = downstream_prerequisites_blocker(
        _plan(completed, downstream),
        completed,
        hydrated_facts={},
    )
    assert blocker is None


def test_downstream_blocker_merges_hydrated_facts_with_other_deps() -> None:
    other_dep = PlanSlice(
        slice_id="stage_1",
        title="Setup",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_project"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={"research.project_id": "proj_1"},
    )
    completed = PlanSlice(
        slice_id="stage_2",
        title="Generic mutation",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={},
        required_output_facts=[],
    )
    downstream = PlanSlice(
        slice_id="stage_3",
        title="Integration",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_analysis"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        runtime_profile="backtests_integration_analysis",
        required_prerequisite_facts=["research.project_id", "research.shortlist_families"],
        required_output_facts=[],
        finalization_mode="none",
        depends_on=["stage_1", "stage_2"],
    )
    # research.project_id from stage_1, but research.shortlist_families missing
    blocker = downstream_prerequisites_blocker(
        _plan(other_dep, completed, downstream),
        completed,
        hydrated_facts={},
    )
    assert blocker is not None
    assert "research.shortlist_families" in blocker.missing_facts
    assert "research.project_id" not in blocker.missing_facts


def test_downstream_blocker_uses_resolve_dependency_for_cross_batch() -> None:
    completed = PlanSlice(
        slice_id="stage_2",
        title="Generic mutation",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={},
        required_output_facts=[],
    )
    downstream = PlanSlice(
        slice_id="stage_3",
        title="Integration",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_analysis"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        required_prerequisite_facts=["research.shortlist_families"],
        depends_on=["stage_2", "stage_0_cross"],
    )
    cross_batch_dep = PlanSlice(
        slice_id="stage_0_cross",
        title="Cross-batch setup",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_project"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={"research.shortlist_families": ["momentum", "funding"]},
    )
    blocker = downstream_prerequisites_blocker(
        _plan(completed, downstream),
        completed,
        resolve_dependency=lambda dep_id: cross_batch_dep if dep_id == "stage_0_cross" else None,
        hydrated_facts={},
    )
    assert blocker is None


# --- upstream_artifact_gate_blocker tests ---


def _stability_slice(
    *,
    slice_id: str = "stage_6",
    depends_on: list[str] | None = None,
    acceptance_predicates: list[str] | None = None,
) -> PlanSlice:
    """Build a slice that mimics compiled_plan_v1_stage_6."""
    return PlanSlice(
        slice_id=slice_id,
        title="Stability and condition analysis",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_conditions", "backtests_analysis", "backtests_runs", "research_memory"],
        evidence_requirements=[],
        policy_tags=["stability", "expensive"],
        max_turns=48,
        max_tool_calls=36,
        max_expensive_calls=12,
        parallel_slot=1,
        runtime_profile="backtests_stability_analysis",
        depends_on=depends_on or ["stage_5"],
        acceptance_contract={
            "kind": "condition_stability_analysis",
            "mode": "strict",
            "required_predicates": acceptance_predicates or ["run_set_non_empty", "each_run_proof_pass"],
        },
    )


def _upstream_with_zero_runs(
    *,
    slice_id: str = "stage_5",
    backtest_runs_found: int = 0,
    strategy_snapshots_found: int = 0,
    reason: str = "no_candidate_snapshots",
) -> PlanSlice:
    """Build an upstream slice that completed with zero backtest artifacts."""
    return PlanSlice(
        slice_id=slice_id,
        title="Standalone backtests",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs", "backtests_strategy", "research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=48,
        max_tool_calls=36,
        max_expensive_calls=12,
        parallel_slot=1,
        status="completed",
        facts={
            "backtest_runs_found": backtest_runs_found,
            "strategy_snapshots_found": strategy_snapshots_found,
            "reason": reason,
        },
    )


def test_upstream_artifact_gate_blocks_when_dependency_has_zero_runs() -> None:
    upstream = _upstream_with_zero_runs()
    downstream = _stability_slice()
    blocker = upstream_artifact_gate_blocker(_plan(upstream, downstream), downstream)
    assert blocker is not None
    assert blocker.reason_code == "upstream_zero_artifacts_gate"
    assert "stage_5" in blocker.blocking_slice_ids


def test_upstream_artifact_gate_blocks_when_dependency_has_zero_snapshots() -> None:
    upstream = PlanSlice(
        slice_id="stage_5",
        title="Standalone backtests",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={
            "backtest_runs_found": 3,
            "strategy_snapshots_found": 0,
        },
    )
    downstream = _stability_slice()
    blocker = upstream_artifact_gate_blocker(_plan(upstream, downstream), downstream)
    assert blocker is not None
    assert blocker.reason_code == "upstream_zero_artifacts_gate"


def test_upstream_artifact_gate_allows_when_dependency_has_runs() -> None:
    upstream = PlanSlice(
        slice_id="stage_5",
        title="Standalone backtests",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={
            "backtest_runs_found": 2,
            "strategy_snapshots_found": 1,
        },
    )
    downstream = _stability_slice()
    blocker = upstream_artifact_gate_blocker(_plan(upstream, downstream), downstream)
    assert blocker is None


def test_upstream_artifact_gate_allows_when_no_run_set_non_empty_predicate() -> None:
    upstream = _upstream_with_zero_runs()
    downstream = PlanSlice(
        slice_id="stage_6",
        title="Summary",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        depends_on=["stage_5"],
        acceptance_contract={
            "kind": "generic",
            "required_predicates": [],
        },
    )
    blocker = upstream_artifact_gate_blocker(_plan(upstream, downstream), downstream)
    assert blocker is None


def test_upstream_artifact_gate_allows_when_no_acceptance_contract() -> None:
    upstream = _upstream_with_zero_runs()
    downstream = PlanSlice(
        slice_id="stage_6",
        title="Summary",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        depends_on=["stage_5"],
    )
    blocker = upstream_artifact_gate_blocker(_plan(upstream, downstream), downstream)
    assert blocker is None


def test_upstream_artifact_gate_allows_when_dependency_not_completed() -> None:
    upstream = PlanSlice(
        slice_id="stage_5",
        title="Standalone backtests",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="running",
        facts={
            "backtest_runs_found": 0,
            "strategy_snapshots_found": 0,
        },
    )
    downstream = _stability_slice()
    blocker = upstream_artifact_gate_blocker(_plan(upstream, downstream), downstream)
    assert blocker is None


def test_upstream_artifact_gate_allows_when_no_dependencies() -> None:
    downstream = _stability_slice(depends_on=[])
    blocker = upstream_artifact_gate_blocker(_plan(downstream), downstream)
    assert blocker is None


def test_upstream_artifact_gate_uses_cross_plan_resolver() -> None:
    upstream = _upstream_with_zero_runs(slice_id="prev_stage_5")
    downstream = _stability_slice(depends_on=["prev_stage_5"])
    blocker = upstream_artifact_gate_blocker(
        _plan(downstream),
        downstream,
        resolve_dependency=lambda dep_id: upstream if dep_id == "prev_stage_5" else None,
    )
    assert blocker is not None
    assert "prev_stage_5" in blocker.blocking_slice_ids


def test_upstream_artifact_gate_blocks_on_reason_no_candidate_snapshots() -> None:
    upstream = PlanSlice(
        slice_id="stage_5",
        title="Standalone backtests",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={
            "reason": "no_candidate_snapshots",
        },
    )
    downstream = _stability_slice()
    blocker = upstream_artifact_gate_blocker(_plan(upstream, downstream), downstream)
    assert blocker is not None
    assert blocker.reason_code == "upstream_zero_artifacts_gate"


def test_upstream_artifact_gate_blocks_on_standalone_candidates_count_zero() -> None:
    """Worker reports standalone_candidates_count=0 (minimax-style facts)."""
    upstream = PlanSlice(
        slice_id="stage_5",
        title="Standalone backtests",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={
            "standalone_candidates_count": 0,
            "shortlist_status": "empty",
            "snapshot_id": "active-signal-v1",
        },
    )
    downstream = _stability_slice()
    blocker = upstream_artifact_gate_blocker(_plan(upstream, downstream), downstream)
    assert blocker is not None
    assert blocker.reason_code == "upstream_zero_artifacts_gate"
    assert "stage_5" in blocker.blocking_slice_ids


def test_upstream_artifact_gate_blocks_on_shortlist_status_empty() -> None:
    """Worker reports shortlist_status=empty without explicit run count."""
    upstream = PlanSlice(
        slice_id="stage_5",
        title="Standalone backtests",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={
            "shortlist_status": "empty",
        },
    )
    downstream = _stability_slice()
    blocker = upstream_artifact_gate_blocker(_plan(upstream, downstream), downstream)
    assert blocker is not None
    assert blocker.reason_code == "upstream_zero_artifacts_gate"


def test_upstream_artifact_gate_allows_when_candidates_exist() -> None:
    """Worker reports standalone_candidates_count > 0 — should NOT block."""
    upstream = PlanSlice(
        slice_id="stage_5",
        title="Standalone backtests",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={
            "standalone_candidates_count": 3,
            "shortlist_status": "populated",
            "backtest_runs_found": 2,
        },
    )
    downstream = _stability_slice()
    blocker = upstream_artifact_gate_blocker(_plan(upstream, downstream), downstream)
    assert blocker is None


def test_upstream_artifact_gate_blocks_on_empty_candidate_handles() -> None:
    """Normalized backtests.candidate_handles is empty dict."""
    upstream = PlanSlice(
        slice_id="stage_5",
        title="Standalone backtests",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["backtests_runs"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="completed",
        facts={
            "backtests.candidate_handles": {},
        },
    )
    downstream = _stability_slice()
    blocker = upstream_artifact_gate_blocker(_plan(upstream, downstream), downstream)
    assert blocker is not None
    assert blocker.reason_code == "upstream_zero_artifacts_gate"
