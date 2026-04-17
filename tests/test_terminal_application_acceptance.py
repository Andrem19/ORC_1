"""
Integration tests for TerminalActionApplicator's interaction with the
acceptance state machine.

Regression: when an agent emits a final_report with verdict=WATCHLIST but the
formal acceptance proof PASSES, the slice must transition to ``accepted_ready``
so downstream dependents are unblocked. Previously the state machine only
honoured the agent's verdict, leaving such slices in ``reported_terminal`` and
causing the orchestrator to stop with ``recoverable_blocked``.
"""

from __future__ import annotations

from typing import Any

from app.execution_models import (
    BaselineRef,
    DirectAttemptMetadata,
    ExecutionPlan,
    ExecutionTurn,
    PlanSlice,
    WorkerAction,
)
from app.services.direct_execution.acceptance import dependency_unblocked_by
from app.services.direct_execution.terminal_application import TerminalActionApplicator


class _RecordingArtifactStore:
    def __init__(self) -> None:
        self.reports: list[dict[str, Any]] = []

    def save_report(self, *, plan_id: str, slice_id: str, turn_id: str, payload: dict[str, Any]) -> str:
        self.reports.append({"plan_id": plan_id, "slice_id": slice_id, "turn_id": turn_id, "payload": payload})
        return f"artifacts/{plan_id}/{slice_id}/{turn_id}.json"


class _RecordingIncidentStore:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, *, summary: str, metadata: dict[str, Any], source: str, severity: str) -> None:
        self.records.append(
            {"summary": summary, "metadata": metadata, "source": source, "severity": severity}
        )


def _slice(slice_id: str, **overrides: Any) -> PlanSlice:
    defaults: dict[str, Any] = dict(
        slice_id=slice_id,
        title=slice_id,
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=4,
        max_expensive_calls=0,
        parallel_slot=1,
    )
    defaults.update(overrides)
    return PlanSlice(**defaults)


def _plan(slices: list[PlanSlice]) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="compiled_plan_v1_batch_2",
        goal="g",
        baseline_ref=BaselineRef(snapshot_id="snap", version=1),
        global_constraints=[],
        slices=slices,
    )


def _final_report_action(verdict: str) -> WorkerAction:
    return WorkerAction(
        action_id="action_final",
        action_type="final_report",
        summary="done",
        verdict=verdict,
        confidence=0.9,
        evidence_refs=["transcript:1:research_memory"],
        facts={
            "direct.successful_tool_count": 1,
            "direct.successful_tool_names": ["research_memory"],
            "direct.successful_mutating_tool_count": 1,
            "direct.supported_evidence_refs": ["transcript:1:research_memory"],
        },
    )


def _turn(slice_id: str, action: WorkerAction) -> ExecutionTurn:
    return ExecutionTurn(
        turn_id=f"turn_{slice_id}",
        plan_id="compiled_plan_v1_batch_2",
        slice_id=slice_id,
        worker_id="worker_1",
        turn_index=1,
        action=action,
        direct_attempt=DirectAttemptMetadata(
            provider="minimax",
            tool_call_count=3,
        ),
    )


def _build_applicator(persisted_plans: list[ExecutionPlan]) -> TerminalActionApplicator:
    return TerminalActionApplicator(
        artifact_store=_RecordingArtifactStore(),
        incident_store=_RecordingIncidentStore(),
        console_controller=None,
        known_facts_for_slice=lambda plan, slice_obj: dict(slice_obj.facts),
        required_output_facts_for_slice=lambda plan, slice_obj: list(slice_obj.required_output_facts),
        persist_plan_snapshot=persisted_plans.append,
    )


def test_apply_final_report_watchlist_with_passing_proof_unblocks_downstream() -> None:
    upstream = _slice(
        "compiled_plan_v1_stage_3_part2",
        verdict="",
        dependency_unblock_mode="accepted_only",
        watchlist_allows_unblock=False,
        max_turns=4,
    )
    upstream.acceptance_proof = {"status": "pass", "blocking_reasons": []}

    downstream = _slice(
        "compiled_plan_v1_stage_4",
        depends_on=["compiled_plan_v1_stage_3_part2"],
        dependency_unblock_mode="accepted_only",
    )

    plan = _plan([upstream, downstream])
    persisted: list[ExecutionPlan] = []
    applicator = _build_applicator(persisted)

    action = _final_report_action(verdict="WATCHLIST")
    turn = _turn(upstream.slice_id, action)

    applicator.apply_final_report(plan, upstream, turn)

    assert upstream.status == "completed"
    assert upstream.verdict == "WATCHLIST"
    assert upstream.acceptance_state == "accepted_ready"
    assert upstream.last_error == ""
    assert dependency_unblocked_by(upstream) is True


def test_apply_final_report_watchlist_without_proof_keeps_blocked() -> None:
    upstream = _slice(
        "compiled_plan_v1_stage_3_part2",
        verdict="",
        dependency_unblock_mode="accepted_only",
        watchlist_allows_unblock=False,
        max_turns=4,
    )
    # No acceptance proof attached — the slice must remain reported_terminal.

    plan = _plan([upstream])
    persisted: list[ExecutionPlan] = []
    applicator = _build_applicator(persisted)

    action = _final_report_action(verdict="WATCHLIST")
    turn = _turn(upstream.slice_id, action)

    applicator.apply_final_report(plan, upstream, turn)

    assert upstream.acceptance_state == "reported_terminal"
    assert upstream.last_error == "watchlist_not_accepted"
    assert dependency_unblocked_by(upstream) is False

def test_apply_abort_uppercase_soft_reason_code_maps_to_blocked_checkpoint() -> None:
    plan = _plan([_slice("compiled_plan_v2_stage_5")])
    slice_obj = plan.slices[0]
    persisted: list[ExecutionPlan] = []
    applicator = _build_applicator(persisted)

    action = WorkerAction(
        action_id="action_abort",
        action_type="abort",
        summary="features are not materialized yet",
        reason_code="FEATURE_DATA_UNAVAILABLE",
    )
    turn = _turn(slice_obj.slice_id, action)

    applicator.apply_abort(
        plan,
        slice_obj,
        turn,
        soft_abort_reason_codes={"feature_data_unavailable"},
    )

    assert slice_obj.status == "checkpointed"
    assert slice_obj.last_checkpoint_status == "blocked"
    assert slice_obj.last_error == "FEATURE_DATA_UNAVAILABLE"
