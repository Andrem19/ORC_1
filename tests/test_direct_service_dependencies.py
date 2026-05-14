from __future__ import annotations

import asyncio

from types import SimpleNamespace

from app.execution_models import BaselineRef, ExecutionPlan, ExecutionStateV2, PlanSlice
from app.services.direct_execution.service import DirectExecutionService
from tests.mcp_catalog_fixtures import make_catalog_snapshot


def _slice(
    slice_id: str,
    *,
    status: str = "pending",
    depends_on: list[str] | None = None,
    facts: dict | None = None,
    acceptance_state: str = "accepted_ready",
    verdict: str = "COMPLETE",
) -> PlanSlice:
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
        acceptance_state=acceptance_state,
        depends_on=list(depends_on or []),
        facts=dict(facts or {}),
        verdict=verdict,
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
    stage3 = _slice("compiled_plan_v1_stage_3", status="completed", acceptance_state="accepted_ready")
    plan1 = _plan("compiled_plan_v1_batch_1", [stage3], status="completed")
    stage4 = _slice("compiled_plan_v1_stage_4", status="pending", depends_on=["compiled_plan_v1_stage_3"])
    plan2 = _plan("compiled_plan_v1_batch_2", [stage4], status="running")

    service = DirectExecutionService.__new__(DirectExecutionService)
    service.state = ExecutionStateV2(goal="g", plans=[plan1, plan2])
    service.catalog_snapshot = make_catalog_snapshot()

    assert service._dependencies_satisfied(plan2, stage4) is True


def test_known_facts_for_slice_merges_cross_batch_dependency_facts() -> None:
    stage3 = _slice(
        "compiled_plan_v1_stage_3",
        status="completed",
        acceptance_state="accepted_ready",
        facts={"research.project_id": "proj_123", "research.shortlist_families": ["alpha"]},
    )
    plan1 = _plan("compiled_plan_v1_batch_1", [stage3], status="completed")
    stage4 = _slice("compiled_plan_v1_stage_4", status="pending", depends_on=["compiled_plan_v1_stage_3"])
    plan2 = _plan("compiled_plan_v1_batch_2", [stage4], status="running")

    service = DirectExecutionService.__new__(DirectExecutionService)
    service.state = ExecutionStateV2(goal="g", plans=[plan1, plan2])
    service.catalog_snapshot = make_catalog_snapshot()

    facts = service._known_facts_for_slice(plan2, stage4)
    assert facts["compiled_plan_v1_stage_3.research.project_id"] == "proj_123"
    assert facts["compiled_plan_v1_stage_3.research.shortlist_families"] == ["alpha"]


# ---------------------------------------------------------------------------
# Blocked-slice retry tests
# ---------------------------------------------------------------------------


def _make_service(*, max_blocked_retries: int = 2) -> DirectExecutionService:
    service = DirectExecutionService.__new__(DirectExecutionService)
    service.state = ExecutionStateV2(goal="g", plans=[])
    service.catalog_snapshot = make_catalog_snapshot()
    service.config = SimpleNamespace(direct_execution=SimpleNamespace(max_blocked_retries=max_blocked_retries))
    service.plan_source = SimpleNamespace(
        mark_plan_failed=lambda *a, **kw: None,
        mark_plan_complete=lambda *a, **kw: None,
    )
    service._persist_plan_snapshot = lambda plan: None  # no-op for tests
    service._maybe_send_sequence_report = lambda plan: asyncio.sleep(0)
    service.console_controller = None
    return service


def _blocked_slice(slice_id: str, *, blocked_retry_count: int = 0, turn_count: int = 0, max_turns: int = 5) -> PlanSlice:
    return PlanSlice(
        slice_id=slice_id,
        title=slice_id,
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_map"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=max_turns,
        max_tool_calls=1,
        max_expensive_calls=0,
        parallel_slot=1,
        status="checkpointed",
        last_checkpoint_status="blocked",
        last_error="zero_tool_calls",
        blocked_retry_count=blocked_retry_count,
        turn_count=turn_count,
    )


def test_retry_blocked_slices_resets_status_to_pending() -> None:
    slice_a = _blocked_slice("s1", blocked_retry_count=0)
    plan = _plan("p1", [slice_a])
    service = _make_service(max_blocked_retries=2)

    retried = service._retry_blocked_slices(plan)

    assert len(retried) == 1
    assert retried[0].slice_id == "s1"
    assert slice_a.status == "pending"
    assert slice_a.blocked_retry_count == 1
    assert slice_a.last_checkpoint_status == ""
    assert slice_a.last_error == ""


def test_retry_blocked_slices_skips_when_retry_budget_exhausted() -> None:
    slice_a = _blocked_slice("s1", blocked_retry_count=2)
    plan = _plan("p1", [slice_a])
    service = _make_service(max_blocked_retries=2)

    retried = service._retry_blocked_slices(plan)

    assert len(retried) == 0
    assert slice_a.status == "checkpointed"


def test_retry_blocked_slices_skips_when_turn_budget_exhausted() -> None:
    slice_a = _blocked_slice("s1", blocked_retry_count=0, turn_count=5, max_turns=5)
    plan = _plan("p1", [slice_a])
    service = _make_service(max_blocked_retries=2)

    retried = service._retry_blocked_slices(plan)

    assert len(retried) == 0
    assert slice_a.status == "checkpointed"


def test_retry_blocked_slices_skips_non_blocked_slices() -> None:
    slice_completed = _slice("s1", status="completed")
    slice_running = _slice("s2", status="running")
    plan = _plan("p1", [slice_completed, slice_running])
    service = _make_service()

    retried = service._retry_blocked_slices(plan)

    assert len(retried) == 0


def test_retry_blocked_slices_increments_retry_count_each_call() -> None:
    slice_a = _blocked_slice("s1", blocked_retry_count=0)
    plan = _plan("p1", [slice_a])
    service = _make_service(max_blocked_retries=3)

    service._retry_blocked_slices(plan)
    assert slice_a.blocked_retry_count == 1
    assert slice_a.status == "pending"

    # Simulate it getting blocked again
    slice_a.status = "checkpointed"
    slice_a.last_checkpoint_status = "blocked"

    service._retry_blocked_slices(plan)
    assert slice_a.blocked_retry_count == 2
    assert slice_a.status == "pending"


def test_retry_blocked_slices_noop_when_max_retries_zero() -> None:
    slice_a = _blocked_slice("s1", blocked_retry_count=0)
    plan = _plan("p1", [slice_a])
    service = _make_service(max_blocked_retries=0)

    retried = service._retry_blocked_slices(plan)

    assert len(retried) == 0
    assert slice_a.status == "checkpointed"


# ---------------------------------------------------------------------------
# Cascade abort with retry budget tests
# ---------------------------------------------------------------------------


def test_blocked_checkpoint_does_not_cascade_abort_when_retries_remaining() -> None:
    """Blocked slice with retries left should NOT abort downstream slices."""
    upstream = _blocked_slice("s1", blocked_retry_count=0)
    downstream = _slice("s2", status="pending", depends_on=["s1"])
    plan = _plan("p1", [upstream, downstream])
    service = _make_service(max_blocked_retries=2)

    changed = service._abort_dependency_blocked_slices(plan)

    assert not changed
    assert downstream.status == "pending"


def test_blocked_checkpoint_cascades_abort_when_retries_exhausted() -> None:
    """Blocked slice with exhausted retries SHOULD abort downstream slices."""
    upstream = _blocked_slice("s1", blocked_retry_count=2)
    downstream = _slice("s2", status="pending", depends_on=["s1"])
    plan = _plan("p1", [upstream, downstream])
    service = _make_service(max_blocked_retries=2)

    changed = service._abort_dependency_blocked_slices(plan)

    assert changed
    assert downstream.status == "aborted"
    assert downstream.last_error == "dependency_blocked"


def test_completed_watchlist_without_acceptance_does_not_unblock_downstream() -> None:
    upstream = _slice(
        "s1",
        status="completed",
        acceptance_state="reported_terminal",
        verdict="WATCHLIST",
    )
    downstream = _slice("s2", status="pending", depends_on=["s1"], acceptance_state="pending", verdict="")
    plan = _plan("p1", [upstream, downstream])
    service = _make_service(max_blocked_retries=2)

    assert service._dependencies_satisfied(plan, downstream) is False


def test_completed_watchlist_without_acceptance_cascades_abort_downstream() -> None:
    upstream = _slice(
        "s1",
        status="completed",
        acceptance_state="reported_terminal",
        verdict="WATCHLIST",
    )
    downstream = _slice("s2", status="pending", depends_on=["s1"], acceptance_state="pending", verdict="")
    plan = _plan("p1", [upstream, downstream])
    service = _make_service(max_blocked_retries=2)

    changed = service._abort_dependency_blocked_slices(plan)

    assert changed is True
    assert downstream.status == "aborted"
    assert downstream.dependency_blocker_reason_code == "watchlist_not_accepted"


def test_failed_slice_always_cascades_abort_regardless_of_retry() -> None:
    """A truly failed slice (not blocked) should always cascade abort."""
    upstream = _slice("s1", status="failed")
    downstream = _slice("s2", status="pending", depends_on=["s1"])
    plan = _plan("p1", [upstream, downstream])
    service = _make_service(max_blocked_retries=99)

    changed = service._abort_dependency_blocked_slices(plan)

    assert changed
    assert downstream.status == "aborted"


# ---------------------------------------------------------------------------
# Finalize plan with blocked-slice retry budget tests
# ---------------------------------------------------------------------------


def test_finalize_plan_does_not_stop_when_blocked_slices_have_retries_left() -> None:
    """Plan should stay running when blocked slices still have retries available."""
    completed = _slice("s1", status="completed")
    blocked = _blocked_slice("s2", blocked_retry_count=0)
    plan = _plan("p1", [completed, blocked])
    service = _make_service(max_blocked_retries=2)
    service.plan_source = SimpleNamespace(mark_plan_failed=lambda *a, **kw: None)

    asyncio.run(service._finalize_plan_if_ready(plan))

    assert plan.status == "running"


def test_finalize_plan_stops_when_all_blocked_slices_exhaust_retries() -> None:
    """Plan should finalize as 'stopped' when blocked slices have exhausted retries."""
    completed = _slice("s1", status="completed")
    blocked = _blocked_slice("s2", blocked_retry_count=2)
    plan = _plan("p1", [completed, blocked])
    service = _make_service(max_blocked_retries=2)
    service.plan_source = SimpleNamespace(mark_plan_failed=lambda *a, **kw: None)

    asyncio.run(service._finalize_plan_if_ready(plan))

    assert plan.status == "stopped"


def test_finalize_plan_stops_when_slice_is_completed_but_not_accepted() -> None:
    completed_unaccepted = _slice(
        "s1",
        status="completed",
        acceptance_state="reported_terminal",
        verdict="WATCHLIST",
    )
    plan = _plan("p1", [completed_unaccepted])
    service = _make_service(max_blocked_retries=2)
    service.plan_source = SimpleNamespace(mark_plan_failed=lambda *a, **kw: None, mark_plan_complete=lambda *a, **kw: None)

    asyncio.run(service._finalize_plan_if_ready(plan))

    assert plan.status == "stopped"
    assert plan.last_error == "watchlist_not_accepted"


def test_finalize_plan_completes_when_all_slices_completed() -> None:
    """All completed slices → plan completed (unchanged behavior)."""
    s1 = _slice("s1", status="completed")
    s2 = _slice("s2", status="completed")
    plan = _plan("p1", [s1, s2])
    service = _make_service()
    service.plan_source = SimpleNamespace(mark_plan_complete=lambda *a, **kw: None)

    asyncio.run(service._finalize_plan_if_ready(plan))

    assert plan.status == "completed"
