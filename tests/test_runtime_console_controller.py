from __future__ import annotations

import time

from rich.console import Console

from app.execution_models import BaselineRef, ExecutionPlan, PlanSlice
from app.runtime_console.controller import ConsoleRuntimeController


def _make_controller() -> ConsoleRuntimeController:
    return ConsoleRuntimeController(console=Console(record=True))


def _make_slice(index: int, slot: int, *, status: str = "pending") -> PlanSlice:
    return PlanSlice(
        slice_id=f"slice_{index}",
        title=f"Task {index}",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["research_search"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=4,
        max_tool_calls=4,
        max_expensive_calls=0,
        parallel_slot=slot,
        status=status,
    )


def _make_plan(plan_id: str, *, seq_id: str = "seq_1", batch_index: int = 1, slice_count: int = 3) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        goal="test",
        baseline_ref=BaselineRef(snapshot_id="s", version=1),
        global_constraints=[],
        slices=[_make_slice(i, slot=(i % 3) + 1) for i in range(slice_count)],
        source_sequence_id=seq_id,
        sequence_batch_index=batch_index,
    )


def test_console_controller_tracks_direct_slice_and_finish() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="test goal")
        controller.on_slice_turn_started(
            slot=1,
            plan_id="plan_1",
            slice_id="slice_a",
            title="Validate data quality",
            worker_id="worker-1",
            turns_used=0,
            turns_total=4,
            tool_calls_used=0,
            tool_calls_total=2,
            summary="starting",
        )
        controller.on_slice_checkpoint(slot=1, summary="catalog ready")
        controller.on_slice_completed(slot=1, summary="done")
        controller.on_runtime_finished(reason="goal_reached", total_errors=0)
    finally:
        controller.stop()

    assert controller.state.runtime_status == "finished"
    assert controller.state.active_plan_id == "plan_1"
    assert controller.state.slices[1].status == "completed"
    assert controller.state.slices[1].title == "Validate data quality"
    assert controller.state.stop_reason == "goal_reached"


def test_console_controller_marks_drain_and_error() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="test")
        controller.on_drain_requested()
        controller.on_runtime_error("direct failed", total_errors=2)
    finally:
        controller.stop()

    assert controller.state.drain_mode is True
    assert controller.state.total_errors == 2
    assert controller.state.last_error == "direct failed"


def test_on_runtime_started_sets_started_at_monotonic() -> None:
    controller = _make_controller()
    controller.start()
    try:
        before = time.monotonic()
        controller.on_runtime_started(goal="uptime test")
        after = time.monotonic()
    finally:
        controller.stop()

    assert before <= controller.state.started_at_monotonic <= after


def test_on_runtime_started_stores_plan_source() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="test", plan_source="compiled_raw")
    finally:
        controller.stop()

    assert controller.state.plan_source == "compiled_raw"


# ---------------------------------------------------------------------------
# Wave uptime tracking
# ---------------------------------------------------------------------------

def test_wave_starts_on_first_slice_turn() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="test")
        assert controller.state.current_wave_started_at == 0.0

        before = time.monotonic()
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="s1", worker_id="qwen_cli",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        after = time.monotonic()
        assert before <= controller.state.current_wave_started_at <= after
    finally:
        controller.stop()


def test_wave_resets_when_all_slices_complete() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="test")
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="s1", worker_id="qwen_cli",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        assert controller.state.current_wave_started_at > 0.0

        controller.on_slice_completed(slot=1, summary="done")
        assert controller.state.current_wave_started_at == 0.0
    finally:
        controller.stop()


def test_wave_stays_active_with_parallel_slices() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="test")
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="s1", worker_id="qwen_cli",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        controller.on_slice_turn_started(
            slot=2, plan_id="p1", slice_id="s2", worker_id="qwen_cli",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        wave_start = controller.state.current_wave_started_at
        assert wave_start > 0.0

        # Complete only one slice — wave should still be active
        controller.on_slice_completed(slot=1, summary="done")
        assert controller.state.current_wave_started_at == wave_start

        # Complete last slice — wave should reset
        controller.on_slice_completed(slot=2, summary="done")
        assert controller.state.current_wave_started_at == 0.0
    finally:
        controller.stop()


def test_wave_resets_on_slice_failed() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="test")
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="s1", worker_id="qwen_cli",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        controller.on_slice_failed(slot=1, summary="error")
        assert controller.state.current_wave_started_at == 0.0
    finally:
        controller.stop()


def test_wave_starts_again_after_reset() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="test")
        # First wave
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="s1", worker_id="qwen_cli",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        first_wave = controller.state.current_wave_started_at
        controller.on_slice_completed(slot=1, summary="done")
        assert controller.state.current_wave_started_at == 0.0

        # Second wave — new timestamp
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="s2", worker_id="qwen_cli",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        second_wave = controller.state.current_wave_started_at
        assert second_wave >= first_wave
    finally:
        controller.stop()


# ---------------------------------------------------------------------------
# Trail map building from plans
# ---------------------------------------------------------------------------

def test_trail_map_builds_from_on_plan_created() -> None:
    plan1 = _make_plan("p1", seq_id="seq_alpha", batch_index=1, slice_count=3)
    plan2 = _make_plan("p2", seq_id="seq_alpha", batch_index=2, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan1, all_plans=[plan1])
        controller.on_plan_created(plan_id="p2", plan=plan2, all_plans=[plan1, plan2])
    finally:
        controller.stop()

    trail = controller.state.trail_map
    assert len(trail.plans) == 1  # one sequence
    assert trail.plans[0].label == "v1"
    assert len(trail.plans[0].batches) == 2
    assert len(trail.plans[0].batches[0].slices) == 3
    assert len(trail.plans[0].batches[1].slices) == 2
    # All pending initially
    for batch in trail.plans[0].batches:
        for slot in batch.slices:
            assert slot.status == "pending"


def test_trail_map_builds_multiple_sequences() -> None:
    plan_a1 = _make_plan("a1", seq_id="seq_a", batch_index=1, slice_count=2)
    plan_b1 = _make_plan("b1", seq_id="seq_b", batch_index=1, slice_count=3)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="a1", plan=plan_a1, all_plans=[plan_a1])
        controller.on_plan_created(plan_id="b1", plan=plan_b1, all_plans=[plan_a1, plan_b1])
    finally:
        controller.stop()

    trail = controller.state.trail_map
    assert len(trail.plans) == 2
    assert trail.plans[0].label == "v1"
    assert trail.plans[0].sequence_id == "seq_a"
    assert trail.plans[1].label == "v2"
    assert trail.plans[1].sequence_id == "seq_b"


def test_trail_map_updates_on_slice_completed() -> None:
    plan = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan, all_plans=[plan])
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="slice_0", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        controller.on_slice_completed(slot=1, summary="done", via="direct")
    finally:
        controller.stop()

    slot = controller.state.trail_map.plans[0].batches[0].slices[0]
    assert slot.status == "completed"
    assert slot.execution_path == "direct"


def test_trail_map_updates_on_slice_failed() -> None:
    plan = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan, all_plans=[plan])
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="slice_0", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        controller.on_slice_failed(slot=1, summary="crash")
    finally:
        controller.stop()

    assert controller.state.trail_map.plans[0].batches[0].slices[0].status == "failed"


def test_trail_map_updates_on_slice_aborted() -> None:
    plan = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan, all_plans=[plan])
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="slice_0", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        controller.on_slice_aborted(slot=1, summary="timeout")
    finally:
        controller.stop()

    assert controller.state.trail_map.plans[0].batches[0].slices[0].status == "aborted"


def test_trail_map_updates_on_slice_skipped() -> None:
    plan = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan, all_plans=[plan])
        controller.on_slice_turn_started(
            slot=2, plan_id="p1", slice_id="slice_1", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        controller.on_slice_skipped(slot=2, summary="dependency not met")
    finally:
        controller.stop()

    assert controller.state.trail_map.plans[0].batches[0].slices[1].status == "skipped"


def test_trail_map_preserves_outcomes_on_rebuild() -> None:
    """When new plans are added, existing slice outcomes must be preserved."""
    plan1 = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=2)
    plan2 = _make_plan("p2", seq_id="seq_1", batch_index=2, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        # Create first plan and complete a slice
        controller.on_plan_created(plan_id="p1", plan=plan1, all_plans=[plan1])
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="slice_0", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        controller.on_slice_completed(slot=1, summary="done")

        # Add second plan — rebuilds map
        controller.on_plan_created(plan_id="p2", plan=plan2, all_plans=[plan1, plan2])
    finally:
        controller.stop()

    # First slice must still show completed
    slot = controller.state.trail_map.plans[0].batches[0].slices[0]
    assert slot.status == "completed"
    # Second batch should now exist
    assert len(controller.state.trail_map.plans[0].batches) == 2


def test_trail_map_mixed_sequence() -> None:
    """Full scenario: 2 plans, 2 batches each, mixed outcomes."""
    p1 = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=3)
    p2 = _make_plan("p2", seq_id="seq_1", batch_index=2, slice_count=3)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=p1, all_plans=[p1, p2])
        # Complete slice_0, fail slice_1, skip slice_2
        for slot_num, sid, event in [
            (1, "slice_0", "completed"),
            (2, "slice_1", "failed"),
            (3, "slice_2", "skipped"),
        ]:
            controller.on_slice_turn_started(
                slot=slot_num, plan_id="p1", slice_id=sid, worker_id="w1",
                turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
            )
            if event == "completed":
                controller.on_slice_completed(slot=slot_num, summary="ok")
            elif event == "failed":
                controller.on_slice_failed(slot=slot_num, summary="err")
            else:
                controller.on_slice_skipped(slot=slot_num, summary="skip")
    finally:
        controller.stop()

    batch = controller.state.trail_map.plans[0].batches[0]
    assert batch.slices[0].status == "completed"
    assert batch.slices[1].status == "failed"
    assert batch.slices[2].status == "skipped"


# ---------------------------------------------------------------------------
# Fallback level propagation through trail
# ---------------------------------------------------------------------------

def test_trail_map_completed_with_fallback_level() -> None:
    plan = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan, all_plans=[plan])
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="slice_0", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        controller.on_slice_completed(slot=1, summary="done via fallback", fallback_level=1)
    finally:
        controller.stop()

    slot = controller.state.trail_map.plans[0].batches[0].slices[0]
    assert slot.status == "completed"
    assert slot.fallback_level == 1


def test_trail_map_failed_with_fallback_level() -> None:
    plan = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan, all_plans=[plan])
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="slice_0", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        controller.on_slice_failed(slot=1, summary="all fallbacks failed", fallback_level=2)
    finally:
        controller.stop()

    slot = controller.state.trail_map.plans[0].batches[0].slices[0]
    assert slot.status == "failed"
    assert slot.fallback_level == 2


def test_trail_map_aborted_with_fallback_level() -> None:
    plan = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan, all_plans=[plan])
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="slice_0", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        controller.on_slice_aborted(slot=1, summary="aborted after fallback", fallback_level=1)
    finally:
        controller.stop()

    slot = controller.state.trail_map.plans[0].batches[0].slices[0]
    assert slot.status == "aborted"
    assert slot.fallback_level == 1


def test_trail_map_preserves_fallback_level_on_rebuild() -> None:
    """Fallback level must survive a trail map rebuild."""
    plan1 = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=2)
    plan2 = _make_plan("p2", seq_id="seq_1", batch_index=2, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan1, all_plans=[plan1])
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="slice_0", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        controller.on_slice_completed(slot=1, summary="done via fb2", fallback_level=2)

        # Rebuild map — fallback_level must survive
        controller.on_plan_created(plan_id="p2", plan=plan2, all_plans=[plan1, plan2])
    finally:
        controller.stop()

    slot = controller.state.trail_map.plans[0].batches[0].slices[0]
    assert slot.status == "completed"
    assert slot.fallback_level == 2


def test_trail_map_running_on_slice_turn_started() -> None:
    """Trail slot should show 'running' when on_slice_turn_started fires."""
    plan = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=3)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan, all_plans=[plan])
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="slice_0", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
    finally:
        controller.stop()

    slot = controller.state.trail_map.plans[0].batches[0].slices[0]
    assert slot.status == "running"
    # Other slices still pending
    assert controller.state.trail_map.plans[0].batches[0].slices[1].status == "pending"


def test_trail_map_running_then_completed() -> None:
    """Trail slot: pending -> running -> completed transition."""
    plan = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan, all_plans=[plan])
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="slice_0", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        assert controller.state.trail_map.plans[0].batches[0].slices[0].status == "running"

        controller.on_slice_completed(slot=1, summary="done", fallback_level=1)
    finally:
        controller.stop()

    slot = controller.state.trail_map.plans[0].batches[0].slices[0]
    assert slot.status == "completed"
    assert slot.fallback_level == 1


def test_trail_map_running_preserved_on_rebuild() -> None:
    """Running status must survive a trail map rebuild."""
    plan1 = _make_plan("p1", seq_id="seq_1", batch_index=1, slice_count=2)
    plan2 = _make_plan("p2", seq_id="seq_1", batch_index=2, slice_count=2)
    controller = _make_controller()
    controller.start()
    try:
        controller.on_plan_created(plan_id="p1", plan=plan1, all_plans=[plan1])
        controller.on_slice_turn_started(
            slot=1, plan_id="p1", slice_id="slice_0", worker_id="w1",
            turns_used=0, turns_total=4, tool_calls_used=0, tool_calls_total=2,
        )
        # Rebuild — running status must survive
        controller.on_plan_created(plan_id="p2", plan=plan2, all_plans=[plan1, plan2])
    finally:
        controller.stop()

    slot = controller.state.trail_map.plans[0].batches[0].slices[0]
    assert slot.status == "running"
