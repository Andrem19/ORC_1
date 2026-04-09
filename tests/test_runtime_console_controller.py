from __future__ import annotations

import time

from rich.console import Console

from app.execution_models import BrokerHealth, ToolResultEnvelope
from app.runtime_console.controller import ConsoleRuntimeController


def _make_controller() -> ConsoleRuntimeController:
    return ConsoleRuntimeController(console=Console(record=True))


def test_console_controller_tracks_planner_slice_and_finish() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="test goal")
        controller.on_broker_bootstrap(
            BrokerHealth(status="degraded", summary="partial bootstrap", warnings=["system_health:empty_payload"], tool_count=42)
        )
        controller.on_planner_started(adapter_name="claude_planner_cli", attempt=1, max_attempts=3)
        controller.on_planner_retry(error="temporary timeout", attempt=1, max_attempts=3)
        controller.on_planner_finished(success=True)
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
        controller.on_tool_call_started(slot=1, plan_id="plan_1", slice_id="slice_a", tool_name="events")
        controller.on_tool_call_finished(
            slot=1,
            result=ToolResultEnvelope(
                call_id="tool_1",
                tool="events",
                ok=True,
                retryable=False,
                duration_ms=12,
                summary="catalog ready",
                response_status="completed",
                tool_response_status="ok",
                plan_id="plan_1",
                slice_id="slice_a",
            ),
        )
        controller.on_slice_checkpoint(slot=1, summary="catalog ready")
        controller.on_slice_completed(slot=1, summary="done")
        controller.on_runtime_finished(reason="goal_reached", total_errors=0)
    finally:
        controller.stop()

    assert controller.state.runtime_status == "finished"
    assert controller.state.broker_status == "degraded"
    assert controller.state.broker_tool_count == 42
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
        controller.on_runtime_error("broker failed", total_errors=2)
    finally:
        controller.stop()

    assert controller.state.drain_mode is True
    assert controller.state.total_errors == 2
    assert controller.state.last_error == "broker failed"


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


def test_on_slice_failed_sets_error_timestamp() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="g")
        before = time.monotonic()
        controller.on_slice_failed(slot=1, summary="worker crashed")
        after = time.monotonic()
    finally:
        controller.stop()

    assert controller.state.last_error == "worker crashed"
    assert before <= controller.state.last_error_at <= after


def test_on_slice_aborted_sets_warning_timestamp() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="g")
        before = time.monotonic()
        controller.on_slice_aborted(slot=1, summary="budget exhausted")
        after = time.monotonic()
    finally:
        controller.stop()

    assert controller.state.last_warning == "budget exhausted"
    assert before <= controller.state.last_warning_at <= after


def test_on_drain_requested_sets_warning_timestamp() -> None:
    controller = _make_controller()
    controller.start()
    try:
        before = time.monotonic()
        controller.on_drain_requested()
        after = time.monotonic()
    finally:
        controller.stop()

    assert controller.state.drain_mode is True
    assert before <= controller.state.last_warning_at <= after


def test_tool_call_failure_sets_error_timestamp() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_runtime_started(goal="g")
        controller.on_tool_call_started(slot=1, plan_id="p1", slice_id="s1", tool_name="events")
        before = time.monotonic()
        controller.on_tool_call_finished(
            slot=1,
            result=ToolResultEnvelope(
                call_id="c1",
                tool="events",
                ok=False,
                retryable=True,
                duration_ms=50,
                summary="connection refused",
                error_class="BrokerConnectionError",
                plan_id="p1",
                slice_id="s1",
            ),
        )
        after = time.monotonic()
    finally:
        controller.stop()

    assert controller.state.last_error == "connection refused"
    assert before <= controller.state.last_error_at <= after


def test_broker_bootstrap_sets_tool_count_and_warning_timestamp() -> None:
    controller = _make_controller()
    controller.start()
    try:
        before = time.monotonic()
        controller.on_broker_bootstrap(
            BrokerHealth(status="healthy", summary="ok", warnings=["minor degradation"], tool_count=55)
        )
        after = time.monotonic()
    finally:
        controller.stop()

    assert controller.state.broker_tool_count == 55
    assert controller.state.last_warning == "minor degradation"
    assert before <= controller.state.last_warning_at <= after


def test_controller_stop_joins_refresh_thread() -> None:
    """stop() must not hang — background thread terminates cleanly."""
    controller = _make_controller()
    controller.start()
    assert controller._refresh_thread is not None
    assert controller._refresh_thread.is_alive()
    controller.stop()
    # After stop the thread should be gone
    assert controller._refresh_thread is None
    assert controller._live is None


def test_on_slice_turn_started_without_title_defaults_empty() -> None:
    controller = _make_controller()
    controller.start()
    try:
        controller.on_slice_turn_started(
            slot=2, plan_id="p", slice_id="s", worker_id="w",
            turns_used=1, turns_total=5, tool_calls_used=0, tool_calls_total=10,
        )
    finally:
        controller.stop()

    assert controller.state.slices[2].title == ""
    assert controller.state.slices[2].slice_id == "s"
