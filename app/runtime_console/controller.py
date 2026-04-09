"""
Broker-only Rich runtime controller.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from rich.console import Console
from rich.live import Live

from app.execution_models import BrokerHealth, ToolResultEnvelope
from app.runtime_console.models import ConsoleRuntimeState, SliceRuntimeView, ToolCallRuntimeView
from app.runtime_console.panel import build_runtime_panel


class ConsoleRuntimeController:
    """Observational live console for broker-only runtime state."""

    def __init__(
        self,
        *,
        console: Console,
        refresh_hz: float = 4.0,
        transient: bool = False,
    ) -> None:
        self.console = console
        self.refresh_interval_seconds = 1.0 / max(1.0, float(refresh_hz or 4.0))
        self.transient = transient
        self.state = ConsoleRuntimeState()
        self._live: Live | None = None
        self._stop_event: threading.Event | None = None
        self._refresh_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._live is not None:
            return
        self._live = Live(
            build_runtime_panel(self.state),
            console=self.console,
            auto_refresh=False,
            transient=self.transient,
        )
        self._live.start()
        self._stop_event = threading.Event()
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=1.0)
            self._refresh_thread = None
        self._stop_event = None
        if self._live is not None:
            self._refresh()
            self._live.stop()
            self._live = None

    def is_active(self) -> bool:
        return self._live is not None

    # ------------------------------------------------------------------
    # Lifecycle events
    # ------------------------------------------------------------------

    def on_runtime_started(self, *, goal: str) -> None:
        self.state.goal = goal
        self.state.runtime_status = "running"
        self.state.started_at_monotonic = time.monotonic()
        self._refresh()

    def on_broker_bootstrap(self, health: BrokerHealth) -> None:
        self.state.broker_status = health.status or "unknown"
        self.state.broker_summary = health.summary
        self.state.broker_warnings = list(health.warnings)
        self.state.broker_tool_count = health.tool_count
        if health.warnings:
            self.state.last_warning = health.warnings[-1]
            self.state.last_warning_at = time.monotonic()
        self._refresh()

    def on_planner_started(self, *, adapter_name: str, attempt: int, max_attempts: int) -> None:
        planner = self.state.planner
        planner.status = "running"
        planner.adapter_name = adapter_name
        planner.attempt = attempt
        planner.max_attempts = max_attempts
        planner.started_at_monotonic = time.monotonic()
        planner.last_error = ""
        self._refresh()

    def on_planner_retry(self, *, error: str, attempt: int, max_attempts: int) -> None:
        planner = self.state.planner
        planner.status = "retrying"
        planner.attempt = attempt
        planner.max_attempts = max_attempts
        planner.last_error = error
        self.state.last_warning = error
        self.state.last_warning_at = time.monotonic()
        self._refresh()

    def on_planner_finished(self, *, success: bool, error: str = "") -> None:
        planner = self.state.planner
        planner.status = "idle" if success else "error"
        planner.started_at_monotonic = 0.0
        planner.last_error = error
        if error:
            self.state.last_error = error
            self.state.last_error_at = time.monotonic()
        self._refresh()

    def on_plan_created(self, *, plan_id: str) -> None:
        self.state.active_plan_id = plan_id
        self._refresh()

    def on_slice_turn_started(
        self,
        *,
        slot: int,
        plan_id: str,
        slice_id: str,
        worker_id: str,
        turns_used: int,
        turns_total: int,
        tool_calls_used: int,
        tool_calls_total: int,
        title: str = "",
        summary: str = "",
        operation_ref: str = "",
        operation_status: str = "",
    ) -> None:
        if plan_id:
            self.state.active_plan_id = plan_id
        self.state.slices[slot] = SliceRuntimeView(
            slot=slot,
            plan_id=plan_id,
            slice_id=slice_id,
            title=title,
            worker_id=worker_id,
            status="running",
            turns_used=turns_used,
            turns_total=turns_total,
            tool_calls_used=tool_calls_used,
            tool_calls_total=tool_calls_total,
            last_summary=summary,
            active_operation_ref=operation_ref,
            active_operation_status=operation_status,
        )
        self._refresh()

    def on_tool_call_started(
        self,
        *,
        slot: int,
        plan_id: str,
        slice_id: str,
        tool_name: str,
        phase: str = "call",
    ) -> None:
        self.state.broker_calls[slot] = ToolCallRuntimeView(
            slot=slot,
            plan_id=plan_id,
            slice_id=slice_id,
            tool_name=tool_name,
            phase=phase,
            started_at_monotonic=time.monotonic(),
        )
        self._refresh()

    def on_tool_call_finished(self, *, slot: int, result: ToolResultEnvelope) -> None:
        started = self.state.broker_calls.get(slot)
        view = ToolCallRuntimeView(
            slot=slot,
            plan_id=result.plan_id,
            slice_id=result.slice_id,
            tool_name=result.tool,
            phase="done",
            retryable=result.retryable,
            response_status=result.response_status,
            tool_response_status=result.tool_response_status,
            operation_ref=result.operation_ref,
            error_class=result.error_class,
            started_at_monotonic=started.started_at_monotonic if started else 0.0,
            duration_ms=result.duration_ms,
            warning_count=len(result.warnings),
        )
        self.state.broker_calls[slot] = view
        if result.warnings:
            self.state.last_warning = result.warnings[-1]
            self.state.last_warning_at = time.monotonic()
        if not result.ok:
            self.state.last_error = result.summary or result.error_class
            self.state.last_error_at = time.monotonic()
        self._refresh()

    def on_slice_checkpoint(self, *, slot: int, summary: str, operation_ref: str = "", operation_status: str = "") -> None:
        self._update_slice(slot, status="checkpointed", summary=summary, operation_ref=operation_ref, operation_status=operation_status)

    def on_slice_completed(self, *, slot: int, summary: str) -> None:
        self._update_slice(slot, status="completed", summary=summary, operation_ref="", operation_status="")
        self.state.broker_calls.pop(slot, None)

    def on_slice_failed(self, *, slot: int, summary: str) -> None:
        self._update_slice(slot, status="failed", summary=summary, operation_ref="", operation_status="")
        self.state.last_error = summary
        self.state.last_error_at = time.monotonic()
        self.state.broker_calls.pop(slot, None)

    def on_slice_aborted(self, *, slot: int, summary: str) -> None:
        self._update_slice(slot, status="aborted", summary=summary, operation_ref="", operation_status="")
        self.state.last_warning = summary
        self.state.last_warning_at = time.monotonic()
        self.state.broker_calls.pop(slot, None)

    def on_runtime_error(self, error: str, *, total_errors: int) -> None:
        self.state.total_errors = total_errors
        self.state.last_error = error
        self.state.last_error_at = time.monotonic()
        self._refresh()

    def on_runtime_cycle(self, *, current_cycle: int, total_errors: int) -> None:
        self.state.current_cycle = current_cycle
        self.state.total_errors = total_errors
        self._refresh()

    def on_drain_requested(self) -> None:
        self.state.drain_mode = True
        self.state.last_warning = "drain_requested"
        self.state.last_warning_at = time.monotonic()
        self._refresh()

    def on_runtime_finished(self, *, reason: str, total_errors: int) -> None:
        self.state.runtime_status = "finished"
        self.state.stop_reason = reason
        self.state.total_errors = total_errors
        self._refresh()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_slice(
        self,
        slot: int,
        *,
        status: str,
        summary: str,
        operation_ref: str,
        operation_status: str,
    ) -> None:
        current = self.state.slices.get(slot)
        if current is None:
            current = SliceRuntimeView(slot=slot)
            self.state.slices[slot] = current
        current.status = status
        current.last_summary = summary
        current.active_operation_ref = operation_ref
        current.active_operation_status = operation_status
        self._refresh()

    def _refresh(self) -> None:
        if self._live is None:
            return
        try:
            self._live.update(build_runtime_panel(self.state), refresh=True)
        except Exception:
            pass

    def _refresh_loop(self) -> None:
        """Background thread: re-render at refresh_hz so elapsed counters tick continuously."""
        stop = self._stop_event
        assert stop is not None
        while not stop.is_set():
            stop.wait(timeout=self.refresh_interval_seconds)
            if self._live is not None:
                try:
                    self._live.update(build_runtime_panel(self.state), refresh=True)
                except Exception:
                    pass


def slot_for_slice(parallel_slot: int) -> int:
    return max(1, int(parallel_slot or 1))
