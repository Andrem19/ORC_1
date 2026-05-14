"""
Direct Rich runtime controller.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

from rich.console import Console
from rich.live import Live

from app.execution_models import ExecutionPlan
from app.runtime_console.models import (
    ConsoleRuntimeState,
    DirectToolRuntimeView,
    SliceRuntimeView,
    SliceTrailEntry,
    TrailBatch,
    TrailMap,
    TrailPlanGroup,
    TrailSliceSlot,
)
from app.runtime_console.panel import build_runtime_panel
from app.runtime_console.progress import build_sequence_progress, load_total_batches


class ConsoleRuntimeController:
    """Observational live console for direct runtime state."""

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
        self._sequence_batch_count_cache: dict[str, int] = {}
        self._trail_seq_counter: int = 0
        self._trail_seq_labels: dict[str, str] = {}
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

    def on_runtime_started(self, *, goal: str, plan_source: str = "") -> None:
        self.state.goal = goal
        self.state.plan_source = plan_source
        self.state.runtime_status = "running"
        self.state.started_at_monotonic = time.monotonic()
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

    def on_plan_created(self, *, plan_id: str, plan: ExecutionPlan | None = None, all_plans: list[ExecutionPlan] | None = None) -> None:
        self.state.active_plan_id = plan_id
        if plan is not None and all_plans is not None:
            self._sync_sequence_progress(plan=plan, all_plans=all_plans, active_slice_id=None)
            self._build_trail_map(plan=plan, all_plans=all_plans)
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
        plan: ExecutionPlan | None = None,
        all_plans: list[ExecutionPlan] | None = None,
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
        if plan is not None and all_plans is not None:
            self._sync_sequence_progress(plan=plan, all_plans=all_plans, active_slice_id=slice_id)
        self._update_trail_running(slot)
        self._maybe_start_wave()
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
        self.state.direct_calls[slot] = DirectToolRuntimeView(
            slot=slot,
            plan_id=plan_id,
            slice_id=slice_id,
            tool_name=tool_name,
            phase=phase,
            started_at_monotonic=time.monotonic(),
        )
        self._refresh()

    def on_slice_checkpoint(self, *, slot: int, summary: str, operation_ref: str = "", operation_status: str = "") -> None:
        self._update_slice(slot, status="checkpointed", summary=summary, operation_ref=operation_ref, operation_status=operation_status)

    def on_slice_completed(self, *, slot: int, summary: str, via: str = "direct", fallback_level: int = 0) -> None:
        self._update_slice(slot, status="completed", summary=summary, operation_ref="", operation_status="")
        self._update_trail_from_slot(slot, status="completed", execution_path=via, fallback_level=fallback_level)
        self.state.direct_calls.pop(slot, None)
        self._maybe_end_wave()

    def on_slice_failed(self, *, slot: int, summary: str, fallback_level: int = 0) -> None:
        self._update_slice(slot, status="failed", summary=summary, operation_ref="", operation_status="")
        self._update_trail_from_slot(slot, status="failed", fallback_level=fallback_level)
        self.state.last_error = summary
        self.state.last_error_at = time.monotonic()
        self.state.direct_calls.pop(slot, None)
        self._maybe_end_wave()

    def on_slice_aborted(self, *, slot: int, summary: str, fallback_level: int = 0) -> None:
        self._update_slice(slot, status="aborted", summary=summary, operation_ref="", operation_status="")
        self._update_trail_from_slot(slot, status="aborted", fallback_level=fallback_level)
        self.state.last_warning = summary
        self.state.last_warning_at = time.monotonic()
        self.state.direct_calls.pop(slot, None)
        self._maybe_end_wave()

    def on_slice_skipped(self, *, slot: int, summary: str = "") -> None:
        self._update_slice(slot, status="skipped", summary=summary, operation_ref="", operation_status="")
        self._update_trail_from_slot(slot, status="skipped")
        self._maybe_end_wave()

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

    def update_worker_id(self, slot: int, worker_id: str, *, fallback_level: int | None = None) -> None:
        """Update the displayed worker label for a running slice (e.g. after fallback)."""
        current = self.state.slices.get(slot)
        if current is None:
            return
        current.worker_id = worker_id
        if fallback_level is not None:
            self._update_running_trail_fallback_level(slot, fallback_level)
        self._refresh()

    def update_budget(self, slot: int, *, turns_used: int | None = None, tool_calls_used: int, tool_calls_total: int) -> None:
        """Live-update budget counters for a running slice."""
        current = self.state.slices.get(slot)
        if current is None:
            return
        if turns_used is not None:
            current.turns_used = turns_used
        current.tool_calls_used = tool_calls_used
        current.tool_calls_total = tool_calls_total
        self._refresh()

    # ------------------------------------------------------------------
    # Wave tracking (per-batch uptime)
    # ------------------------------------------------------------------

    def _maybe_start_wave(self) -> None:
        """Start the wave timer when the first slice starts running."""
        if self.state.current_wave_started_at == 0.0:
            self.state.current_wave_started_at = time.monotonic()

    def _maybe_end_wave(self) -> None:
        """Reset the wave timer when no slices are running."""
        if self.state.current_wave_started_at == 0.0:
            return
        any_running = any(
            s.status == "running"
            for s in self.state.slices.values()
        )
        if not any_running:
            self.state.current_wave_started_at = 0.0

    # ------------------------------------------------------------------
    # Trail map
    # ------------------------------------------------------------------

    def _get_trail_label(self, sequence_id: str) -> str:
        if sequence_id in self._trail_seq_labels:
            return self._trail_seq_labels[sequence_id]
        # Extract real version from sequence_id like "compiled_plan_v2"
        match = re.search(r"_v(\d+)", sequence_id)
        if match:
            label = f"v{match.group(1)}"
        else:
            self._trail_seq_counter += 1
            label = f"v{self._trail_seq_counter}"
        self._trail_seq_labels[sequence_id] = label
        return label

    def _build_trail_map(self, *, plan: ExecutionPlan, all_plans: list[ExecutionPlan]) -> None:
        """Build/update the trail map from all known plans, preserving outcomes."""
        # Collect existing outcomes: (plan_id, slice_id) → (status, execution_path, fallback_level)
        outcomes: dict[tuple[str, str], tuple[str, str, int]] = {}
        for pg in self.state.trail_map.plans:
            for batch in pg.batches:
                for slot in batch.slices:
                    if slot.status != "pending" and slot.slice_id:
                        outcomes[(batch.plan_id, slot.slice_id)] = (slot.status, slot.execution_path, slot.fallback_level)

        # Group all plans by source_sequence_id
        sequences: dict[str, list[ExecutionPlan]] = {}
        for p in all_plans:
            seq_id = str(p.source_sequence_id or "").strip()
            if seq_id:
                sequences.setdefault(seq_id, []).append(p)

        # Build new map preserving outcomes
        _TERMINAL = {"completed", "failed", "aborted", "skipped"}
        new_plans: list[TrailPlanGroup] = []
        for seq_id in sorted(sequences.keys()):
            plans_in_seq = sorted(sequences[seq_id], key=lambda p: int(p.sequence_batch_index or 0))
            batches: list[TrailBatch] = []
            for p in plans_in_seq:
                slices: list[TrailSliceSlot] = []
                for s in p.slices:
                    key = (p.plan_id, s.slice_id)
                    if key in outcomes:
                        st, ep, fl = outcomes[key]
                        slices.append(TrailSliceSlot(slice_id=s.slice_id, status=st, execution_path=ep, fallback_level=fl))
                    elif s.status in _TERMINAL:
                        slices.append(TrailSliceSlot(slice_id=s.slice_id, status=s.status, execution_path="direct"))
                    else:
                        slices.append(TrailSliceSlot(slice_id=s.slice_id))
                batches.append(TrailBatch(plan_id=p.plan_id, slices=slices))
            new_plans.append(TrailPlanGroup(
                label=self._get_trail_label(seq_id),
                sequence_id=seq_id,
                batches=batches,
            ))

        self.state.trail_map.plans = new_plans

    def _update_trail_running(self, slot: int) -> None:
        """Mark the trail slot as running when a slice starts execution."""
        existing = self.state.slices.get(slot)
        if existing is None:
            return
        plan_id = existing.plan_id
        slice_id = existing.slice_id
        if not plan_id or not slice_id:
            return
        for pg in self.state.trail_map.plans:
            for batch in pg.batches:
                if batch.plan_id == plan_id:
                    for ts in batch.slices:
                        if ts.slice_id == slice_id:
                            ts.status = "running"
                            return

    def _update_running_trail_fallback_level(self, slot: int, fallback_level: int) -> None:
        """Attach fallback level to an already-running trail slot after provider switch."""
        existing = self.state.slices.get(slot)
        if existing is None:
            return
        plan_id = existing.plan_id
        slice_id = existing.slice_id
        if not plan_id or not slice_id:
            return
        for pg in self.state.trail_map.plans:
            for batch in pg.batches:
                if batch.plan_id != plan_id:
                    continue
                for ts in batch.slices:
                    if ts.slice_id != slice_id or ts.status != "running":
                        continue
                    ts.fallback_level = max(0, int(fallback_level or 0))
                    return

    def _update_trail_from_slot(self, slot: int, status: str, execution_path: str = "direct", fallback_level: int = 0) -> None:
        """Find the trail slot from a parallel slot number and update it."""
        existing = self.state.slices.get(slot)
        if existing is None:
            return
        plan_id = existing.plan_id
        slice_id = existing.slice_id
        if not plan_id or not slice_id:
            return
        self.state.slice_trail.append(
            SliceTrailEntry(
                status=status,
                execution_path=execution_path,
                title=existing.title,
                summary=existing.last_summary,
                fallback_level=fallback_level,
            )
        )
        for pg in self.state.trail_map.plans:
            for batch in pg.batches:
                if batch.plan_id == plan_id:
                    for ts in batch.slices:
                        if ts.slice_id == slice_id:
                            ts.status = status
                            ts.execution_path = execution_path
                            ts.fallback_level = fallback_level
                            return

    def sync_sequence_progress(
        self,
        *,
        plan: ExecutionPlan,
        all_plans: list[ExecutionPlan],
        active_slice_id: str | None = None,
    ) -> None:
        self._sync_sequence_progress(plan=plan, all_plans=all_plans, active_slice_id=active_slice_id)
        self._build_trail_map(plan=plan, all_plans=all_plans)
        self._refresh()

    def _sync_sequence_progress(
        self,
        *,
        plan: ExecutionPlan,
        all_plans: list[ExecutionPlan],
        active_slice_id: str | None,
    ) -> None:
        total_batches = load_total_batches(
            plan=plan,
            cache=self._sequence_batch_count_cache,
        )
        self.state.sequence_progress = build_sequence_progress(
            plan=plan,
            all_plans=all_plans,
            active_slice_id=active_slice_id,
            total_batches=total_batches,
        )

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
