"""Task health monitoring — timeouts, silent workers, MCP health."""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.models import (
    OrchestratorEvent,
    TaskStatus,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("orchestrator.plan")


class TaskHealthMixin:
    """Monitors task health: timeouts, silent workers, and periodic MCP checks."""

    # ---------------------------------------------------------------
    # Timeout enforcement
    # ---------------------------------------------------------------

    def _check_timeouts(self) -> None:
        """Kill tasks that exceeded the timeout threshold."""
        timeout_seconds = getattr(
            self.config, "plan_task_timeout_seconds", 600,
        )
        now = datetime.now(timezone.utc)
        orch = self.orch

        for task in list(self.state.active_tasks()):
            if task.status != TaskStatus.RUNNING:
                continue
            pi = self.state.find_process(task.task_id)
            if pi is None:
                continue
            try:
                started = datetime.fromisoformat(pi.started_at)
                elapsed = (now - started).total_seconds()
            except (ValueError, TypeError):
                continue

            if elapsed >= timeout_seconds:
                logger.warning(
                    "Task %s timed out after %.0fs (limit=%ds)",
                    task.task_id, elapsed, timeout_seconds,
                )
                self.worker_service.terminate_task(task.task_id)
                self.state.remove_process(task.task_id)
                task.mark_timed_out()
                self.state.total_errors += 1
                self.memory_service.record_event(
                    self.state,
                    f"Task {task.task_id} timed out after {elapsed:.0f}s",
                )
                orch._log_event(
                    OrchestratorEvent.WORKER_FAILED,
                    f"task={task.task_id} timed_out after {elapsed:.0f}s",
                )

                # Also mark PlanTask
                if task.metadata.get("plan_mode") and self._current_plan:
                    stage_num = task.metadata.get("stage_number", -1)
                    pt = self._current_plan.get_task_by_stage(stage_num)
                    if pt:
                        pt.status = TaskStatus.TIMED_OUT
                        pt.completed_at = datetime.now(timezone.utc).isoformat()
                        self._persist_current_plan()

    # ---------------------------------------------------------------
    # Silent worker detection
    # ---------------------------------------------------------------

    _SILENT_WARN_SECONDS = 300  # 5 min of zero output before warning

    def _check_silent_workers(self) -> None:
        """Diagnose silent or stalled workers without killing them."""
        now = datetime.now(timezone.utc)

        for task in list(self.state.active_tasks()):
            if task.status != TaskStatus.RUNNING:
                continue
            pi = self.state.find_process(task.task_id)
            if pi is None:
                continue
            try:
                started = datetime.fromisoformat(pi.started_at)
                elapsed_seconds = (now - started).total_seconds()
            except (ValueError, TypeError):
                continue

            handle = self.worker_service._active_handles.get(task.task_id)
            if not (handle and handle.process and handle.process.poll() is None):
                continue

            stdout_len = len((pi.partial_output or "").strip())
            stderr_len = len((pi.partial_error_output or "").strip())
            last_output_age = None
            if pi.last_output_at:
                try:
                    last_output_age = (now - datetime.fromisoformat(pi.last_output_at)).total_seconds()
                except (TypeError, ValueError):
                    last_output_age = None

            state = ""
            if pi.first_output_at is None and elapsed_seconds >= self._SILENT_WARN_SECONDS:
                state = "stderr_only" if stderr_len > 0 else "no_output"
            elif last_output_age is not None and last_output_age >= self._SILENT_WARN_SECONDS:
                state = "stalled"
            elif pi.first_output_at is not None and elapsed_seconds >= self._SILENT_WARN_SECONDS:
                state = "slow_active"

            if not state:
                pi.monitor_state = ""
                pi.monitor_warning_sent = False
                continue
            if pi.monitor_state == state and pi.monitor_warning_sent:
                continue

            pi.monitor_state = state
            pi.monitor_warning_sent = True
            logger.warning(
                "worker_%s: task=%s pid=%s elapsed=%.0fs stdout=%d stderr=%d last_output_age=%s",
                state,
                task.task_id,
                pi.pid,
                elapsed_seconds,
                pi.stdout_bytes,
                pi.stderr_bytes,
                f"{last_output_age:.0f}s" if last_output_age is not None else "n/a",
            )

    # ---------------------------------------------------------------
    # MCP health check
    # ---------------------------------------------------------------

    def _periodic_mcp_health_check(self) -> None:
        """Check MCP health every 5 cycles.

        NEVER spawns a CLI subprocess probe while a worker is active,
        because a second CLI process would compete for the same MCP
        stdio connection and kill the running worker's session.
        """
        if self.state.current_cycle - self._mcp_check_cycle < 5:
            return
        self._mcp_check_cycle = self.state.current_cycle

        # If workers are running, do NOT spawn a competing CLI probe.
        active = self.state.active_tasks()
        if active:
            logger.debug(
                "MCP health check: skipped (subprocess probe) because "
                "%d worker(s) active — inferring healthy",
                len(active),
            )
            return

        # No workers running — safe to probe.
        self._mcp_healthy = self.worker_service.check_mcp_health(
            self.config.worker_adapter.cli_path,
            timeout=60,
        )
        if self._mcp_healthy:
            logger.debug("MCP health check: OK")
        else:
            logger.warning("MCP health check: FAILED — MCP tasks will be skipped")
