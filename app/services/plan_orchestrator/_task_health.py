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


# Default configurable via config.silent_worker_warn_seconds
_SILENT_WARN_SECONDS = 300


class TaskHealthMixin:
    """Monitors task health: timeouts, silent workers, and periodic MCP checks."""

    # ---------------------------------------------------------------
    # Timeout enforcement
    # ---------------------------------------------------------------

    def _check_timeouts(self) -> None:
        """Kill tasks that exceeded the timeout threshold."""
        base_timeout = getattr(
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

            timeout_seconds = self._estimate_stage_timeout(task, base_timeout)

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
                        max_attempts = getattr(task, "max_attempts", 3)
                        if task.attempts < max_attempts:
                            # Retry: reset both Task and PlanTask to PENDING
                            task.attempts += 1
                            task.status = TaskStatus.PENDING
                            task.assigned_worker_id = None
                            pt.status = TaskStatus.PENDING
                            pt.assigned_worker_id = None
                            pt.completed_at = None
                            self._stage_retry_counts[stage_num] = (
                                self._stage_retry_counts.get(stage_num, 0) + 1
                            )
                            logger.info(
                                "Plan task stage %d timed out, retrying "
                                "(attempt %d/%d)",
                                stage_num, task.attempts, max_attempts,
                            )
                        else:
                            pt.status = TaskStatus.TIMED_OUT
                            pt.completed_at = datetime.now(timezone.utc).isoformat()
                            logger.warning(
                                "Plan task stage %d timed out permanently "
                                "(attempt %d/%d)",
                                stage_num, task.attempts, max_attempts,
                            )
                        self._persist_current_plan()

    # ---------------------------------------------------------------
    # Dynamic timeout estimation
    # ---------------------------------------------------------------

    # Tools that involve heavy computation (model training, dataset builds)
    _HEAVY_TOOLS: frozenset[str] = frozenset({
        "models_train",
        "features_dataset",      # build action is expensive
        "features_analytics",    # build_outcomes is expensive
        "backtests_walkforward", # rolling OOS is compute-heavy
        "backtests_studies",     # multi-variant batches
    })

    def _estimate_stage_timeout(self, task, base_timeout: int) -> int:
        """Estimate timeout for a plan task based on step count and tool types."""
        if not task.metadata.get("plan_mode") or not self._current_plan:
            return base_timeout
        stage_num = task.metadata.get("stage_number", -1)
        pt = self._current_plan.get_task_by_stage(stage_num)
        if pt is None:
            return base_timeout

        steps = pt.steps or []
        step_count = len(steps)
        # +60s per step beyond 3
        extra = max(0, step_count - 3) * 60
        # +300s if stage contains heavy tools
        if any(
            getattr(s, "tool_name", "") in self._HEAVY_TOOLS
            for s in steps
        ):
            extra += 300
        return base_timeout + extra

    # ---------------------------------------------------------------
    # Silent worker detection
    # ---------------------------------------------------------------

    _SILENT_WARN_SECONDS = 300  # 5 min of zero output before warning
    _INTERMEDIATE_COLLECT_SECONDS = 450  # snapshot partial output before timeout

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

            # Dynamic warn threshold based on step count
            warn_seconds = self._SILENT_WARN_SECONDS
            if task.metadata.get("plan_mode") and self._current_plan:
                stage_num = task.metadata.get("stage_number", -1)
                pt = self._current_plan.get_task_by_stage(stage_num)
                if pt:
                    step_count = len(pt.steps) if pt.steps else 0
                    warn_seconds = self._SILENT_WARN_SECONDS + max(0, step_count - 3) * 60

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
            if pi.first_output_at is None and elapsed_seconds >= warn_seconds:
                state = "stderr_only" if stderr_len > 0 else "no_output"
            elif last_output_age is not None and last_output_age >= warn_seconds:
                state = "stalled"
            elif pi.first_output_at is not None and elapsed_seconds >= warn_seconds:
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

            # Intermediate collection: snapshot partial output at 450s
            if (
                state in ("stalled", "no_output", "stderr_only")
                and elapsed_seconds >= self._INTERMEDIATE_COLLECT_SECONDS
                and not pi.intermediate_collected
            ):
                pi.intermediate_collected = True
                partial_len = len(pi.partial_output or "")
                logger.info(
                    "worker_intermediate_collect: task=%s pid=%s "
                    "collecting partial output (%d bytes)",
                    task.task_id, pi.pid, partial_len,
                )
                self.memory_service.record_event(
                    self.state,
                    f"Worker task {task.task_id} stalled at {elapsed_seconds:.0f}s, "
                    f"partial output captured ({partial_len} bytes)",
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
            self._cycles_since_last_real_health_check += 1

            # Warn if MCP health hasn't been verified for too long
            if self._cycles_since_last_real_health_check >= 20:
                logger.warning(
                    "MCP health check: %d cycles since last real probe "
                    "(%d worker(s) active) — MCP health is STALE",
                    self._cycles_since_last_real_health_check,
                    len(active),
                )

            # Check for suspiciously silent workers (potential MCP death)
            now = datetime.now(timezone.utc)
            for task in active:
                if task.status != TaskStatus.RUNNING:
                    continue
                pi = self.state.find_process(task.task_id)
                if pi is None or not pi.last_output_at:
                    continue
                try:
                    last_output = datetime.fromisoformat(pi.last_output_at)
                    silence_seconds = (now - last_output).total_seconds()
                    if silence_seconds >= 600:
                        logger.warning(
                            "MCP health suspected dead: worker task=%s "
                            "has been silent for %.0fs while supposedly active",
                            task.task_id, silence_seconds,
                        )
                except (TypeError, ValueError):
                    pass

            logger.debug(
                "MCP health check: skipped (subprocess probe) because "
                "%d worker(s) active — inferring healthy",
                len(active),
            )
            return

        # No workers running — safe to probe.
        self._cycles_since_last_real_health_check = 0
        self._mcp_healthy = self.worker_service.check_mcp_health(
            self.config.worker_adapter.cli_path,
            timeout=60,
        )
        if self._mcp_healthy:
            logger.debug("MCP health check: OK")
            self.state.mcp_consecutive_failures = 0
        else:
            self.state.mcp_consecutive_failures += 1
            logger.warning(
                "MCP health check: FAILED (%d consecutive) — MCP tasks will be skipped",
                self.state.mcp_consecutive_failures,
            )
