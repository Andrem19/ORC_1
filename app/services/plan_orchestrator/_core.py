"""
Plan orchestrator core — main loop, lifecycle, and shared helpers.

Provides PlanOrchestratorCore: __init__, run(), request_stop(),
set_research_context(), and small helpers shared across mixins.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.models import (
    OrchestratorEvent,
    StopReason,
    TaskStatus,
)
from app.plan_models import ResearchPlan
from app.plan_validation import (
    PlanValidationResult,
    validate_plan,
)

if TYPE_CHECKING:
    from app.config import OrchestratorConfig
    from app.mcp_problem_scanner import McpProblemScanner
    from app.models import OrchestratorState, TaskResult
    from app.plan_store import PlanStore
    from app.scheduler import Scheduler
    from app.services.memory_service import MemoryService
    from app.services.notification_service import NotificationService
    from app.services.planner_service import PlannerService
    from app.services.worker_service import WorkerService
    from app.state_store import StateStore

logger = logging.getLogger("orchestrator.plan")


class PlanOrchestratorCore:
    """Base class providing __init__, the main run() loop, and shared helpers."""

    def __init__(self, orch: Any) -> None:
        self.orch = orch
        # Convenience aliases
        self.config: OrchestratorConfig = orch.config
        self.state: OrchestratorState = orch.state
        self.planner_service: PlannerService = orch.planner_service
        self.worker_service: WorkerService = orch.worker_service
        self.scheduler: Scheduler = orch.scheduler
        self.memory_service: MemoryService = orch.memory_service
        self.notification_service: NotificationService = orch.notification_service
        self._plan_store: PlanStore | None = orch._plan_store
        self._worker_ids: list[str] = orch._worker_ids
        self._mcp_scanner: McpProblemScanner | None = orch._mcp_scanner

        self._current_plan: ResearchPlan | None = None
        self._next_worker_idx: int = 0

        # MCP health tracking
        self._mcp_healthy: bool = True
        self._mcp_check_cycle: int = 0
        self._cycles_since_last_real_health_check: int = 0

        # Graceful stop flag (set externally via request_stop)
        self._stop_requested: bool = False

        # Drain mode (set externally via request_drain)
        self._drain_mode: bool = False
        self._drain_started_at: float | None = None

        # Per-stage retry tracking (prevents infinite retry on one stage)
        self._stage_retry_counts: dict[int, int] = {}
        self._mcp_skip_counts: dict[int, int] = {}
        self._max_stage_retries: int = 3
        self._max_plan_attempts: int = 5  # 1 create + 4 repairs
        self._terminal_stop_reason: StopReason | None = None
        self._terminal_stop_summary: str = ""
        self._last_repair_error_count: int = 0

    def set_research_context(self, text: str | None) -> None:
        orch = self.orch
        orch._research_context_text = text

    def request_stop(self) -> None:
        self._stop_requested = True

    # ---------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------

    def run(self) -> StopReason:
        """Plan-driven orchestrator loop."""
        orch = self.orch
        self.state.status = "running"
        orch.save_state()
        orch._log_event(
            OrchestratorEvent.CONFIG_LOADED,
            f"[plan_mode] goal={self.config.goal[:80]}",
        )
        self.notification_service.send_lifecycle(
            "started", f"[plan_mode] Goal: {self.config.goal[:100]}",
        )

        # Restore current plan from disk if available
        if self._plan_store:
            latest = self._plan_store.load_latest_plan()
            if latest:
                self._current_plan = latest
                self.state.current_plan_version = latest.version
                self._clear_invalid_plan_state()
                logger.info("Restored plan v%d from disk", latest.version)
                self._reconcile_current_plan_state()

        # Clean up stale non-plan tasks from pre-plan-mode runs
        self._cancel_non_plan_tasks()

        while True:
            if self._stop_requested:
                orch._finish(StopReason.NO_PROGRESS, "Stopped by signal")
                return StopReason.NO_PROGRESS

            self.state.current_cycle += 1
            logger.info(
                "=== Plan Cycle %d (plan v%d) ===",
                self.state.current_cycle, self.state.current_plan_version,
            )

            # 0. Check timeouts on running tasks
            self._check_timeouts()

            # 0b. Warn about silent-but-alive workers
            self._check_silent_workers()

            # 1. Collect results from running workers (delegates to Orchestrator)
            new_results = orch._collect_results()

            # 1b. Drain mode: wait for running tasks to finish
            if self._drain_mode:
                active = self.state.active_tasks()
                elapsed = time.monotonic() - (self._drain_started_at or time.monotonic())
                timeout = self.config.drain_timeout_seconds
                if not active:
                    logger.info("Drain complete — all running tasks finished")
                    orch._finish(StopReason.GRACEFUL_STOP, "Graceful drain completed")
                    return StopReason.GRACEFUL_STOP
                if elapsed >= timeout:
                    logger.warning(
                        "Drain timeout (%ds) exceeded with %d tasks still running — forcing stop",
                        timeout, len(active),
                    )
                    self._terminate_draining_tasks(active)
                    orch._finish(
                        StopReason.GRACEFUL_STOP,
                        f"Drain timeout ({timeout}s), {len(active)} tasks terminated",
                    )
                    return StopReason.GRACEFUL_STOP
                logger.info(
                    "Drain mode: %d tasks still running (elapsed %.0fs / %ds timeout)",
                    len(active), elapsed, timeout,
                )
                orch.save_state()
                self._plan_sleep()
                continue

            # 2. If planner is running, check on it
            if self.planner_service.is_running:
                plan_data, is_finished = self.planner_service.check_plan_output()
                self._sync_planner_runtime_state()
                if not is_finished:
                    self._check_planner_watchdog()
                    if self._should_check_stop():
                        return self._get_stop_reason()
                    orch.save_state()
                    self._plan_sleep()
                    continue

                if is_finished and plan_data is not None:
                    # Stop planner progress spinner
                    from app.rich_handler import ProgressManager
                    pm = ProgressManager._instance
                    if pm and pm.is_active():
                        pm.stop_planner_wait()
                    self._process_plan_data(plan_data)
                    self._sync_planner_runtime_state(clear=True)

                if self._should_check_stop():
                    return self._get_stop_reason()

                orch.save_state()
                self._plan_sleep()
                continue

            # 3. Convert new TaskResults to TaskReports + retry logic
            self._process_plan_results(new_results)

            # 4. No current plan → create one
            if self._current_plan is None:
                if self._should_attempt_plan_repair():
                    self._repair_plan()
                else:
                    self._create_plan()
                orch.save_state()
                self._plan_sleep()
                continue

            # 5. MCP health check (periodic)
            self._periodic_mcp_health_check()

            # 6. Dispatch pending plan tasks (respects explicit dependencies)
            dispatchable = self._current_plan.dispatchable_tasks()
            can_dispatch = self.scheduler.plan_tasks_to_dispatch(
                self.state, self.config.max_concurrent_plan_tasks,
            )
            if dispatchable and can_dispatch > 0:
                to_launch = dispatchable[:can_dispatch]
                self._dispatch_plan_tasks(to_launch)
                orch.save_state()
                self._plan_sleep()
                continue

            # 7. All dispatched tasks resolved → revise plan
            if self._current_plan.all_dispatched_resolved():
                self._revise_plan()
                orch.save_state()
                self._plan_sleep()
                continue

            # 8. Check stop conditions
            if self._should_check_stop():
                return self._get_stop_reason()

            # 9. Nothing to do
            if self.state.active_tasks():
                orch.save_state()
                self._plan_sleep()
                continue

            self.state.empty_cycles += 1
            orch.save_state()
            self._plan_sleep()

    # ---------------------------------------------------------------
    # Drain mode helpers
    # ---------------------------------------------------------------

    def _terminate_draining_tasks(self, active_tasks: list) -> None:
        """Terminate tasks still running after drain timeout."""
        orch = self.orch
        for task in active_tasks:
            if task.status == TaskStatus.RUNNING:
                self.worker_service.terminate_task(task.task_id)
                orch.state.remove_process(task.task_id)
                task.mark_cancelled()
                if task.metadata.get("plan_mode") and self._current_plan:
                    stage_num = task.metadata.get("stage_number", -1)
                    pt = self._current_plan.get_task_by_stage(stage_num)
                    if pt:
                        pt.status = TaskStatus.CANCELLED
                        pt.completed_at = datetime.now(timezone.utc).isoformat()
        self._persist_current_plan()

    # ---------------------------------------------------------------
    # Stale task cleanup
    # ---------------------------------------------------------------

    def _cancel_non_plan_tasks(self) -> None:
        """Cancel all non-plan tasks that are still active or pending."""
        cancelled_count = 0
        for task in self.state.tasks:
            if task.metadata.get("plan_mode"):
                continue
            if task.status in (
                TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_RESULT,
            ):
                if task.status == TaskStatus.RUNNING:
                    self.worker_service.terminate_task(task.task_id)
                    self.state.remove_process(task.task_id)
                task.mark_cancelled()
                cancelled_count += 1
        if cancelled_count > 0:
            logger.info("Cancelled %d non-plan tasks on plan mode entry", cancelled_count)
            self.memory_service.record_event(
                self.state,
                f"Cancelled {cancelled_count} pre-plan tasks",
            )

    # ---------------------------------------------------------------
    # Plan state reconciliation
    # ---------------------------------------------------------------

    def _reconcile_current_plan_state(self) -> None:
        """Reconcile plan progress from saved reports and orchestrator state."""
        if not self._current_plan:
            return

        reports_by_task: dict[str, Any] = {}
        if self._plan_store:
            for report in self._plan_store.load_reports_for_plan(self._current_plan.version):
                reports_by_task[report.task_id] = report

        state_tasks = {
            task.task_id: task
            for task in self.state.tasks
            if task.metadata.get("plan_mode")
            and task.metadata.get("plan_version") == self._current_plan.version
        }

        for pt in self._current_plan.tasks:
            task = state_tasks.get(pt.task_id)
            report = reports_by_task.get(pt.task_id)

            if report is not None:
                pt.results_table_rows = report.results_table
                pt.verdict = report.verdict
                if report.status == "success":
                    pt.status = TaskStatus.COMPLETED
                    pt.completed_at = report.timestamp
                    self._maybe_update_plan_baseline(pt, report)
                elif report.status == "error" and pt.status == TaskStatus.PENDING:
                    pt.status = TaskStatus.FAILED
                    pt.completed_at = report.timestamp
                elif report.status == "partial" and pt.status == TaskStatus.PENDING:
                    pt.status = TaskStatus.COMPLETED
                    pt.completed_at = report.timestamp
                    self._maybe_update_plan_baseline(pt, report)

            if task is not None:
                pt.assigned_worker_id = task.assigned_worker_id
                # Only sync Task status → PlanTask when the PlanTask is not
                # already in a terminal state.  This prevents reconciliation
                # from overwriting a partial-result COMPLETED back to FAILED.
                if not pt.is_resolved:
                    pt.status = task.status
                if task.status in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                    TaskStatus.TIMED_OUT,
                    TaskStatus.STALLED,
                } and pt.completed_at is None:
                    pt.completed_at = task.updated_at

        self._persist_current_plan()

    # ---------------------------------------------------------------
    # Shared helpers (used by multiple mixins)
    # ---------------------------------------------------------------

    def _should_check_stop(self) -> bool:
        if self._terminal_stop_reason is not None:
            self.orch._finish(self._terminal_stop_reason, self._terminal_stop_summary)
            return True
        stop = self.scheduler.should_stop_orchestrator(self.state)
        if stop:
            self.orch._finish(StopReason(stop))
            return True
        return False

    def _get_stop_reason(self) -> StopReason:
        if self._terminal_stop_reason is not None:
            return self._terminal_stop_reason
        return StopReason(self.state.stop_reason or "no_progress")

    def _plan_sleep(self) -> None:
        sleep_seconds = self.scheduler.sleep_interval(self.state)
        if self.planner_service.is_running:
            sleep_seconds = min(sleep_seconds, 30)
        self.orch._log_event(OrchestratorEvent.SLEEPING, f"{sleep_seconds}s")
        self.scheduler.sleep(seconds=sleep_seconds)

    def _persist_current_plan(self) -> None:
        """Persist the current plan snapshot whenever runtime progress changes."""
        if self._plan_store and self._current_plan:
            self._plan_store.save_plan(self._current_plan)

    def _clear_invalid_plan_state(self) -> None:
        self.state.current_plan_attempt = 0
        self.state.current_plan_attempt_type = None
        self.state.current_plan_validation_errors = []
        self.state.last_rejected_plan_version = None
        self.state.last_rejected_plan_attempt_at = None
        self.state.last_rejected_plan_artifact = None
        self._last_repair_error_count = 0

    def _validate_plan(self, plan: ResearchPlan) -> PlanValidationResult:
        """Validate plan structure and executable instructions."""
        return validate_plan(plan)

    def _build_planner_context(self) -> str:
        baseline_bootstrap = getattr(self.config, "research_config", None)
        parts: list[str] = []
        if isinstance(baseline_bootstrap, dict) and baseline_bootstrap:
            parts.append("## Baseline Bootstrap")
            for key in (
                "baseline_snapshot_id",
                "baseline_version",
                "symbol",
                "anchor_timeframe",
                "execution_timeframe",
            ):
                value = baseline_bootstrap.get(key)
                if value is not None:
                    parts.append(f"- {key}: {value}")
        if self.orch._research_context_text:
            if parts:
                parts.append("")
            parts.append("## Live Research Context")
            parts.append(self.orch._research_context_text)
        return "\n".join(parts)

    def _get_mcp_summary(self) -> str | None:
        if self._mcp_scanner:
            return self._mcp_scanner.get_context_for_planner(
                max_items=self.config.mcp_review.max_problems_in_context,
            )
        return None
