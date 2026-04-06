"""
Plan-mode orchestrator service.

Extracted from Orchestrator to keep file sizes under control.
Handles the plan-driven loop: create plan → dispatch tasks → collect reports → revise.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.models import (
    OrchestratorEvent,
    StopReason,
    TaskStatus,
)
from app.plan_models import (
    AntiPattern,
    DecisionGate,
    PlanTask,
    ResearchPlan,
    plan_task_to_task,
)
from app.plan_symbolic_refs import resolve_symbolic_references
from app.plan_validation import (
    PlanRepairRequest,
    PlanValidationError,
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


class PlanOrchestratorService:
    """Runs the plan-driven orchestrator loop.

    Receives an ``orch`` reference to delegate shared operations
    (_collect_results, save_state, _finish, _log_event, etc.).
    """

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

        # Graceful stop flag (set externally via request_stop)
        self._stop_requested: bool = False

        # Per-stage retry tracking (prevents infinite retry on one stage)
        self._stage_retry_counts: dict[int, int] = {}
        self._max_stage_retries: int = 3
        self._max_plan_attempts: int = 3  # 1 create + 2 repairs
        self._terminal_stop_reason: StopReason | None = None
        self._terminal_stop_summary: str = ""

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
    # Result processing + retry
    # ---------------------------------------------------------------

    def _process_plan_results(self, new_results: list[TaskResult]) -> None:
        """Convert TaskResults to TaskReports, apply retry logic for failures."""
        orch = self.orch

        for result in new_results:
            task = self.state.find_task(result.task_id)
            if not task:
                continue

            if not task.metadata.get("plan_mode"):
                # Non-plan result — already handled by _collect_results → _handle_task_result
                continue

            report = result.plan_report
            if report is None:
                logger.warning(
                    "Plan task %s missing structured plan_report; using degraded fallback from raw output",
                    result.task_id,
                )
                from app.result_parser import parse_task_report

                report = parse_task_report(
                    result.raw_output,
                    task_id=result.task_id,
                    worker_id=result.worker_id,
                    plan_version=task.metadata.get("plan_version", 0),
                )
            if self._plan_store:
                self._plan_store.save_report(report)

            # Update PlanTask in current plan
            if not self._current_plan:
                continue

            stage_num = task.metadata.get("stage_number", -1)
            pt = self._current_plan.get_task_by_stage(stage_num)
            if not pt:
                continue

            pt.results_table_rows = report.results_table
            pt.verdict = report.verdict
            pt.assigned_worker_id = task.assigned_worker_id
            pt.completed_at = datetime.now(timezone.utc).isoformat()

            if result.status == "success":
                pt.status = TaskStatus.COMPLETED
                self._maybe_update_plan_baseline(pt, report)
                orch._log_event(
                    OrchestratorEvent.WORKER_COMPLETED,
                    f"task={task.task_id} stage={stage_num}",
                )
            else:
                # Error — retry logic
                max_attempts = task.max_attempts
                if task.attempts < max_attempts:
                    task.attempts += 1
                    # Track per-stage retry count
                    self._stage_retry_counts[stage_num] = (
                        self._stage_retry_counts.get(stage_num, 0) + 1
                    )
                    # Reset for re-dispatch
                    task.status = TaskStatus.PENDING
                    task.assigned_worker_id = None
                    pt.status = TaskStatus.PENDING
                    pt.assigned_worker_id = None
                    pt.completed_at = None

                    mcp_fail = self._is_mcp_failure(report)
                    if mcp_fail:
                        # Don't mark MCP unhealthy — the retry gets a fresh
                        # subprocess with a fresh MCP connection.
                        logger.info(
                            "MCP failure on stage %d, retrying with fresh "
                            "subprocess (attempt %d/%d)",
                            pt.stage_number, task.attempts, max_attempts,
                        )
                    else:
                        logger.info(
                            "Stage %d failed, retrying (attempt %d/%d)",
                            pt.stage_number, task.attempts, max_attempts,
                        )
                    self.memory_service.record_event(
                        self.state,
                        f"Retrying plan stage {pt.stage_number} "
                        f"(attempt {task.attempts}/{max_attempts})",
                    )
                else:
                    # Max attempts exhausted — permanent failure
                    pt.status = TaskStatus.FAILED
                    if self._is_mcp_failure(report):
                        self._mcp_healthy = False
                        logger.warning(
                            "MCP failure on stage %d after %d retries — "
                            "marking MCP unhealthy",
                            pt.stage_number, task.attempts,
                        )
                    orch._log_event(
                        OrchestratorEvent.WORKER_FAILED,
                        f"task={task.task_id} stage={stage_num} "
                        f"permanently failed after {task.attempts} attempts",
                    )

            self._persist_current_plan()

    @staticmethod
    def _is_mcp_failure(report: Any) -> bool:
        """Detect MCP-specific failures from a TaskReport."""
        mcp_indicators = [
            "tool not found in registry",
            "mcp tools are not available",
            "mcp dev_space1",
            "mcp server not connected",
            "server may not be running",
        ]
        text = f"{getattr(report, 'error', '')} {getattr(report, 'raw_output', '')}".lower()
        return any(indicator in text for indicator in mcp_indicators)

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
        """Warn about workers that have produced no output for extended periods.

        Unlike timeouts, this does NOT kill the worker — qwen CLI buffers
        output internally, so long silent periods are normal when waiting
        for MCP tool responses.
        """
        now = datetime.now(timezone.utc)

        for task in list(self.state.active_tasks()):
            if task.status != TaskStatus.RUNNING:
                continue
            pi = self.state.find_process(task.task_id)
            if pi is None:
                continue
            try:
                started = datetime.fromisoformat(pi.started_at)
                silent_seconds = (now - started).total_seconds()
            except (ValueError, TypeError):
                continue

            if silent_seconds >= self._SILENT_WARN_SECONDS:
                # Check if process is actually alive
                handle = self.worker_service._active_handles.get(task.task_id)
                if handle and handle.process and handle.process.poll() is None:
                    stdout_len = len((pi.partial_output or "").strip())
                    stderr_len = len((pi.partial_error_output or "").strip())
                    if stdout_len == 0 and stderr_len == 0:
                        state = "no stdout/stderr"
                    elif stdout_len == 0 and stderr_len > 0:
                        state = "stderr only"
                    else:
                        continue
                    logger.warning(
                        "Task %s (pid=%s) quiet for %.0fs with %s — likely CLI buffering or backend stall",
                        task.task_id, pi.pid, silent_seconds, state,
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
    # Plan creation and revision
    # ---------------------------------------------------------------

    def _create_plan(self) -> None:
        """Call planner to create a new research plan."""
        if not self._plan_store:
            return

        anti_patterns = self._plan_store.load_all_anti_patterns()
        cumulative = self._plan_store.load_cumulative_summary()

        prev_md = None
        if self.state.current_plan_version > 0:
            prev_md = self._plan_store.load_plan_markdown(self.state.current_plan_version)

        plan_version = self.state.current_plan_version + 1
        self.state.current_plan_attempt = 1
        self.state.current_plan_attempt_type = "create"
        self.state.current_plan_validation_errors = []
        logger.info("Creating plan (v%d) via planner", plan_version)
        # Progress tracking — planner spinner (yellow)
        from app.rich_handler import ProgressManager
        pm = ProgressManager._instance
        if pm and pm.is_active():
            pm.start_planner_wait(
                model=self.config.planner_adapter.model,
                action=f"Creating research plan v{plan_version} (attempt 1)",
            )
        self.planner_service.start_plan_creation(
            goal=self.config.goal,
            research_context=self.orch._research_context_text,
            anti_patterns=anti_patterns if anti_patterns else None,
            cumulative_summary=cumulative,
            worker_ids=self._worker_ids,
            mcp_problem_summary=self._get_mcp_summary(),
            previous_plan_markdown=prev_md,
            plan_version=plan_version,
            attempt_number=1,
        )

    def _repair_plan(self) -> None:
        """Call planner to repair the latest rejected create-plan attempt."""
        if not self._plan_store:
            return

        repair_request = self._build_repair_request()
        if repair_request is None:
            logger.warning("Repair requested but no rejected plan payload is available; falling back to create")
            self._clear_invalid_plan_state()
            self._create_plan()
            return

        self.state.current_plan_attempt = repair_request.attempt_number
        self.state.current_plan_attempt_type = "repair"
        logger.info(
            "Repairing rejected plan v%d via planner (attempt %d/%d)",
            repair_request.plan_version,
            repair_request.attempt_number,
            self._max_plan_attempts,
        )
        from app.rich_handler import ProgressManager
        pm = ProgressManager._instance
        if pm and pm.is_active():
            pm.start_planner_wait(
                model=self.config.planner_adapter.model,
                action=(
                    f"Repairing plan v{repair_request.plan_version} "
                    f"(attempt {repair_request.attempt_number}/{self._max_plan_attempts})"
                ),
            )
        self.planner_service.start_plan_repair(
            repair_request=repair_request,
            research_context=self.orch._research_context_text,
            worker_ids=self._worker_ids,
            mcp_problem_summary=self._get_mcp_summary(),
        )

    def _revise_plan(self) -> None:
        """Call planner to revise the current plan based on collected reports."""
        if not self._plan_store or not self._current_plan:
            return

        if len(self._current_plan.tasks) == 0:
            logger.warning(
                "Cannot revise empty plan v%d — clearing for re-creation",
                self._current_plan.version,
            )
            self._current_plan = None
            return

        reports = self._plan_store.load_reports_for_plan(self._current_plan.version)
        anti_patterns = self._plan_store.load_all_anti_patterns()

        logger.info(
            "Revising plan v%d → v%d (%d reports)",
            self._current_plan.version,
            self._current_plan.version + 1,
            len(reports),
        )
        # Progress tracking — planner spinner (yellow)
        from app.rich_handler import ProgressManager
        pm = ProgressManager._instance
        if pm and pm.is_active():
            pm.start_planner_wait(
                model=self.config.planner_adapter.model,
                action=f"Revising plan v{self._current_plan.version} → v{self._current_plan.version + 1}",
            )
        self.planner_service.start_plan_revision(
            goal=self.config.goal,
            current_plan=self._current_plan,
            reports=reports,
            research_context=self.orch._research_context_text,
            anti_patterns=anti_patterns if anti_patterns else None,
            worker_ids=self._worker_ids,
            mcp_problem_summary=self._get_mcp_summary(),
        )

    def _process_plan_data(self, data: dict) -> None:
        """Process parsed plan data from the planner."""
        orch = self.orch
        action = data.get("plan_action", "create")
        version = data.get("plan_version", self.state.current_plan_version + 1)

        plan = ResearchPlan(
            schema_version=int(data.get("schema_version", 1) or 1),
            version=version,
            frozen_base=data.get("frozen_base", ""),
            baseline_run_id=data.get("baseline_run_id"),
            baseline_snapshot_ref=data.get("baseline_snapshot_ref"),
            baseline_metrics=data.get("baseline_metrics", {}) if isinstance(data.get("baseline_metrics"), dict) else {},
            goal=self.config.goal,
            principles=data.get("principles", []),
            cumulative_summary=data.get("cumulative_summary", ""),
            plan_markdown=data.get("plan_markdown", ""),
            status="active",
        )

        for t_data in data.get("tasks", []):
            gates = []
            for g in t_data.get("decision_gates", []):
                if isinstance(g, dict):
                    gates.append(DecisionGate(**g))
                elif isinstance(g, DecisionGate):
                    gates.append(g)

            pt = PlanTask(
                plan_version=version,
                stage_number=t_data.get("stage_number", 0),
                stage_name=t_data.get("stage_name", ""),
                theory=t_data.get("theory", ""),
                depends_on=t_data.get("depends_on", []),
                agent_instructions=t_data.get("agent_instructions", []),
                results_table_columns=t_data.get("results_table_columns", []),
                decision_gates=gates,
                verdict="PENDING",
            )
            plan.tasks.append(pt)

        for ap_data in data.get("anti_patterns_new", []):
            plan.anti_patterns.append(AntiPattern(
                category=ap_data.get("category", ""),
                description=ap_data.get("description", ""),
                evidence_count=ap_data.get("evidence_count", 0),
                evidence_summary=ap_data.get("evidence_summary", ""),
                verdict="REJECTED",
                source_plan_version=version,
            ))

        # Carry forward anti-patterns from previous plan
        if self._current_plan:
            existing_ids = {ap.pattern_id for ap in plan.anti_patterns}
            for ap in self._current_plan.anti_patterns:
                if ap.pattern_id not in existing_ids:
                    plan.anti_patterns.append(ap)
                    existing_ids.add(ap.pattern_id)
            if not plan.baseline_run_id:
                plan.baseline_run_id = self._current_plan.baseline_run_id
            if not plan.baseline_snapshot_ref:
                plan.baseline_snapshot_ref = self._current_plan.baseline_snapshot_ref
            if not plan.baseline_metrics:
                plan.baseline_metrics = dict(self._current_plan.baseline_metrics)

        plan.execution_order = data.get("tasks_to_dispatch", [])

        # --- Reject empty plans ---
        if len(plan.tasks) == 0:
            validation = PlanValidationResult(errors=[
                PlanValidationError(
                    stage_number=-1,
                    code="empty_plan",
                    message=(
                        "Planner returned 0 tasks"
                        + (
                            f" (parse_failed={data.get('_parse_failed', False)}; "
                            f"reason={str(data.get('reason', ''))[:200]})"
                        )
                    ),
                )
            ])
            self._handle_invalid_plan(
                plan_version=version,
                parsed_data=data,
                validation=validation,
                raw_output=self.planner_service.last_plan_raw_output,
            )
            return

        validation = self._validate_plan(plan)
        if not validation.is_valid:
            self._handle_invalid_plan(
                plan_version=version,
                parsed_data=data,
                validation=validation,
                raw_output=self.planner_service.last_plan_raw_output,
            )
            return

        # Legacy fallback for older planner outputs
        if plan.schema_version < 2 and not plan.execution_order:
            plan.execution_order = [
                t.stage_number for t in sorted(plan.tasks, key=lambda t: t.stage_number)
            ]

        self._current_plan = plan
        self.state.current_plan_version = version
        self._stage_retry_counts.clear()
        self._persist_current_plan()
        self._clear_invalid_plan_state()

        self.memory_service.record_event(
            self.state,
            f"Plan v{version} {action}d: {len(plan.tasks)} tasks, "
            f"{len(plan.tasks)} stages with explicit dependencies",
        )
        self.notification_service.send_lifecycle(
            "plan_created" if action == "create" else "plan_revised",
            f"Plan v{version}: {len(plan.tasks)} stages",
        )
        self.state.last_change_at = datetime.now(timezone.utc).isoformat()
        self.state.empty_cycles = 0

    # ---------------------------------------------------------------
    # Task dispatch
    # ---------------------------------------------------------------

    def _dispatch_plan_tasks(self, plan_tasks: list[PlanTask]) -> None:
        """Dispatch PlanTasks to workers (respects MCP health)."""
        orch = self.orch
        reports_by_stage = self._reports_by_stage()

        for pt in plan_tasks:
            resolved_instructions = self._resolve_plan_task_instructions(pt, reports_by_stage)
            if resolved_instructions is None:
                continue

            # MCP health gate
            instructions_text = " ".join(str(i) for i in resolved_instructions)
            if self._is_mcp_instructions(instructions_text) and not self._mcp_healthy:
                logger.warning(
                    "Skipping MCP-dependent stage %d — MCP unhealthy",
                    pt.stage_number,
                )
                continue

            # Per-stage retry limit — skip stages that exceeded max retries
            stage_retries = self._stage_retry_counts.get(pt.stage_number, 0)
            if stage_retries >= self._max_stage_retries:
                logger.warning(
                    "Skipping stage %d — exceeded max per-stage retries (%d)",
                    pt.stage_number, self._max_stage_retries,
                )
                pt.status = TaskStatus.FAILED
                continue

            worker_id = self._pick_worker()
            if not worker_id:
                logger.warning("No workers available for plan task stage %d", pt.stage_number)
                continue

            pt.assigned_worker_id = worker_id
            pt.status = TaskStatus.RUNNING
            pt.completed_at = None

            task = plan_task_to_task(pt)
            task.mark_running()
            self.state.add_task(task)

            self.state.plan_task_dispatch_map[str(pt.stage_number)] = task.task_id

            process_info = self.worker_service.start_plan_task(
                task=task,
                plan_version=pt.plan_version,
                stage_number=pt.stage_number,
                stage_name=pt.stage_name,
                theory=pt.theory,
                agent_instructions=resolved_instructions,
                results_table_columns=pt.results_table_columns,
            )
            self.state.processes.append(process_info)

            # Progress tracking — worker spinner (green)
            from app.rich_handler import ProgressManager
            pm = ProgressManager._instance
            if pm and pm.is_active():
                pm.add_worker(
                    task.task_id, worker_id,
                    pid=process_info.pid,
                    description=f"Stage {pt.stage_number}: {pt.stage_name}",
                )

            orch._log_event(
                OrchestratorEvent.WORKER_LAUNCHED,
                f"[plan] stage={pt.stage_number} task={task.task_id} worker={worker_id}",
            )

        self.state.last_change_at = datetime.now(timezone.utc).isoformat()
        self.state.empty_cycles = 0
        self._persist_current_plan()

    @staticmethod
    def _is_mcp_instructions(text: str) -> bool:
        """Check if task instructions reference MCP tools."""
        keywords = [
            "backtest", "snapshot", "feature", "model", "strategy",
            "dataset", "mcp", "signal", "heatmap", "walk-forward",
            "diagnostics", "research_project", "research_record",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _should_attempt_plan_repair(self) -> bool:
        return (
            self.state.current_plan_version == 0
            and self.state.last_rejected_plan_version is not None
            and self.state.current_plan_attempt_type in {"create", "repair"}
            and 1 < self.state.current_plan_attempt <= self._max_plan_attempts
            and bool(self.state.last_rejected_plan_artifact)
        )

    def _build_repair_request(self) -> PlanRepairRequest | None:
        artifact_path = self.state.last_rejected_plan_artifact
        if not artifact_path:
            return None
        try:
            import json
            from pathlib import Path

            payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load rejected plan artifact %s: %s", artifact_path, exc)
            return None

        errors = [
            PlanValidationError(
                stage_number=int(err.get("stage_number", -1)),
                instruction_index=err.get("instruction_index"),
                code=str(err.get("code", "")),
                message=str(err.get("message", "")),
                offending_text=str(err.get("offending_text", "")),
            )
            for err in payload.get("validation_errors", [])
            if isinstance(err, dict)
        ]

        return PlanRepairRequest(
            goal=self.config.goal,
            plan_version=int(payload.get("plan_version", self.state.last_rejected_plan_version or 1)),
            attempt_number=int(payload.get("attempt_number", 1)) + 1,
            invalid_plan_data=payload.get("parsed_data", {}),
            validation_errors=errors,
        )

    def _clear_invalid_plan_state(self) -> None:
        self.state.current_plan_attempt = 0
        self.state.current_plan_attempt_type = None
        self.state.current_plan_validation_errors = []
        self.state.last_rejected_plan_version = None
        self.state.last_rejected_plan_attempt_at = None
        self.state.last_rejected_plan_artifact = None

    def _reports_by_stage(self) -> dict[int, Any]:
        reports_by_stage: dict[int, Any] = {}
        if not self._current_plan or not self._plan_store:
            return reports_by_stage

        task_to_stage = {
            pt.task_id: pt.stage_number
            for pt in self._current_plan.tasks
        }
        for report in self._plan_store.load_reports_for_plan(self._current_plan.version):
            stage_number = task_to_stage.get(report.task_id)
            if stage_number is not None:
                reports_by_stage[stage_number] = report
        return reports_by_stage

    def _resolve_plan_task_instructions(
        self,
        plan_task: PlanTask,
        reports_by_stage: dict[int, Any],
    ) -> list[str] | None:
        resolved: list[str] = []
        for instruction in plan_task.agent_instructions:
            resolution = resolve_symbolic_references(instruction, reports_by_stage)
            if not resolution.is_resolved:
                blocking_errors = []
                runtime_errors = []
                for err in resolution.unresolved:
                    if err.stage_number not in reports_by_stage:
                        blocking_errors.append(err)
                    else:
                        runtime_errors.append(err)

                if runtime_errors:
                    message = "; ".join(err.message for err in runtime_errors)
                    logger.error(
                        "Stage %d symbolic refs failed after dependencies resolved: %s",
                        plan_task.stage_number,
                        message,
                    )
                    plan_task.status = TaskStatus.FAILED
                    plan_task.completed_at = datetime.now(timezone.utc).isoformat()
                    self.memory_service.record_error(
                        self.state,
                        f"Plan stage {plan_task.stage_number} symbolic ref error: {message}",
                    )
                    self.state.total_errors += 1
                    self._persist_current_plan()
                    return None

                logger.info(
                    "Stage %d waiting on symbolic refs before dispatch: %s",
                    plan_task.stage_number,
                    ", ".join(err.raw for err in blocking_errors),
                )
                return None

            resolved.append(resolution.resolved_text)

        return resolved

    def _pick_worker(self) -> str | None:
        active_workers = {t.assigned_worker_id for t in self.state.active_tasks()}
        idle = [w for w in self._worker_ids if w not in active_workers]
        if idle:
            worker_id = idle[self._next_worker_idx % len(idle)]
            self._next_worker_idx += 1
            return worker_id
        if self._worker_ids:
            worker_id = self._worker_ids[self._next_worker_idx % len(self._worker_ids)]
            self._next_worker_idx += 1
            return worker_id
        return None

    def _get_mcp_summary(self) -> str | None:
        if self._mcp_scanner:
            return self._mcp_scanner.get_context_for_planner(
                max_items=self.config.mcp_review.max_problems_in_context,
            )
        return None

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
                elif report.status in {"error", "partial"} and pt.status == TaskStatus.PENDING:
                    pt.status = TaskStatus.FAILED
                    pt.completed_at = report.timestamp

            if task is not None:
                pt.assigned_worker_id = task.assigned_worker_id
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

    def _validate_plan(self, plan: ResearchPlan) -> PlanValidationResult:
        """Validate plan structure and executable instructions."""
        return validate_plan(plan)

    def _sync_planner_runtime_state(self, clear: bool = False) -> None:
        snapshot = None if clear else self.planner_service.plan_runtime_snapshot()
        if snapshot is None:
            self.state.planner_started_at = None
            self.state.planner_first_output_at = None
            self.state.planner_last_output_at = None
            self.state.planner_output_bytes = 0
            self.state.planner_stderr_bytes = 0
            return

        self.state.planner_started_at = snapshot.started_at_iso
        self.state.planner_first_output_at = snapshot.first_output_at_iso
        self.state.planner_last_output_at = snapshot.last_output_at_iso
        self.state.planner_output_bytes = snapshot.output_bytes
        self.state.planner_stderr_bytes = snapshot.stderr_bytes

    def _check_planner_watchdog(self) -> None:
        snapshot = self.planner_service.plan_runtime_snapshot()
        if snapshot is None:
            return

        adapter_cfg = getattr(self.config, "planner_adapter", None)
        no_first_byte_seconds = max(30, int(getattr(adapter_cfg, "no_first_byte_seconds", 180) or 180))
        soft_stall_seconds = max(no_first_byte_seconds, int(getattr(adapter_cfg, "soft_timeout_seconds", 300) or 300))
        hard_stall_seconds = max(soft_stall_seconds, int(getattr(adapter_cfg, "hard_timeout_seconds", 900) or 900))

        elapsed = snapshot.elapsed_seconds
        if not snapshot.has_first_byte and elapsed >= no_first_byte_seconds and not snapshot.no_first_byte_warning_sent:
            snapshot.no_first_byte_warning_sent = True
            logger.warning(
                "planner_silent: type=%s version=%d attempt=%d elapsed=%.0fs prompt=%d chars stdout=0 stderr=%d",
                snapshot.request_type,
                snapshot.request_version,
                snapshot.attempt_number,
                elapsed,
                snapshot.prompt_length,
                snapshot.stderr_bytes,
            )

        if elapsed >= soft_stall_seconds and not snapshot.soft_warning_sent:
            snapshot.soft_warning_sent = True
            stdout_state = "no stdout"
            if snapshot.output_bytes > 0 and snapshot.stream_event_count > 0:
                stdout_state = "stream events"
            elif snapshot.output_bytes > 0:
                stdout_state = "partial stdout"
            elif snapshot.stderr_bytes > 0:
                stdout_state = "stderr only"

            logger.warning(
                "planner_soft_stall: type=%s version=%d attempt=%d elapsed=%.0fs state=%s stdout=%d stderr=%d prompt=%d",
                snapshot.request_type,
                snapshot.request_version,
                snapshot.attempt_number,
                elapsed,
                stdout_state,
                snapshot.output_bytes,
                snapshot.stderr_bytes,
                snapshot.prompt_length,
            )
            if not snapshot.soft_notification_sent:
                snapshot.soft_notification_sent = True
                self.notification_service.send_error(
                    (
                        f"Planner slow: {snapshot.request_type} v{snapshot.request_version} "
                        f"attempt {snapshot.attempt_number}, elapsed {elapsed:.0f}s, "
                        f"stdout={snapshot.output_bytes}B, stderr={snapshot.stderr_bytes}B, "
                        f"prompt={snapshot.prompt_length} chars"
                    ),
                    context="plan_mode",
                )

        if elapsed >= hard_stall_seconds:
            self._handle_planner_timeout(snapshot)

    def _handle_planner_timeout(self, snapshot: Any) -> None:
        final_snapshot = self.planner_service.terminate_plan_run("hard_timeout")
        if final_snapshot is None:
            return

        self._sync_planner_runtime_state(clear=True)
        self.state.planner_timeout_count += 1
        artifact_path = self._persist_planner_run(final_snapshot)
        summary = (
            f"Planner timeout: {final_snapshot.request_type} v{final_snapshot.request_version} "
            f"attempt {final_snapshot.attempt_number}, elapsed {final_snapshot.elapsed_seconds:.0f}s, "
            f"stdout={final_snapshot.output_bytes}B, stderr={final_snapshot.stderr_bytes}B"
        )

        if final_snapshot.timeout_retry_count < 1 and self.planner_service.restart_plan_request():
            logger.warning("%s — retrying once", summary)
            self.notification_service.send_error(
                f"{summary}. Retrying once. Artifact: {artifact_path}",
                context="plan_mode",
            )
            self.state.empty_cycles = 0
            self._sync_planner_runtime_state()
            return

        logger.error("%s — stopping orchestrator", summary)
        self.notification_service.send_error(
            f"{summary}. Stopping. Artifact: {artifact_path}",
            context="plan_mode",
        )
        self._terminal_stop_reason = StopReason.PLANNER_TIMEOUT
        self._terminal_stop_summary = summary

    def _persist_planner_run(self, snapshot: Any) -> str | None:
        if not self._plan_store:
            return None
        path = self._plan_store.save_planner_run(
            request_type=snapshot.request_type,
            request_version=snapshot.request_version,
            attempt_number=snapshot.attempt_number,
            payload=snapshot.to_dict(),
        )
        return str(path)

    def _handle_invalid_plan(
        self,
        plan_version: int,
        parsed_data: dict[str, Any],
        validation: PlanValidationResult,
        raw_output: str,
    ) -> None:
        """Persist rejection details and either trigger repair or stop."""
        attempt_number = max(1, self.state.current_plan_attempt or 1)
        attempt_type = self.state.current_plan_attempt_type or "create"
        first_error = validation.errors[0] if validation.errors else None
        summary = validation.summary()

        logger.warning(
            "Rejecting plan v%d attempt=%d type=%s: %d validation errors (%s)",
            plan_version,
            attempt_number,
            attempt_type,
            len(validation.errors),
            summary,
        )

        artifact_path = None
        if self._plan_store:
            artifact_path = self._plan_store.save_rejected_plan_attempt(
                plan_version=plan_version,
                attempt_number=attempt_number,
                attempt_type=attempt_type,
                raw_output=raw_output,
                parsed_data=parsed_data,
                validation_errors=validation.as_dicts(),
            )

        self.state.current_plan_attempt = attempt_number
        self.state.current_plan_attempt_type = attempt_type
        self.state.current_plan_validation_errors = validation.as_dicts()
        self.state.last_rejected_plan_version = plan_version
        self.state.last_rejected_plan_attempt_at = datetime.now(timezone.utc).isoformat()
        self.state.last_rejected_plan_artifact = str(artifact_path) if artifact_path else None
        self.state.empty_cycles = 0
        self.state.last_change_at = datetime.now(timezone.utc).isoformat()

        if first_error:
            memory_text = (
                f"Plan v{plan_version} repair needed: {len(validation.errors)} errors, "
                f"stage {first_error.stage_number} {first_error.code}: {first_error.message}"
            )
        else:
            memory_text = f"Plan v{plan_version} repair needed: validation failed"
        self.memory_service.record_event(self.state, memory_text)

        if attempt_number >= self._max_plan_attempts:
            final_message = (
                f"Planner produced invalid plan v{plan_version} "
                f"{attempt_number} times: {summary}"
            )
            self.notification_service.send_error(final_message, context="plan_mode")
            self._terminal_stop_reason = StopReason.INVALID_PLAN_LOOP
            self._terminal_stop_summary = final_message
            return

        next_attempt = attempt_number + 1
        self.notification_service.send_error(
            (
                f"Plan v{plan_version} rejected on attempt {attempt_number}. "
                f"Scheduling repair attempt {next_attempt}/{self._max_plan_attempts}. "
                f"{summary}"
            ),
            context="plan_mode",
        )
        self.state.current_plan_attempt = next_attempt
        self.state.current_plan_attempt_type = "repair"
        self._repair_plan()

    def _maybe_update_plan_baseline(self, plan_task: PlanTask, report: Any) -> None:
        """Capture the measured baseline from ETAP 0 so later plans use real metrics."""
        if not self._current_plan or plan_task.stage_number != 0 or report.status != "success":
            return

        baseline_row = report.results_table[0] if report.results_table else {}
        baseline_run_id = None
        if isinstance(baseline_row, dict):
            baseline_run_id = baseline_row.get("run_id")

        if not baseline_run_id:
            for artifact in report.artifacts:
                if isinstance(artifact, str) and artifact.startswith("run_id:"):
                    baseline_run_id = artifact.split(":", 1)[1].strip()
                    break

        baseline_snapshot_ref = None
        if isinstance(baseline_row, dict) and baseline_row.get("snapshot_id"):
            snapshot_id = str(baseline_row["snapshot_id"])
            version = baseline_row.get("version")
            baseline_snapshot_ref = f"{snapshot_id}@{version}" if version else snapshot_id
        if not baseline_snapshot_ref:
            for artifact in report.artifacts:
                if isinstance(artifact, str) and artifact.startswith("snapshot:"):
                    baseline_snapshot_ref = artifact.split(":", 1)[1].strip()
                    break

        if baseline_run_id:
            self._current_plan.baseline_run_id = baseline_run_id
        if baseline_snapshot_ref:
            self._current_plan.baseline_snapshot_ref = baseline_snapshot_ref
        if report.key_metrics:
            self._current_plan.baseline_metrics = dict(report.key_metrics)
