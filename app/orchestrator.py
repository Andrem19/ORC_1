"""
Orchestrator — the main loop.

Core principle: models are called ONLY when state changes.
The orchestrator sleeps between checks and only wakes the planner
when there is new information to act on.
Both planner and workers run as background processes — no blocking.
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime, timezone
from typing import Any

from app.adapters.base import BaseAdapter
from app.config import OrchestratorConfig
from app.mcp_problem_scanner import McpProblemScanner
from app.mcp_problem_store import McpProblemStore
from app.models import (
    MemoryEntry,
    OrchestratorState,
    OrchestratorEvent,
    PlannerDecision,
    PlannerOutput,
    ProcessInfo,
    RestartReason,
    StopReason,
    Task,
    TaskResult,
    TaskStatus,
)
from app.result_parser import is_duplicate_result, is_useless_result
from app.scheduler import Scheduler
from app.services.memory_service import MemoryService
from app.services.notification_service import NotificationService
from app.services.planner_service import PlannerService
from app.services.task_supervisor import TaskSupervisor
from app.services.worker_service import WorkerService
from app.state_store import StateStore

logger = logging.getLogger("orchestrator")


class Orchestrator:
    """Main orchestrator that drives the planner-worker loop."""

    def __init__(
        self,
        config: OrchestratorConfig,
        state_store: StateStore,
        planner_adapter: BaseAdapter,
        worker_adapter: BaseAdapter,
        scheduler: Scheduler | None = None,
        memory_service: MemoryService | None = None,
        task_supervisor: TaskSupervisor | None = None,
        notification_service: NotificationService | None = None,
        mcp_problem_scanner: McpProblemScanner | None = None,
    ) -> None:
        self.config = config
        self.state_store = state_store
        self.planner_service = PlannerService(
            planner_adapter,
            timeout=config.planner_timeout_seconds,
            planner_system_prompt=config.planner_system_prompt,
            operator_directives=config.operator_directives,
            stages_guidance=config.plan_stages_guidance,
        )
        self.worker_service = WorkerService(worker_adapter, timeout=config.worker_timeout_seconds)
        self.scheduler = scheduler or Scheduler(
            poll_interval_seconds=config.poll_interval_seconds,
            max_empty_cycles=config.max_empty_cycles,
            max_errors_total=config.max_errors_total,
            max_mcp_failures=config.max_mcp_failures,
        )
        self.memory_service = memory_service or MemoryService()
        self.notification_service = notification_service or NotificationService()
        self.task_supervisor = task_supervisor or TaskSupervisor(
            max_task_attempts=config.max_task_attempts,
            max_worker_timeout_count=config.max_worker_timeout_count,
        )

        # MCP problem scanner (optional, created from config if enabled)
        if mcp_problem_scanner:
            self._mcp_scanner = mcp_problem_scanner
        elif config.mcp_review.enabled:
            problem_store = McpProblemStore(config.mcp_review.fixes_dir)
            self._mcp_scanner = McpProblemScanner(
                planner_adapter=planner_adapter,
                problem_store=problem_store,
                review_every_n_cycles=config.mcp_review.review_every_n_cycles,
                review_after_n_errors=config.mcp_review.review_after_n_errors,
                planner_timeout=config.mcp_review.planner_timeout,
            )
        else:
            self._mcp_scanner = None

        self.state = OrchestratorState(goal=config.goal)
        self._finish_completed = False
        self._worker_ids = [w.worker_id for w in config.workers]
        self._next_worker_idx: int = 0
        self._research_context_text: str | None = None
        self._log_event(OrchestratorEvent.STARTED)

        # Graceful stop flag
        self._stop_requested: bool = False
        self._drain_mode: bool = False
        self._drain_started_at: float | None = None
        self._plan_service: "PlanOrchestratorService | None" = None

        # Plan mode (structured research plans)
        self._plan_store: "PlanStore | None" = None
        self._current_plan: "ResearchPlan | None" = None
        if config.plan_mode:
            from app.plan_store import PlanStore
            self._plan_store = PlanStore(config.plan_dir)
            self._plan_store.ensure_dirs()

    # Events that are too frequent to log at INFO every cycle
    _DEBUG_EVENTS = frozenset({OrchestratorEvent.STATE_SAVED, OrchestratorEvent.SLEEPING})

    def _log_event(self, event: OrchestratorEvent, detail: str = "") -> None:
        msg = f"[{event.value}] {detail}" if detail else f"[{event.value}]"
        if event in self._DEBUG_EVENTS:
            logger.debug(msg)
        else:
            logger.info(msg)

    # ---------------------------------------------------------------
    # Research context (MCP dev_space1 integration)
    # ---------------------------------------------------------------

    def load_research_context(self) -> None:
        """Load MCP research context from state/research_context.json."""
        if not self.config.research_config:
            return
        try:
            from app.research_context import format_research_context_for_planner, load_research_context
            ctx = load_research_context(self.config.state_dir)
            if ctx:
                summaries = [
                    r.summary for r in self.state.results
                    if r.status == "success" and r.summary
                ][-30:]
                self._research_context_text = format_research_context_for_planner(
                    ctx, completed_summaries=summaries,
                )
                logger.info("Research context loaded (%d chars)", len(self._research_context_text))
            else:
                logger.warning("No research context file found")
        except Exception as e:
            logger.error("Failed to load research context: %s", e)

    # ---------------------------------------------------------------
    # State management
    # ---------------------------------------------------------------

    def load_state(self) -> bool:
        """Load saved state. Returns True if state was restored."""
        saved = self.state_store.load()
        if saved is not None:
            self.state = saved
            self._log_event(OrchestratorEvent.STATE_RESTORED, f"cycle={saved.current_cycle}")
            self._recover_from_restart()
            return True
        return False

    def save_state(self) -> None:
        self.state_store.save(self.state)
        self._log_event(OrchestratorEvent.STATE_SAVED)

    def _recover_from_restart(self) -> None:
        """After restart, mark running tasks as stalled since subprocesses are gone."""
        recovered = False
        for task in self.state.tasks:
            if task.status in (TaskStatus.RUNNING, TaskStatus.WAITING_RESULT):
                task.mark_stalled()
                self.memory_service.record_event(
                    self.state,
                    f"Task {task.task_id} marked stalled after restart recovery",
                )
                recovered = True
        # Clear stale process info (Popen handles are lost on restart)
        if self.state.processes:
            self.state.processes.clear()
            recovered = True
        if recovered:
            self._log_event(OrchestratorEvent.RESTART_RECOVERY)
            self.save_state()

    # ---------------------------------------------------------------
    # Core loop
    # ---------------------------------------------------------------

    def run(self) -> StopReason:
        """Main orchestrator loop. Returns the reason it stopped."""
        if self.config.plan_mode:
            return self._run_plan_mode()

        self.state.status = "running"
        self.save_state()
        self._log_event(OrchestratorEvent.CONFIG_LOADED, f"goal={self.config.goal[:80]}")
        self.notification_service.send_lifecycle("started", f"Goal: {self.config.goal[:100]}")

        while True:
            # 1. Increment cycle
            self.state.current_cycle += 1
            cycle_start = _time.monotonic()
            logger.info("=== Cycle %d ===", self.state.current_cycle)

            # 1b. Drain mode: wait for running tasks to finish
            if self._drain_mode:
                active = self.state.active_tasks()
                elapsed = _time.monotonic() - (self._drain_started_at or _time.monotonic())
                timeout = self.config.drain_timeout_seconds
                if not active:
                    logger.info("Drain complete — all running tasks finished")
                    self._finish(StopReason.GRACEFUL_STOP, "Graceful drain completed")
                    return StopReason.GRACEFUL_STOP
                if elapsed >= timeout:
                    logger.warning(
                        "Drain timeout (%ds) exceeded with %d tasks still running — forcing stop",
                        timeout, len(active),
                    )
                    for task in active:
                        if task.status == TaskStatus.RUNNING:
                            self.worker_service.terminate_task(task.task_id)
                    self._finish(
                        StopReason.GRACEFUL_STOP,
                        f"Drain timeout ({timeout}s), {len(active)} tasks terminated",
                    )
                    return StopReason.GRACEFUL_STOP
                logger.info(
                    "Drain mode: %d tasks still running (elapsed %.0fs / %ds timeout)",
                    len(active), elapsed, timeout,
                )
                # Collect results from running workers, save state, sleep
                self._collect_results()
                self.save_state()
                sleep_seconds = self.scheduler.sleep_interval(self.state)
                self._log_event(OrchestratorEvent.SLEEPING, f"{sleep_seconds}s")
                self.scheduler.sleep(seconds=sleep_seconds)
                continue

            # 2. Check on running workers (non-blocking)
            new_results = self._collect_results()

            # 2.5. MCP problem review
            mcp_problem_summary = None
            if self._mcp_scanner:
                for r in new_results:
                    worker_problems = self._mcp_scanner.extract_worker_problems(r)
                    if worker_problems:
                        for p in worker_problems:
                            p.cycle = self.state.current_cycle
                        self._mcp_scanner.problem_store.save_report(
                            worker_problems, cycle=self.state.current_cycle,
                        )

                if self._mcp_scanner.should_review(self.state, new_results):
                    problems = self._mcp_scanner.run_review(
                        self.state, self.state.results[-20:],
                    )
                    if problems:
                        self._log_event(
                            OrchestratorEvent.ERROR,
                            f"mcp_review: {len(problems)} problems found",
                        )

                # Heuristic structural problem detection (cheap, runs every cycle)
                heuristic_problems = self._mcp_scanner.detect_structural_problems(self.state)
                if heuristic_problems:
                    self._mcp_scanner.problem_store.save_report(
                        heuristic_problems, cycle=self.state.current_cycle,
                    )
                    self._log_event(
                        OrchestratorEvent.ERROR,
                        f"heuristic: {len(heuristic_problems)} structural problems detected",
                    )

                mcp_problem_summary = self._mcp_scanner.get_context_for_planner(
                    max_items=self.config.mcp_review.max_problems_in_context,
                )

            # 3. Planner: check running OR start new consultation
            if self.planner_service.is_running:
                # Planner is already running — check on it (non-blocking)
                output, is_finished = self.planner_service.check_consultation()
                if is_finished and output is not None:
                    # Stop planner spinner
                    from app.rich_handler import ProgressManager
                    pm = ProgressManager._instance
                    if pm and pm.is_active():
                        pm.stop_planner_wait()
                    self._process_planner_output(output)
                    if self._should_stop_after_planner(output):
                        return self._stop_reason_from_planner(output)
                # else: planner still working, will check again next cycle
            elif self.scheduler.should_call_planner(self.state, new_results):
                # No planner running — start one
                self._log_event(OrchestratorEvent.PLANNER_CALLED)
                # Progress tracking — planner spinner
                from app.rich_handler import ProgressManager
                pm = ProgressManager._instance
                if pm and pm.is_active():
                    pm.start_planner_wait(
                        model=self.config.planner_adapter.model,
                        action="Analyzing state and deciding next action",
                    )
                if self.state.current_cycle % 10 == 0:
                    self.load_research_context()
                self.planner_service.start_consultation(
                    self.state,
                    new_results=new_results if new_results else None,
                    worker_ids=self._worker_ids,
                    research_context=self._research_context_text,
                    mcp_problem_summary=mcp_problem_summary,
                )
                # Don't increment empty_cycles — we just started the planner
            else:
                self.state.empty_cycles += 1
                self._log_event(OrchestratorEvent.NO_CHANGE, f"empty_cycles={self.state.empty_cycles}")

            # 4. Check stop conditions
            stop_reason = self.scheduler.should_stop_orchestrator(self.state)
            if stop_reason:
                self._finish(StopReason(stop_reason))
                return StopReason(stop_reason)

            # 5. Save state and sleep
            self.save_state()
            cycle_elapsed = _time.monotonic() - cycle_start
            logger.debug(
                "Cycle %d completed in %.1fs — tasks=%d results=%d errors=%d memory=%d empty=%d planner_running=%s",
                self.state.current_cycle, cycle_elapsed,
                len(self.state.tasks), len(self.state.results),
                self.state.total_errors, len(self.state.memory),
                self.state.empty_cycles,
                self.planner_service.is_running,
            )
            sleep_seconds = self.scheduler.sleep_interval(self.state)
            self._log_event(OrchestratorEvent.SLEEPING, f"{sleep_seconds}s")
            self.scheduler.sleep(seconds=sleep_seconds)

    # ---------------------------------------------------------------
    # Planner output handling
    # ---------------------------------------------------------------

    def _process_planner_output(self, output: PlannerOutput) -> None:
        """Process a completed planner output."""
        self.state.last_planner_decision = output.decision
        self.state.last_planner_call_at = datetime.now(timezone.utc).isoformat()
        self.memory_service.record_planner_decision(self.state, output)
        logger.debug(
            "Planner full decision: decision=%s worker=%s instruction='%s' reason='%s' check_after=%ds memory='%s'",
            output.decision.value,
            output.target_worker_id or "N/A",
            output.task_instruction[:200] if output.task_instruction else "",
            output.reason[:200],
            output.check_after_seconds,
            output.memory_update[:100] if output.memory_update else "",
        )
        self._log_event(OrchestratorEvent.PLANNER_RESULT, output.decision.value)
        self.notification_service.send_planner_decision(output, self.state.current_cycle)
        self._execute_planner_decision(output)

        if output.decision == PlannerDecision.WAIT:
            self.state.empty_cycles += 1
            self._log_event(OrchestratorEvent.NO_CHANGE, f"empty_cycles={self.state.empty_cycles}")

    def _should_stop_after_planner(self, output: PlannerOutput) -> bool:
        return output.should_finish or output.decision == PlannerDecision.FINISH

    def _stop_reason_from_planner(self, output: PlannerOutput) -> StopReason:
        reason = StopReason.GOAL_REACHED if output.should_finish else StopReason.GOAL_IMPOSSIBLE
        self._finish(reason, output.final_summary)
        return reason

    # ---------------------------------------------------------------
    # Result collection (non-blocking)
    # ---------------------------------------------------------------

    def _collect_results(self) -> list[TaskResult]:
        """Check on running workers and collect any completed results. Non-blocking."""
        new_results: list[TaskResult] = []

        for task in list(self.state.active_tasks()):
            if task.status != TaskStatus.RUNNING:
                continue

            process_info = self.state.find_process(task.task_id)
            if process_info is None:
                logger.warning("Task %s is RUNNING but has no process info", task.task_id)
                task.mark_failed()
                self.state.total_errors += 1
                continue

            result, is_finished = self.worker_service.check_task(task, process_info)

            if not is_finished:
                output_len = len(process_info.partial_output or "")
                elapsed_min = 0.0
                try:
                    started = datetime.fromisoformat(process_info.started_at)
                    elapsed_min = (datetime.now(timezone.utc) - started).total_seconds() / 60
                except (ValueError, TypeError):
                    pass
                logger.info(
                    "Task %s still running (pid=%s, output_so_far=%d chars, elapsed=%.1fmin)",
                    task.task_id, process_info.pid, output_len, elapsed_min,
                )
                continue

            # Worker finished — clean up process info
            self.state.remove_process(task.task_id)

            if result is None:
                logger.error("Task %s finished but no result returned", task.task_id)
                task.mark_failed()
                self.state.total_errors += 1
                continue

            # Check for duplicates
            prev = self._last_result_for_task(task.task_id)
            if self.config.detect_duplicate_results and is_duplicate_result(prev, result):
                logger.info("Duplicate result for task %s, skipping", task.task_id)
                continue

            if is_useless_result(result):
                logger.warning(
                    "Useless result for task %s (status=%s, confidence=%.2f)",
                    task.task_id, result.status, result.confidence,
                )
                self.state.total_errors += 1

            # Handle result
            self._handle_task_result(task, result)
            new_results.append(result)

        return new_results

    def _last_result_for_task(self, task_id: str) -> TaskResult | None:
        """Find the last result for a task."""
        for r in reversed(self.state.results):
            if r.task_id == task_id:
                return r
        return None

    def _handle_task_result(self, task: Task, result: TaskResult) -> None:
        """Process a task result: complete, retry, or fail the task."""
        self.state.results.append(result)
        self.memory_service.record_worker_result(self.state, result)
        self.notification_service.send_worker_result(result, self.state.current_cycle)

        # Remove worker progress tracker
        from app.rich_handler import ProgressManager
        pm = ProgressManager._instance
        if pm and pm.is_active():
            pm.remove_worker(task.task_id)

        if result.status == "success":
            task.mark_completed()
            self.state.last_change_at = datetime.now(timezone.utc).isoformat()
            self._log_event(OrchestratorEvent.WORKER_COMPLETED, f"task={task.task_id}")
            return

        # Plan-mode partial results are treated as completed by the plan orchestrator.
        # We must mark the Task COMPLETED here so Task and PlanTask statuses agree,
        # preventing _reconcile_current_plan_state from overwriting the PlanTask later.
        if task.metadata.get("plan_mode") and result.status == "partial":
            task.mark_completed()
            self.state.last_change_at = datetime.now(timezone.utc).isoformat()
            self._log_event(OrchestratorEvent.WORKER_COMPLETED, f"task={task.task_id} (partial)")
            return

        # Error — check if we should force-stop
        stop_reason = self.task_supervisor.should_stop_task(task, result)
        if stop_reason:
            self.task_supervisor.stop_task(task, stop_reason)
            self.state.total_errors += 1
            self._log_event(OrchestratorEvent.WORKER_STOPPED, f"task={task.task_id} reason={stop_reason.value}")
            return

        # Mark failed so planner can decide to retry or not
        task.mark_failed()
        self.state.total_errors += 1
        self._log_event(OrchestratorEvent.WORKER_FAILED, f"task={task.task_id} failed (attempt {task.attempts})")

    # ---------------------------------------------------------------
    # Planner decision execution
    # ---------------------------------------------------------------

    def _execute_planner_decision(self, output: PlannerOutput) -> None:
        """Execute the action prescribed by the planner."""
        decision = output.decision

        if decision == PlannerDecision.LAUNCH_WORKER:
            self._launch_worker(output)

        elif decision == PlannerDecision.RETRY_WORKER:
            self._retry_worker(output)

        elif decision == PlannerDecision.STOP_WORKER:
            self._stop_worker(output)

        elif decision == PlannerDecision.REASSIGN_TASK:
            self._reassign_task(output)

        elif decision == PlannerDecision.WAIT:
            logger.info("Planner says wait. No action this cycle.")

        elif decision == PlannerDecision.FINISH:
            pass  # handled in run()

    def _start_task_on_worker(self, task: Task) -> None:
        """Launch a worker as a background process for the given task."""
        process_info = self.worker_service.start_task(
            task,
            memory_entries=self.memory_service.get_context_for_worker(self.state, task.task_id),
        )
        self.state.processes.append(process_info)
        # Progress tracking
        from app.rich_handler import ProgressManager
        pm = ProgressManager._instance
        if pm and pm.is_active():
            pm.add_worker(
                task.task_id, task.assigned_worker_id,
                pid=process_info.pid,
                description=task.description,
            )

    def _launch_worker(self, output: PlannerOutput) -> None:
        """Create a new task and launch it as a background worker."""
        worker_id = output.target_worker_id
        if not worker_id or worker_id not in self._worker_ids:
            if self._worker_ids:
                # Prefer idle workers, then round-robin
                active_workers = {t.assigned_worker_id for t in self.state.active_tasks()}
                idle = [w for w in self._worker_ids if w not in active_workers]
                if idle:
                    worker_id = idle[self._next_worker_idx % len(idle)]
                else:
                    worker_id = self._worker_ids[self._next_worker_idx % len(self._worker_ids)]
                self._next_worker_idx += 1
            else:
                logger.error("No workers available")
                return

        logger.debug(
            "Launching worker '%s' with instruction: %s",
            worker_id, output.task_instruction[:300],
        )

        task = Task(
            description=output.task_instruction,
            assigned_worker_id=worker_id,
        )
        task.mark_running()
        self.state.add_task(task)
        self._start_task_on_worker(task)
        self.state.last_change_at = datetime.now(timezone.utc).isoformat()
        self.state.empty_cycles = 0
        self._log_event(OrchestratorEvent.WORKER_LAUNCHED, f"task={task.task_id} worker={worker_id}")

    def _retry_worker(self, output: PlannerOutput) -> None:
        """Retry a failed/stalled task."""
        for task in reversed(self.state.tasks):
            if task.status in (TaskStatus.FAILED, TaskStatus.STALLED, TaskStatus.TIMED_OUT):
                self.task_supervisor.prepare_retry(
                    task,
                    RestartReason.RETRY_REQUESTED,
                    updated_instruction=output.task_instruction or task.description,
                )
                if output.target_worker_id:
                    task.assigned_worker_id = output.target_worker_id
                task.mark_running()
                self._start_task_on_worker(task)
                self.state.empty_cycles = 0
                self._log_event(OrchestratorEvent.WORKER_LAUNCHED, f"retry task={task.task_id}")
                return

        logger.warning("Retry requested but no failed tasks found")

    def _stop_worker(self, output: PlannerOutput) -> None:
        """Stop tasks assigned to a specific worker."""
        target = output.target_worker_id
        for task in self.state.tasks:
            if task.assigned_worker_id == target and task.status in (
                TaskStatus.RUNNING, TaskStatus.WAITING_RESULT, TaskStatus.FAILED, TaskStatus.STALLED,
            ):
                if task.status == TaskStatus.RUNNING:
                    self.worker_service.terminate_task(task.task_id)
                    self.state.remove_process(task.task_id)
                task.mark_cancelled()
                self._log_event(OrchestratorEvent.WORKER_STOPPED, f"task={task.task_id} worker={target}")

    def _reassign_task(self, output: PlannerOutput) -> None:
        """Reassign a task to a different worker."""
        new_worker = output.reassign_to_worker_id
        if not new_worker:
            logger.warning("Reassign requested but no target worker specified")
            return

        for task in reversed(self.state.tasks):
            if task.status in (
                TaskStatus.STALLED, TaskStatus.FAILED,
            ):
                if self.state.find_process(task.task_id):
                    self.worker_service.terminate_task(task.task_id)
                    self.state.remove_process(task.task_id)
                task.assigned_worker_id = new_worker
                if output.task_instruction:
                    task.description = output.task_instruction
                task.mark_running()
                self._start_task_on_worker(task)
                self.state.empty_cycles = 0
                self._log_event(OrchestratorEvent.WORKER_LAUNCHED, f"reassign task={task.task_id} to {new_worker}")
                return

        logger.warning("Reassign requested but no eligible tasks found")

    def _finish(self, reason: StopReason, summary: str = "") -> None:
        """Finish orchestrator execution, collecting partial output where possible."""
        if self._finish_completed:
            return
        self._finish_completed = True
        for task in self.state.active_tasks():
            if task.status != TaskStatus.RUNNING:
                continue
            # Try to collect any buffered output before terminating
            process_info = self.state.find_process(task.task_id)
            if process_info is not None:
                try:
                    result, is_finished = self.worker_service.check_task(task, process_info)
                    if is_finished and result is not None:
                        self.state.remove_process(task.task_id)
                        self.state.results.append(result)
                        task.mark_completed()
                        continue
                except Exception:
                    pass
                # Save partial output if available and parseable
                partial = process_info.partial_output or ""
                if partial.strip() and task.metadata.get("plan_mode"):
                    try:
                        from app.result_parser import parse_task_report
                        from app.plan_models import task_report_to_task_result
                        report = parse_task_report(
                            partial,
                            task_id=task.task_id,
                            worker_id=task.assigned_worker_id or "unknown",
                            plan_version=task.metadata.get("plan_version", 0),
                        )
                        if report.what_was_done:
                            report.status = "partial"
                            result = task_report_to_task_result(report)
                            self.state.results.append(result)
                    except Exception:
                        pass

            self.worker_service.terminate_task(task.task_id)
            task.mark_interrupted()
        self.state.processes.clear()

        self.state.status = "finished"
        self.state.stop_reason = reason
        self.memory_service.record_event(self.state, f"Orchestrator finished: {reason.value}")
        self.notification_service.flush()
        self.notification_service.send_lifecycle(
            "finished", f"Reason: {reason.value}. {summary[:200]}"
        )
        self.save_state()
        self._log_event(OrchestratorEvent.FINISHED, f"reason={reason.value} summary={summary[:100]}")
        logger.info("Orchestrator finished: %s. %s", reason.value, summary)

    def request_stop(self) -> None:
        """Signal the orchestrator to stop gracefully at the next cycle."""
        self._stop_requested = True
        if self._plan_service is not None:
            self._plan_service._stop_requested = True
        logger.info("Stop requested via signal")

    def request_drain(self) -> None:
        """Enter drain mode: stop dispatching new tasks, let running tasks finish."""
        self._drain_mode = True
        self._drain_started_at = _time.monotonic()
        if self._plan_service is not None:
            self._plan_service._drain_mode = True
            self._plan_service._drain_started_at = self._drain_started_at
        active_count = len(self.state.active_tasks())
        logger.info(
            "Drain mode requested — waiting for %d running tasks to finish",
            active_count,
        )
        self.notification_service.flush()
        self.notification_service.send_lifecycle(
            "draining",
            f"Drain mode: waiting for {active_count} running tasks to finish",
        )

    # ---------------------------------------------------------------
    # Plan-driven mode (delegates to PlanOrchestratorService)
    # ---------------------------------------------------------------

    def _run_plan_mode(self) -> StopReason:
        """Plan-driven orchestrator loop — delegates to PlanOrchestratorService."""
        from app.services.plan_orchestrator_service import PlanOrchestratorService

        svc = PlanOrchestratorService(orch=self)
        self._plan_service = svc
        svc.set_research_context(self._research_context_text)
        try:
            return svc.run()
        except KeyboardInterrupt:
            self._finish(StopReason.NO_PROGRESS, "Interrupted by user (Ctrl+C)")
            return StopReason.NO_PROGRESS
        finally:
            self._plan_service = None

    def _get_mcp_summary(self) -> str | None:
        """Get MCP problem summary if scanner is available."""
        if self._mcp_scanner:
            return self._mcp_scanner.get_context_for_planner(
                max_items=self.config.mcp_review.max_problems_in_context,
            )
        return None
