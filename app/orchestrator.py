"""
Orchestrator — the main loop.

Core principle: models are called ONLY when state changes.
The orchestrator sleeps between checks and only wakes the planner
when there is new information to act on.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.adapters.base import BaseAdapter
from app.config import OrchestratorConfig
from app.models import (
    MemoryEntry,
    OrchestratorState,
    OrchestratorEvent,
    PlannerDecision,
    PlannerOutput,
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
    ) -> None:
        self.config = config
        self.state_store = state_store
        self.planner_service = PlannerService(planner_adapter, timeout=config.planner_timeout_seconds)
        self.worker_service = WorkerService(worker_adapter, timeout=config.worker_timeout_seconds)
        self.scheduler = scheduler or Scheduler(
            poll_interval_seconds=config.poll_interval_seconds,
            max_empty_cycles=config.max_empty_cycles,
            max_errors_total=config.max_errors_total,
        )
        self.memory_service = memory_service or MemoryService()
        self.notification_service = notification_service or NotificationService()
        self.task_supervisor = task_supervisor or TaskSupervisor(
            max_task_attempts=config.max_task_attempts,
            max_worker_timeout_count=config.max_worker_timeout_count,
        )

        self.state = OrchestratorState(goal=config.goal)
        self._worker_ids = [w.worker_id for w in config.workers]
        self._research_context_text: str | None = None
        self._log_event(OrchestratorEvent.STARTED)

    def _log_event(self, event: OrchestratorEvent, detail: str = "") -> None:
        msg = f"[{event.value}] {detail}" if detail else f"[{event.value}]"
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
                self._research_context_text = format_research_context_for_planner(ctx)
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
            # Mark any running tasks as stalled (processes died during restart)
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
        if recovered:
            self._log_event(OrchestratorEvent.RESTART_RECOVERY)
            self.save_state()

    # ---------------------------------------------------------------
    # Core loop
    # ---------------------------------------------------------------

    def run(self) -> StopReason:
        """Main orchestrator loop. Returns the reason it stopped."""
        self.state.status = "running"
        self.save_state()
        self._log_event(OrchestratorEvent.CONFIG_LOADED, f"goal={self.config.goal[:80]}")
        self.notification_service.send_lifecycle("started", f"Goal: {self.config.goal[:100]}")

        while True:
            # 1. Increment cycle
            self.state.current_cycle += 1
            logger.info("=== Cycle %d ===", self.state.current_cycle)

            # 2. Collect new results from active tasks
            new_results = self._collect_results()

            # 3. Check if planner should be called
            if self.scheduler.should_call_planner(self.state, new_results):
                self._log_event(OrchestratorEvent.PLANNER_CALLED)
                # Refresh research context every 10 cycles
                if self.state.current_cycle % 10 == 0:
                    self.load_research_context()
                output = self.planner_service.consult(
                    self.state,
                    new_results=new_results if new_results else None,
                    worker_ids=self._worker_ids,
                    research_context=self._research_context_text,
                )
                self.state.last_planner_decision = output.decision
                self.state.last_planner_call_at = datetime.now(timezone.utc).isoformat()
                self.memory_service.record_planner_decision(self.state, output)
                self._log_event(OrchestratorEvent.PLANNER_RESULT, output.decision.value)
                self.notification_service.send_planner_decision(output, self.state.current_cycle)

                # Process planner decision
                self._execute_planner_decision(output)

                if output.should_finish or output.decision == PlannerDecision.FINISH:
                    reason = StopReason.GOAL_REACHED if output.should_finish else StopReason.GOAL_IMPOSSIBLE
                    self._finish(reason, output.final_summary)
                    return reason

                # Track empty planner cycles (planner said wait but nothing changed)
                if output.decision == PlannerDecision.WAIT:
                    self.state.empty_cycles += 1
                    self._log_event(OrchestratorEvent.NO_CHANGE, f"empty_cycles={self.state.empty_cycles}")
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
            self._log_event(OrchestratorEvent.SLEEPING, f"{self.config.poll_interval_seconds}s")
            self.scheduler.sleep(seconds=self.config.poll_interval_seconds)

    def _collect_results(self) -> list[TaskResult]:
        """Collect results from active/waiting tasks."""
        new_results: list[TaskResult] = []
        for task in list(self.state.active_tasks()):
            if task.status != TaskStatus.WAITING_RESULT:
                continue

            result = self.worker_service.execute_task(
                task,
                memory_entries=self.memory_service.get_context_for_worker(self.state, task.task_id),
            )

            # Check for duplicates
            prev = self._last_result_for_task(task.task_id)
            if self.config.detect_duplicate_results and is_duplicate_result(prev, result):
                logger.info("Duplicate result for task %s, skipping", task.task_id)
                continue

            if is_useless_result(result):
                logger.info("Useless result for task %s", task.task_id)
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

        if result.status == "success":
            task.mark_completed()
            self.state.last_change_at = datetime.now(timezone.utc).isoformat()
            self._log_event(OrchestratorEvent.WORKER_COMPLETED, f"task={task.task_id}")
            return

        # Error or partial — check if we should force-stop
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

    def _launch_worker(self, output: PlannerOutput) -> None:
        """Create a new task and assign it to a worker."""
        worker_id = output.target_worker_id
        if not worker_id or worker_id not in self._worker_ids:
            if self._worker_ids:
                worker_id = self._worker_ids[0]
            else:
                logger.error("No workers available")
                return

        task = Task(
            description=output.task_instruction,
            assigned_worker_id=worker_id,
        )
        task.mark_waiting_result()
        self.state.add_task(task)
        self.state.last_change_at = datetime.now(timezone.utc).isoformat()
        self.state.empty_cycles = 0
        self._log_event(OrchestratorEvent.WORKER_LAUNCHED, f"task={task.task_id} worker={worker_id}")

    def _retry_worker(self, output: PlannerOutput) -> None:
        """Retry a failed/stalled task."""
        # Find most recent failed task
        for task in reversed(self.state.tasks):
            if task.status in (TaskStatus.FAILED, TaskStatus.STALLED, TaskStatus.TIMED_OUT):
                self.task_supervisor.prepare_retry(
                    task,
                    RestartReason.RETRY_REQUESTED,
                    updated_instruction=output.task_instruction or task.description,
                )
                if output.target_worker_id:
                    task.assigned_worker_id = output.target_worker_id
                task.mark_waiting_result()
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
                task.mark_cancelled()
                self.state.remove_process(task.task_id)
                self._log_event(OrchestratorEvent.WORKER_STOPPED, f"task={task.task_id} worker={target}")

    def _reassign_task(self, output: PlannerOutput) -> None:
        """Reassign a task to a different worker."""
        new_worker = output.reassign_to_worker_id
        if not new_worker:
            logger.warning("Reassign requested but no target worker specified")
            return

        for task in reversed(self.state.tasks):
            if task.status in (
                TaskStatus.RUNNING, TaskStatus.WAITING_RESULT,
                TaskStatus.STALLED, TaskStatus.FAILED,
            ):
                task.assigned_worker_id = new_worker
                if output.task_instruction:
                    task.description = output.task_instruction
                task.mark_waiting_result()
                self.state.empty_cycles = 0
                self._log_event(OrchestratorEvent.WORKER_LAUNCHED, f"reassign task={task.task_id} to {new_worker}")
                return

    def _finish(self, reason: StopReason, summary: str = "") -> None:
        """Finish orchestrator execution."""
        self.state.status = "finished"
        self.state.stop_reason = reason
        self.memory_service.record_event(self.state, f"Orchestrator finished: {reason.value}")
        self.notification_service.send_lifecycle(
            "finished", f"Reason: {reason.value}. {summary[:200]}"
        )
        self.save_state()
        self._log_event(OrchestratorEvent.FINISHED, f"reason={reason.value} summary={summary[:100]}")
        logger.info("Orchestrator finished: %s. %s", reason.value, summary)
