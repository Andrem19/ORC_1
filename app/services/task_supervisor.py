"""
Task supervisor — handles stop/retry/reassign logic for tasks.
"""

from __future__ import annotations

import logging

from app.models import (
    OrchestratorState,
    RestartReason,
    StopReason,
    Task,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger("orchestrator.task_supervisor")


class TaskSupervisor:
    """Decides when to stop, retry, or reassign tasks."""

    def __init__(
        self,
        max_task_attempts: int = 3,
        max_worker_timeout_count: int = 3,
    ) -> None:
        self.max_task_attempts = max_task_attempts
        self.max_worker_timeout_count = max_worker_timeout_count
        self._timeout_counts: dict[str, int] = {}

    def should_stop_task(self, task: Task, result: TaskResult) -> StopReason | None:
        """Return a stop reason if the task should be stopped."""
        if task.status == TaskStatus.TIMED_OUT:
            self._timeout_counts[task.task_id] = self._timeout_counts.get(task.task_id, 0) + 1
            if self._timeout_counts[task.task_id] >= self.max_worker_timeout_count:
                return StopReason.TIMEOUT

        if result.status == "error" and task.attempts >= self.max_task_attempts:
            return StopReason.MAX_ERRORS

        if result.status == "error" and "subprocess" in result.error.lower():
            return StopReason.SUBPROCESS_ERROR

        return None

    def should_retry_task(self, task: Task, result: TaskResult) -> RestartReason | None:
        """Return a restart reason if the task should be retried."""
        if result.status == "error" and task.attempts < self.max_task_attempts:
            return RestartReason.TEMPORARY_ERROR

        if result.status == "partial" and task.attempts < self.max_task_attempts:
            return RestartReason.RETRY_REQUESTED

        return None

    def prepare_retry(self, task: Task, reason: RestartReason, updated_instruction: str = "") -> None:
        """Reset a task for retry."""
        task.attempts += 1
        if updated_instruction:
            task.description = updated_instruction
        task.status = TaskStatus.PENDING
        task.assigned_worker_id = None
        logger.info("Task %s prepared for retry (attempt %d, reason: %s)", task.task_id, task.attempts, reason.value)

    def stop_task(self, task: Task, reason: StopReason) -> None:
        """Mark a task as stopped with the given reason."""
        task.mark_failed()
        logger.info("Task %s stopped: %s", task.task_id, reason.value)

    def cancel_stale_tasks(self, state: OrchestratorState) -> list[str]:
        """Cancel tasks that are no longer relevant (e.g., parent completed)."""
        cancelled_ids: list[str] = []
        for task in state.tasks:
            if task.parent_task_id and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                parent = state.find_task(task.parent_task_id)
                if parent and parent.status == TaskStatus.COMPLETED:
                    task.mark_cancelled()
                    cancelled_ids.append(task.task_id)
                    logger.info("Task %s cancelled (parent %s completed)", task.task_id, parent.task_id)
        return cancelled_ids
