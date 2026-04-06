"""
Scheduler — decides when to wake the planner, when to poll, when to sleep.

Core principle: do NOT call the planner unless something changed.
"""

from __future__ import annotations

import logging
import threading
import time

from app.models import (
    OrchestratorState,
    PlannerDecision,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger("orchestrator.scheduler")


class Scheduler:
    """Determines orchestrator timing and planner call necessity."""

    def __init__(
        self,
        poll_interval_seconds: int = 300,
        max_empty_cycles: int = 12,
        max_errors_total: int = 20,
    ) -> None:
        self.poll_interval_seconds = poll_interval_seconds
        self.max_empty_cycles = max_empty_cycles
        self.max_errors_total = max_errors_total

    def should_call_planner(
        self,
        state: OrchestratorState,
        new_results: list[TaskResult],
    ) -> bool:
        """Decide whether the planner needs to be invoked this cycle.

        Call planner when:
        - It's the very first cycle (no decision yet)
        - New results arrived from workers
        - All workers are idle (pending tasks but nothing running)
        - All tasks settled (no active, no pending) — planner must decide next
        """
        # First cycle ever (cycle counter is incremented before this check)
        if state.current_cycle == 1:
            logger.debug("Scheduler: first cycle, calling planner")
            return True

        # New results trigger planner
        if new_results:
            logger.debug("Scheduler: %d new results, calling planner", len(new_results))
            return True

        # Pending tasks but nothing active — need assignments
        if state.pending_tasks() and not state.active_tasks():
            logger.debug("Scheduler: pending tasks with no active workers, calling planner")
            return True

        # All tasks done (completed or failed), planner must decide next
        if not state.active_tasks() and not state.pending_tasks() and state.current_cycle > 1:
            logger.debug("Scheduler: all tasks settled, calling planner for next move")
            return True

        return False

    def should_stop_orchestrator(self, state: OrchestratorState) -> str | None:
        """Return a StopReason if the orchestrator should halt, else None."""
        if state.total_errors >= self.max_errors_total:
            return "max_errors"

        if state.empty_cycles >= self.max_empty_cycles:
            return "no_progress"

        if state.last_planner_decision == PlannerDecision.FINISH:
            return "goal_reached"

        return None

    # ---------------------------------------------------------------
    # Plan-mode scheduling
    # ---------------------------------------------------------------

    def should_create_plan(self, state: OrchestratorState) -> bool:
        """True if there is no current plan and we need to create one."""
        return getattr(state, "current_plan_version", 0) == 0

    def should_revise_plan(self, state: OrchestratorState) -> bool:
        """True if all dispatched plan tasks are resolved and plan needs revision."""
        plan_version = getattr(state, "current_plan_version", 0)
        if plan_version == 0:
            return False
        # All active tasks must be done (no running/waiting)
        active = state.active_tasks()
        # Only revise if there are resolved tasks (at least some work was done this round)
        resolved = [
            t for t in state.tasks
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT)
            and t.metadata.get("plan_mode")
        ]
        return len(active) == 0 and len(resolved) > 0

    def plan_tasks_to_dispatch(
        self,
        state: OrchestratorState,
        max_concurrent: int = 2,
    ) -> int:
        """How many new plan tasks can be dispatched right now."""
        active = len(state.active_tasks())
        return max(0, max_concurrent - active)

    def sleep_interval(self, state: OrchestratorState | None = None) -> int:
        """Return seconds to sleep before next cycle.

        When workers are actively running, polls more frequently to monitor
        progress. When idle, uses the full configured interval.
        """
        if state is not None and state.active_tasks():
            return min(self.poll_interval_seconds, 30)
        return self.poll_interval_seconds

    # Shared event — set by the signal handler to interrupt sleep immediately
    _wake = threading.Event()

    def sleep(self, seconds: int | None = None, state: OrchestratorState | None = None) -> None:
        """Sleep for the specified or computed interval with progress bar.

        Uses threading.Event.wait() instead of time.sleep() so that SIGINT
        (Ctrl+C) interrupts the sleep immediately — even with a custom signal
        handler installed (PEP 475 causes time.sleep to auto-retry).
        """
        if seconds is None:
            seconds = self.sleep_interval(state)
        logger.info("Sleeping for %d seconds", seconds)
        self._wake.clear()

        from app.rich_handler import ProgressManager
        pm = ProgressManager._instance
        if pm and pm.is_active():
            pm.start_sleep(seconds)
            for tick in range(seconds):
                if self._wake.wait(timeout=1.0):
                    break  # interrupted by signal
                pm.update_sleep(tick + 1)
            pm.stop_sleep()
        else:
            self._wake.wait(timeout=seconds)
