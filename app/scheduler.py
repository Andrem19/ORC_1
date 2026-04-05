"""
Scheduler — decides when to wake the planner, when to poll, when to sleep.

Core principle: do NOT call the planner unless something changed.
"""

from __future__ import annotations

import logging
import time

from app.models import (
    OrchestratorState,
    PlannerDecision,
    TaskResult,
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
        - Error count crossed a threshold
        """
        # First cycle ever
        if state.current_cycle == 0:
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

    def sleep_interval(self, state: OrchestratorState) -> int:
        """Return seconds to sleep before next cycle.

        Uses the planner's suggested check_after_seconds if available,
        otherwise falls back to the configured poll interval.
        """
        # Default
        interval = self.poll_interval_seconds

        # Planner may have suggested a custom interval
        if state.last_planner_call_at and state.current_cycle > 0:
            # Use default poll for now — the planner's suggestion is stored
            # but we keep things simple and predictable
            pass

        return interval

    def sleep(self, seconds: int | None = None, state: OrchestratorState | None = None) -> None:
        """Sleep for the specified or computed interval."""
        if seconds is None and state is not None:
            seconds = self.sleep_interval(state)
        elif seconds is None:
            seconds = self.poll_interval_seconds

        logger.info("Sleeping for %d seconds", seconds)
        time.sleep(seconds)
