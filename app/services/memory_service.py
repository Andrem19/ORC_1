"""
Memory service — manages short-term memory for the orchestrator.

Provides bounded, relevant context to the planner without
sending the full conversation history.
"""

from __future__ import annotations

import logging

from app.models import (
    MemoryEntry,
    OrchestratorState,
    PlannerOutput,
    TaskResult,
)

logger = logging.getLogger("orchestrator.memory_service")


class MemoryService:
    """Manages concise memory entries for cross-cycle context."""

    def __init__(self, max_entries: int = 50) -> None:
        self.max_entries = max_entries

    def record_planner_decision(self, state: OrchestratorState, output: PlannerOutput) -> None:
        """Record a planner decision as a memory entry."""
        content = f"Decision: {output.decision.value}"
        if output.reason:
            content += f" | Reason: {output.reason[:150]}"
        if output.memory_update:
            content += f" | Note: {output.memory_update[:150]}"

        entry = MemoryEntry(
            content=content,
            source="planner",
            tags=["decision", output.decision.value],
        )
        state.add_memory(entry)

    def record_worker_result(self, state: OrchestratorState, result: TaskResult) -> None:
        """Record a worker result as a memory entry."""
        content = f"Worker {result.worker_id} | Task {result.task_id} | {result.status}"
        if result.summary:
            content += f" | {result.summary[:100]}"

        entry = MemoryEntry(
            content=content,
            source=f"worker:{result.worker_id}",
            tags=["result", result.status],
        )
        state.add_memory(entry)

    def record_error(self, state: OrchestratorState, error: str, source: str = "system") -> None:
        """Record an error."""
        entry = MemoryEntry(
            content=f"ERROR: {error[:200]}",
            source=source,
            tags=["error"],
        )
        state.add_memory(entry)

    def record_event(self, state: OrchestratorState, event: str, source: str = "system") -> None:
        """Record a general event."""
        entry = MemoryEntry(
            content=event[:200],
            source=source,
            tags=["event"],
        )
        state.add_memory(entry)

    def get_context_for_planner(self, state: OrchestratorState, limit: int = 5) -> list[MemoryEntry]:
        """Get the most relevant memory entries for the planner."""
        return state.get_recent_memory(limit=limit)

    def get_context_for_worker(self, state: OrchestratorState, task_id: str, limit: int = 3) -> list[MemoryEntry]:
        """Get relevant memory entries for a worker task."""
        # Recent entries, excluding errors from other workers
        entries = state.get_recent_memory(limit=limit * 2)
        return [
            e for e in entries
            if "error" not in e.tags or f"worker" in e.source
        ][:limit]
