"""
Prompt construction for planner and workers.

Builds the structured prompts that are sent to models.
All prompts enforce JSON output for reliability.
"""

from __future__ import annotations

from app.models import (
    MemoryEntry,
    OrchestratorState,
    PlannerDecision,
    Task,
    TaskResult,
)


PLANNER_JSON_SCHEMA = """{
  "decision": "launch_worker|wait|retry_worker|stop_worker|reassign_task|finish",
  "target_worker_id": "worker id or null",
  "task_instruction": "instruction text for the worker",
  "reason": "why this decision",
  "check_after_seconds": 300,
  "memory_update": "brief note to remember",
  "should_finish": false,
  "final_summary": "summary if finishing, empty otherwise",
  "reassign_to_worker_id": "worker id for reassign, or null"
}"""

WORKER_JSON_SCHEMA = """{
  "status": "success|error|partial",
  "summary": "what was done",
  "artifacts": ["list of produced files or items"],
  "next_hint": "suggestion for next step",
  "confidence": 0.9,
  "error": "error message if any, empty otherwise"
}"""


def build_planner_prompt(
    state: OrchestratorState,
    new_results: list[TaskResult] | None = None,
    worker_ids: list[str] | None = None,
) -> str:
    """Build the full prompt for the planner model."""
    parts: list[str] = []

    parts.append("## Global Goal")
    parts.append(state.goal)
    parts.append("")

    # Concise memory
    recent = state.get_recent_memory(limit=5)
    if recent:
        parts.append("## Recent Memory")
        for m in recent:
            parts.append(f"- [{m.source}] {m.content}")
        parts.append("")

    # Current state summary
    active = state.active_tasks()
    pending = state.pending_tasks()
    completed = state.completed_tasks()
    failed = state.failed_tasks()

    parts.append("## Current State")
    parts.append(f"Cycle: {state.current_cycle}")
    parts.append(f"Active tasks: {len(active)}")
    parts.append(f"Pending tasks: {len(pending)}")
    parts.append(f"Completed tasks: {len(completed)}")
    parts.append(f"Failed tasks: {len(failed)}")
    parts.append(f"Total errors: {state.total_errors}")
    parts.append(f"Empty cycles: {state.empty_cycles}")
    parts.append("")

    # Active tasks detail
    if active:
        parts.append("## Active Tasks")
        for t in active:
            parts.append(f"- [{t.task_id}] {t.description[:120]} (assigned to {t.assigned_worker_id}, status: {t.status.value})")
        parts.append("")

    # New results
    if new_results:
        parts.append("## New Results")
        for r in new_results:
            parts.append(f"- Task {r.task_id} by {r.worker_id}: status={r.status}, confidence={r.confidence}")
            if r.summary:
                parts.append(f"  Summary: {r.summary[:200]}")
            if r.error:
                parts.append(f"  Error: {r.error[:200]}")
        parts.append("")

    # Available workers
    if worker_ids:
        parts.append("## Available Workers")
        parts.append(", ".join(worker_ids))
        parts.append("")

    # Instructions
    parts.append("## Required Output Format")
    parts.append("Respond ONLY with a JSON object matching this schema:")
    parts.append("```json")
    parts.append(PLANNER_JSON_SCHEMA)
    parts.append("```")
    parts.append("")

    parts.append("## Decision Rules")
    parts.append("- Choose `launch_worker` to start a new task on an available worker.")
    parts.append("- Choose `wait` if workers are still running and no action needed.")
    parts.append("- Choose `retry_worker` if a task failed but should be retried.")
    parts.append("- Choose `stop_worker` if a worker is stuck or misbehaving.")
    parts.append("- Choose `reassign_task` to move a task to a different worker.")
    parts.append("- Choose `finish` if the goal is reached or further work is pointless.")

    return "\n".join(parts)


def build_worker_prompt(
    task: Task,
    memory_entries: list[MemoryEntry] | None = None,
) -> str:
    """Build the prompt for a worker agent."""
    parts: list[str] = []

    parts.append("## Task")
    parts.append(f"Task ID: {task.task_id}")
    parts.append(task.description)
    parts.append("")

    if memory_entries:
        parts.append("## Context")
        for m in memory_entries:
            parts.append(f"- {m.content}")
        parts.append("")

    parts.append("## Required Output Format")
    parts.append("Respond ONLY with a JSON object matching this schema:")
    parts.append("```json")
    parts.append(WORKER_JSON_SCHEMA)
    parts.append("```")

    return "\n".join(parts)
