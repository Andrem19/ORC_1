"""
Plan-mode data models.

Structured research plan types that support the plan-driven orchestrator loop:
plan → dispatch tasks → collect reports → revise plan → repeat.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.models import Task, TaskResult, TaskStatus


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

@dataclass
class DecisionGate:
    """Quantitative accept/reject criterion for a plan task."""
    metric: str = ""          # e.g. "pnl", "sharpe", "auc"
    threshold: float = 0.0
    comparator: str = "gte"   # "gte" | "lte" | "gt" | "lt" | "eq"
    verdict_pass: str = "PROMOTE"
    verdict_fail: str = "REJECT"


@dataclass
class AntiPattern:
    """A categorically rejected approach with evidence."""
    pattern_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    category: str = ""
    description: str = ""
    evidence_count: int = 0
    evidence_summary: str = ""
    verdict: str = "REJECTED"  # REJECTED | EXHAUSTED
    source_plan_version: int = 0


# ---------------------------------------------------------------------------
# Plan task and report
# ---------------------------------------------------------------------------

@dataclass
class PlanTask:
    """A single stage within a structured research plan."""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    plan_version: int = 0
    stage_number: int = 0
    stage_name: str = ""
    theory: str = ""
    depends_on: list[int] = field(default_factory=list)
    agent_instructions: list[str] = field(default_factory=list)
    results_table_columns: list[str] = field(default_factory=list)
    results_table_rows: list[dict[str, Any]] = field(default_factory=list)
    decision_gates: list[DecisionGate] = field(default_factory=list)
    verdict: str = "PENDING"  # PROMOTE | WATCHLIST | REJECT | PENDING
    assigned_worker_id: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
        )


@dataclass
class TaskReport:
    """Structured report returned by a worker for a plan task."""
    task_id: str = ""
    worker_id: str = ""
    plan_version: int = 0
    status: str = "success"  # success | error | partial
    what_was_requested: str = ""
    what_was_done: str = ""
    results_table: list[dict[str, Any]] = field(default_factory=list)
    key_metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    verdict: str = "PENDING"
    error: str = ""
    raw_output: str = ""
    mcp_problems: list[dict[str, str]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_empty(self) -> bool:
        return (
            not self.what_was_done
            and not self.artifacts
            and not self.error
            and not self.results_table
        )


# ---------------------------------------------------------------------------
# Research plan
# ---------------------------------------------------------------------------

@dataclass
class ResearchPlan:
    """A structured multi-task research plan — the primary planner artifact."""
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    schema_version: int = 2
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    frozen_base: str = ""
    baseline_run_id: str | None = None
    baseline_snapshot_ref: str | None = None
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    goal: str = ""
    principles: list[str] = field(default_factory=list)
    anti_patterns: list[AntiPattern] = field(default_factory=list)
    cumulative_summary: str = ""
    tasks: list[PlanTask] = field(default_factory=list)
    execution_order: list[int] = field(default_factory=list)  # stage numbers in order
    status: str = "active"  # active | completed | superseded
    plan_markdown: str = ""  # Full markdown text from planner
    previous_version_id: str | None = None

    def get_task_by_stage(self, stage_number: int) -> PlanTask | None:
        for t in self.tasks:
            if t.stage_number == stage_number:
                return t
        return None

    def dispatchable_tasks(self) -> list[PlanTask]:
        """Tasks that are PENDING and whose explicit dependencies are resolved."""
        pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        if not pending:
            return []

        resolved_stages = {t.stage_number for t in self.tasks if t.is_resolved}
        ready: list[PlanTask] = []

        for task in sorted(pending, key=lambda t: t.stage_number):
            dependencies = self._dependencies_for_stage(task.stage_number)
            if all(dep in resolved_stages for dep in dependencies):
                ready.append(task)

        return ready

    def _dependencies_for_stage(self, stage_number: int) -> list[int]:
        """Return explicit dependencies, falling back to legacy execution_order semantics."""
        task = self.get_task_by_stage(stage_number)
        if task is None:
            return []

        if self.schema_version >= 2:
            return sorted(set(task.depends_on))

        if not self.execution_order:
            return []

        deps: list[int] = []
        for earlier in self.execution_order:
            if earlier == stage_number:
                break
            deps.append(earlier)
        return deps

    def dispatched_tasks(self) -> list[PlanTask]:
        """Tasks that have been dispatched (running or waiting)."""
        return [
            t for t in self.tasks
            if t.status in (TaskStatus.RUNNING, TaskStatus.WAITING_RESULT)
        ]

    def resolved_tasks(self) -> list[PlanTask]:
        """Tasks that have completed (success or failure)."""
        return [t for t in self.tasks if t.is_resolved]

    def all_dispatched_resolved(self) -> bool:
        """True if every dispatched task is in a terminal state."""
        dispatched = [
            t for t in self.tasks
            if t.status != TaskStatus.PENDING
        ]
        return len(dispatched) > 0 and all(t.is_resolved for t in dispatched)


# ---------------------------------------------------------------------------
# Bridge functions (compatibility with existing infrastructure)
# ---------------------------------------------------------------------------

def plan_task_to_task(plan_task: PlanTask) -> Task:
    """Convert a PlanTask to a Task for the existing WorkerService."""
    instruction_parts: list[str] = []
    instruction_parts.append(f"# ETAP {plan_task.stage_number}: {plan_task.stage_name}")
    instruction_parts.append("")
    if plan_task.theory:
        instruction_parts.append(f"## Theory")
        instruction_parts.append(plan_task.theory)
        instruction_parts.append("")
    instruction_parts.append("## Instructions")
    for i, step in enumerate(plan_task.agent_instructions, 1):
        instruction_parts.append(f"{i}. {step}")
    if plan_task.results_table_columns:
        instruction_parts.append("")
        instruction_parts.append("## Results table columns to fill")
        instruction_parts.append(" | ".join(plan_task.results_table_columns))
    return Task(
        task_id=plan_task.task_id,
        description="\n".join(instruction_parts),
        assigned_worker_id=plan_task.assigned_worker_id,
        metadata={
            "plan_version": plan_task.plan_version,
            "stage_number": plan_task.stage_number,
            "plan_mode": True,
        },
    )


def task_report_to_task_result(report: TaskReport) -> TaskResult:
    """Convert a TaskReport to a TaskResult for compatibility."""
    return TaskResult(
        task_id=report.task_id,
        worker_id=report.worker_id,
        status=report.status,
        summary=report.what_was_done[:1000],
        artifacts=report.artifacts,
        confidence=report.confidence,
        error=report.error,
        raw_output=report.raw_output,
        mcp_problems=report.mcp_problems,
        plan_report=report,
    )
