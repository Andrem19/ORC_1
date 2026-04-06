"""
Data models for the orchestrator.

All shared types, enums, and dataclasses live here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.plan_models import TaskReport


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_RESULT = "waiting_result"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    STALLED = "stalled"
    CANCELLED = "cancelled"


class PlannerDecision(str, Enum):
    LAUNCH_WORKER = "launch_worker"
    WAIT = "wait"
    RETRY_WORKER = "retry_worker"
    STOP_WORKER = "stop_worker"
    REASSIGN_TASK = "reassign_task"
    FINISH = "finish"


class StopReason(str, Enum):
    TIMEOUT = "timeout"
    PLANNER_TIMEOUT = "planner_timeout"
    NO_PROGRESS = "no_progress"
    INVALID_OUTPUT = "invalid_output"
    INVALID_PLAN_LOOP = "invalid_plan_loop"
    SUBPROCESS_ERROR = "subprocess_error"
    TASK_STALE = "task_stale"
    PLANNER_REQUESTED = "planner_requested"
    MAX_ERRORS = "max_errors"
    GOAL_REACHED = "goal_reached"
    GOAL_IMPOSSIBLE = "goal_impossible"


class RestartReason(str, Enum):
    TEMPORARY_ERROR = "temporary_error"
    RETRY_REQUESTED = "retry_requested"
    CLARIFIED_PROMPT = "clarified_prompt"
    REASSIGN = "reassign"
    WORKER_AVAILABLE = "worker_available"


class OrchestratorEvent(str, Enum):
    STARTED = "started"
    CONFIG_LOADED = "config_loaded"
    STATE_RESTORED = "state_restored"
    PLANNER_CALLED = "planner_called"
    PLANNER_RESULT = "planner_result"
    PLANNER_ERROR = "planner_error"
    WORKER_LAUNCHED = "worker_launched"
    WORKER_COMPLETED = "worker_completed"
    WORKER_FAILED = "worker_failed"
    WORKER_TIMED_OUT = "worker_timed_out"
    WORKER_STOPPED = "worker_stopped"
    NEW_RESULT = "new_result"
    NO_CHANGE = "no_change"
    STATE_SAVED = "state_saved"
    SLEEPING = "sleeping"
    FINISHED = "finished"
    ERROR = "error"
    RESTART_RECOVERY = "restart_recovery"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentDescriptor:
    agent_id: str
    role: str
    adapter_name: str
    max_concurrent_tasks: int = 1


@dataclass
class WorkerConfig:
    worker_id: str
    role: str
    system_prompt: str


@dataclass
class Task:
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    assigned_worker_id: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parent_task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_running(self) -> None:
        self.status = TaskStatus.RUNNING
        self._touch()

    def mark_waiting_result(self) -> None:
        self.status = TaskStatus.WAITING_RESULT
        self._touch()

    def mark_completed(self) -> None:
        self.status = TaskStatus.COMPLETED
        self._touch()

    def mark_failed(self) -> None:
        self.status = TaskStatus.FAILED
        self._touch()

    def mark_timed_out(self) -> None:
        self.status = TaskStatus.TIMED_OUT
        self._touch()

    def mark_stalled(self) -> None:
        self.status = TaskStatus.STALLED
        self._touch()

    def mark_cancelled(self) -> None:
        self.status = TaskStatus.CANCELLED
        self._touch()

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TaskResult:
    task_id: str
    worker_id: str
    status: str  # "success" | "error" | "partial"
    summary: str = ""
    artifacts: list[str] = field(default_factory=list)
    next_hint: str = ""
    confidence: float = 0.0
    error: str = ""
    raw_output: str = ""
    mcp_problems: list[dict[str, str]] = field(default_factory=list)
    plan_report: "TaskReport | None" = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_empty(self) -> bool:
        return not self.summary and not self.artifacts and not self.error


@dataclass
class PlannerOutput:
    decision: PlannerDecision
    target_worker_id: str | None = None
    task_instruction: str = ""
    reason: str = ""
    check_after_seconds: int = 300
    memory_update: str = ""
    should_finish: bool = False
    final_summary: str = ""
    reassign_to_worker_id: str | None = None


@dataclass
class MemoryEntry:
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = ""  # "planner" | "worker:<id>" | "system"
    tags: list[str] = field(default_factory=list)


@dataclass
class ProcessInfo:
    task_id: str
    worker_id: str
    pid: int | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    first_output_at: str | None = None
    last_output_at: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    partial_output: str = ""
    partial_error_output: str = ""
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    monitor_state: str = ""
    monitor_warning_sent: bool = False
    prompt: str = ""
    returncode: int | None = None


@dataclass
class OrchestratorState:
    goal: str = ""
    status: str = "idle"  # idle | running | finished | error
    current_cycle: int = 0
    empty_cycles: int = 0
    total_errors: int = 0
    tasks: list[Task] = field(default_factory=list)
    results: list[TaskResult] = field(default_factory=list)
    processes: list[ProcessInfo] = field(default_factory=list)
    memory: list[MemoryEntry] = field(default_factory=list)
    last_planner_decision: PlannerDecision | None = None
    last_planner_call_at: str | None = None
    planner_started_at: str | None = None
    planner_first_output_at: str | None = None
    planner_last_output_at: str | None = None
    planner_output_bytes: int = 0
    planner_stderr_bytes: int = 0
    planner_timeout_count: int = 0
    last_change_at: str | None = None
    stop_reason: StopReason | None = None
    current_plan_version: int = 0
    current_plan_attempt: int = 0
    current_plan_attempt_type: str | None = None
    current_plan_validation_errors: list[dict[str, Any]] = field(default_factory=list)
    last_rejected_plan_version: int | None = None
    last_rejected_plan_attempt_at: str | None = None
    last_rejected_plan_artifact: str | None = None
    plan_task_dispatch_map: dict[str, str] = field(default_factory=dict)  # stage_id -> task_id
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)
        self._touch()

    def find_task(self, task_id: str) -> Task | None:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def active_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status in (TaskStatus.RUNNING, TaskStatus.WAITING_RESULT)]

    def pending_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def completed_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def failed_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.STALLED)]

    def find_process(self, task_id: str) -> ProcessInfo | None:
        for p in self.processes:
            if p.task_id == task_id:
                return p
        return None

    def remove_process(self, task_id: str) -> None:
        self.processes = [p for p in self.processes if p.task_id != task_id]
        self._touch()

    def add_memory(self, entry: MemoryEntry) -> None:
        self.memory.append(entry)
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]
        self._touch()

    def get_recent_memory(self, limit: int = 10) -> list[MemoryEntry]:
        return self.memory[-limit:]
