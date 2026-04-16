"""
Data models for the orchestrator.

All shared types, enums, and dataclasses live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.plan_models import PlanReport


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StopReason(str, Enum):
    NO_PROGRESS = "no_progress"
    SUBPROCESS_ERROR = "subprocess_error"
    MAX_ERRORS = "max_errors"
    GOAL_REACHED = "goal_reached"
    GOAL_IMPOSSIBLE = "goal_impossible"
    RECOVERABLE_BLOCKED = "recoverable_blocked"
    MCP_UNHEALTHY = "mcp_unhealthy"
    GRACEFUL_STOP = "graceful_stop"


class OrchestratorEvent(str, Enum):
    STARTED = "started"
    STATE_RESTORED = "state_restored"
    STATE_SAVED = "state_saved"
    FINISHED = "finished"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WorkerConfig:
    worker_id: str
    role: str
    system_prompt: str


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
    plan_report: "PlanReport | None" = None
    result_fingerprint: str = ""
    synthetic: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Semantic fields populated from slice/action data for richer notifications
    title: str = ""                                               # human-readable slice title
    verdict: str = ""                                             # PROMOTE | WATCHLIST | REJECT | FAILED | PENDING
    findings: list[str] = field(default_factory=list)            # confirmed findings (top 3)
    key_metrics: dict[str, Any] = field(default_factory=dict)    # numeric metrics from worker
    next_actions: list[str] = field(default_factory=list)        # recommended next steps (top 3)
    sequence_label: str = ""                                      # e.g. "plan_v4 B2" — sequence + batch context

    @property
    def is_empty(self) -> bool:
        return not self.summary and not self.artifacts and not self.error

@dataclass
class DirectOrchestratorState:
    goal: str = ""
    status: str = "idle"  # idle | running | finished | error
    current_cycle: int = 0
    total_errors: int = 0
    results: list[TaskResult] = field(default_factory=list)
    last_change_at: str | None = None
    stop_reason: StopReason | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def _touch(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        self.updated_at = now_iso
        self.last_change_at = now_iso


OrchestratorState = DirectOrchestratorState
