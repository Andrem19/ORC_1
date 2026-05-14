"""Legacy archival plan models retained for reset/archive compatibility."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

@dataclass
class PlanExecutionSlice:
    """One bounded ETAP execution slice within a plan."""

    slice_index: int
    etap_number: int
    title: str = ""
    markdown: str = ""
    status: str = "pending"  # pending | running | completed | failed | skipped
    task_id: str | None = None
    checkpoint_summary: str = ""
    completed_artifacts: list[str] = field(default_factory=list)
    worker_attempt_seq: int = 0
    continuation_of_task_id: str | None = None
    terminal_reason: str = ""
    recovery_count: int = 0
    recovery_history: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_artifacts: list[str] = field(default_factory=list)
    last_resume_checkpoint_hash: str = ""
    last_terminal_failure_class: str = ""
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    terminalized_at: str | None = None
    terminalization_reason: str = ""
    synthetic_report_used: bool = False
    terminal_report_path: str = ""


@dataclass
class Plan:
    """One markdown research plan executed by one worker."""

    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    version: int = 0
    wave_id: int = 0
    wave_target_size: int = 0
    wave_position: int = 0
    wave_status: str = "idle"  # idle | filling | running | awaiting_results | complete
    markdown: str = ""
    status: str = "pending"  # pending | running | completed | failed
    assigned_worker_id: str | None = None
    task_id: str | None = None
    launch_failed: bool = False
    launch_error: str = ""
    execution_mode: str = "sequential_slices"
    current_slice_index: int = 0
    slice_count: int = 0
    slices: list[PlanExecutionSlice] = field(default_factory=list)
    slice_reports: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "failed"}

    @property
    def is_active(self) -> bool:
        return self.status == "running"

    @property
    def current_slice(self) -> PlanExecutionSlice | None:
        if not self.slices:
            return None
        if self.current_slice_index < 0 or self.current_slice_index >= len(self.slices):
            return None
        return self.slices[self.current_slice_index]


@dataclass
class PlanReport:
    """Structured worker report for one plan."""

    task_id: str = ""
    plan_id: str = ""
    plan_version: int = 0
    wave_id: int = 0
    wave_position: int = 0
    wave_target_size: int = 0
    worker_id: str = ""
    status: str = "success"  # success | error | partial
    what_was_requested: str = ""
    what_was_done: str = ""
    results_table: list[dict[str, Any]] = field(default_factory=list)
    key_metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    verdict: str = "PENDING"
    confidence: float = 0.0
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

    @classmethod
    def from_task_result(
        cls,
        result: Any,
        *,
        plan_id: str,
        plan_version: int,
        wave_id: int = 0,
        wave_position: int = 0,
        wave_target_size: int = 0,
    ) -> "PlanReport":
        report = getattr(result, "plan_report", None)
        if report is not None:
            payload = {
                key: getattr(report, key)
                for key in cls.__dataclass_fields__.keys()
                if hasattr(report, key)
            }
            payload.update(
                {
                    "task_id": getattr(result, "task_id", payload.get("task_id", "")),
                    "plan_id": plan_id,
                    "plan_version": plan_version,
                    "wave_id": wave_id or payload.get("wave_id", 0),
                    "wave_position": wave_position or payload.get("wave_position", 0),
                    "wave_target_size": wave_target_size or payload.get("wave_target_size", 0),
                    "worker_id": getattr(result, "worker_id", payload.get("worker_id", "")),
                    "raw_output": getattr(result, "raw_output", payload.get("raw_output", "")),
                    "error": getattr(result, "error", payload.get("error", "")),
                },
            )
            return cls(**payload)
        return cls(
            task_id=getattr(result, "task_id", ""),
            plan_id=plan_id,
            plan_version=plan_version,
            wave_id=wave_id,
            wave_position=wave_position,
            wave_target_size=wave_target_size,
            worker_id=getattr(result, "worker_id", ""),
            status=getattr(result, "status", "error"),
            what_was_done=getattr(result, "summary", ""),
            artifacts=getattr(result, "artifacts", []) or [],
            confidence=float(getattr(result, "confidence", 0.0) or 0.0),
            error=getattr(result, "error", ""),
            raw_output=getattr(result, "raw_output", ""),
        )
