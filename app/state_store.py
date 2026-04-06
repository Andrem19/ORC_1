"""
State persistence for the orchestrator.

Uses JSON file storage. Chosen over SQLite for:
- Simplicity and zero dependencies
- Easy debugging (human-readable)
- Atomic writes via temp file + rename
- Sufficient performance for single-process orchestrator
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.models import OrchestratorState

logger = logging.getLogger("orchestrator.state")


def _serialize_state(state: OrchestratorState) -> dict[str, Any]:
    """Convert state to a JSON-serializable dict."""
    data = asdict(state)
    # Enum values -> strings
    if state.last_planner_decision is not None:
        data["last_planner_decision"] = state.last_planner_decision.value
    if state.stop_reason is not None:
        data["stop_reason"] = state.stop_reason.value
    return data


def _deserialize_state(data: dict[str, Any]) -> OrchestratorState:
    """Reconstruct state from a dict."""
    from app.models import PlannerDecision, StopReason, Task, TaskResult, ProcessInfo, MemoryEntry, TaskStatus
    from app.plan_models import TaskReport

    tasks = [Task(**t) for t in data.pop("tasks", [])]
    # Restore enum status for tasks
    for t in tasks:
        t.status = TaskStatus(t.status)

    results = []
    for r in data.pop("results", []):
        plan_report = r.get("plan_report")
        if isinstance(plan_report, dict):
            r = dict(r)
            r["plan_report"] = TaskReport(**plan_report)
        results.append(TaskResult(**r))
    processes = [ProcessInfo(**p) for p in data.pop("processes", [])]
    memory = [MemoryEntry(**m) for m in data.pop("memory", [])]

    lpd = data.pop("last_planner_decision", None)
    if lpd is not None:
        data["last_planner_decision"] = PlannerDecision(lpd)

    sr = data.pop("stop_reason", None)
    if sr is not None:
        data["stop_reason"] = StopReason(sr)

    state = OrchestratorState(**data)
    state.tasks = tasks
    state.results = results
    state.processes = processes
    state.memory = memory
    return state


class StateStore:
    """File-based state store with atomic writes."""

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)

    def save(self, state: OrchestratorState) -> None:
        """Atomically save state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = _serialize_state(state)
        payload = json.dumps(data, indent=2, ensure_ascii=False)

        # Atomic write: write to temp, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.state_path.parent),
            suffix=".tmp",
        )
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            Path(tmp_path).rename(self.state_path)
            logger.debug("State saved to %s", self.state_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def load(self) -> OrchestratorState | None:
        """Load state from disk. Returns None if no state exists."""
        if not self.state_path.exists():
            logger.info("No state file at %s, starting fresh", self.state_path)
            return None
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            state = _deserialize_state(data)
            logger.info("State loaded from %s (cycle %d)", self.state_path, state.current_cycle)
            return state
        except Exception as e:
            logger.error("Failed to load state: %s", e)
            return None

    def clear(self) -> None:
        """Remove state file."""
        if self.state_path.exists():
            self.state_path.unlink()
            logger.info("State file removed: %s", self.state_path)

    def archive_to(self, target_dir: Path) -> bool:
        """Copy state file to target_dir for archival. Returns True if file existed."""
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            shutil.copy2(self.state_path, target_dir / self.state_path.name)
            return True
        return False
