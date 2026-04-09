"""Minimal legacy state utility used only for reset/archive compatibility."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.models import OrchestratorState
from app.run_context import build_state_pointer, ensure_current_run, extract_pointed_path, read_current_run_id, resolve_run_dir

logger = logging.getLogger("orchestrator.state")
_RESULT_RAW_OUTPUT_RETAIN = 24_000
def _tail_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _serialize_state(state: OrchestratorState) -> dict[str, Any]:
    data = asdict(state)
    for result in data.get("results", []):
        raw_output = str(result.get("raw_output", "") or "")
        result["raw_output"] = _tail_text(raw_output, _RESULT_RAW_OUTPUT_RETAIN)
        plan_report = result.get("plan_report")
        if isinstance(plan_report, dict):
            report_raw = str(plan_report.get("raw_output", "") or "")
            plan_report["raw_output"] = _tail_text(report_raw, _RESULT_RAW_OUTPUT_RETAIN)
    if state.stop_reason is not None:
        data["stop_reason"] = state.stop_reason.value
    return data


def _deserialize_state(data: dict[str, Any]) -> OrchestratorState:
    """Reconstruct the compact broker-only presentation state from a dict."""
    from app.models import StopReason, TaskResult
    from app.plan_models import PlanReport

    results = []
    for r in data.pop("results", []):
        plan_report = r.get("plan_report")
        if isinstance(plan_report, dict):
            r = dict(r)
            r["plan_report"] = PlanReport(**plan_report)
        results.append(TaskResult(**r))
    sr = data.pop("stop_reason", None)
    if sr is not None:
        data["stop_reason"] = StopReason(sr)

    valid_keys = set(OrchestratorState.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    state = OrchestratorState(**filtered)
    state.results = results
    return state


class StateStore:
    """File-based state store with atomic writes."""

    def __init__(self, state_path: str | Path, *, run_id: str = "") -> None:
        self.state_path = Path(state_path)
        self.run_id = run_id or read_current_run_id(self.state_path.parent)

    @property
    def current_state_path(self) -> Path:
        if not self.run_id:
            return self.state_path
        run_dir = ensure_current_run(self.state_path.parent, self.run_id)
        return run_dir / self.state_path.name

    def save(self, state: OrchestratorState) -> None:
        """Atomically save state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state._touch()
        data = _serialize_state(state)
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        current_state_path = self.current_state_path
        current_state_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(current_state_path.parent),
            suffix=".tmp",
        )
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            Path(tmp_path).rename(current_state_path)
            if current_state_path != self.state_path:
                pointer = build_state_pointer(run_id=self.run_id, state_path=current_state_path)
                self.state_path.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.debug("State saved to %s", current_state_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def load(self) -> OrchestratorState | None:
        """Load state from disk. Returns None if no state exists."""
        load_path = self._resolve_load_path()
        if load_path is None or not load_path.exists():
            logger.info("No state file at %s, starting fresh", self.state_path)
            return None
        try:
            raw = load_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            state = _deserialize_state(data)
            logger.info("State loaded from %s (cycle %d)", load_path, state.current_cycle)
            return state
        except Exception as e:
            logger.error("Failed to load state: %s", e)
            return None

    def clear(self) -> None:
        """Remove state file."""
        current_state_path = self._resolve_load_path()
        if current_state_path and current_state_path != self.state_path and current_state_path.exists():
            current_state_path.unlink()
        if self.state_path.exists():
            self.state_path.unlink()
            logger.info("State file removed: %s", self.state_path)

    def archive_to(self, target_dir: Path) -> bool:
        """Copy state file to target_dir for archival. Returns True if file existed."""
        target_dir.mkdir(parents=True, exist_ok=True)
        current_state_path = self._resolve_load_path()
        copied = False
        if current_state_path and current_state_path.exists():
            shutil.copy2(current_state_path, target_dir / current_state_path.name)
            copied = True
        if self.state_path.exists():
            shutil.copy2(self.state_path, target_dir / self.state_path.name)
            copied = True
        if copied:
            return True
        return False

    def _resolve_load_path(self) -> Path | None:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                return self.state_path
            pointed = extract_pointed_path(data, pointer_type="run_state") if isinstance(data, dict) else None
            if pointed is not None:
                return pointed
        run_dir = resolve_run_dir(self.state_path.parent, self.run_id)
        candidate = run_dir / self.state_path.name
        if candidate.exists():
            return candidate
        if self.state_path.exists():
            return self.state_path
        return None
