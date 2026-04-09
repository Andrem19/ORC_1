"""
Artifact helpers for brokered execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.execution_models import ExecutionPlan, ToolResultEnvelope, WorkerAction
from app.run_context import ensure_current_run, read_current_run_id


class ExecutionArtifactStore:
    def __init__(self, root_dir: str | Path, *, run_id: str = "") -> None:
        self.root_dir = Path(root_dir)
        self.run_id = run_id or read_current_run_id(self.root_dir)

    @property
    def active_root(self) -> Path:
        if not self.run_id:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            return self.root_dir
        return ensure_current_run(self.root_dir, self.run_id)

    def save_plan(self, plan: ExecutionPlan) -> Path:
        path = self.active_root / "plans" / f"{plan.plan_id}.json"
        self._write_json(path, plan)
        return path

    def save_turn_action(self, *, plan_id: str, slice_id: str, turn_id: str, payload: WorkerAction | dict[str, Any]) -> Path:
        path = self.active_root / "turns" / plan_id / slice_id / f"{turn_id}_action.json"
        self._write_json(path, payload)
        return path

    def save_tool_result(self, envelope: ToolResultEnvelope, raw_payload: dict[str, Any]) -> Path:
        raw_path = self.active_root / "broker" / "raw" / f"{envelope.call_id}.json"
        self._write_json(raw_path, raw_payload)
        return raw_path

    def save_report(self, *, plan_id: str, slice_id: str, turn_id: str, payload: dict[str, Any]) -> Path:
        path = self.active_root / "reports" / plan_id / slice_id / f"{turn_id}.json"
        self._write_json(path, payload)
        return path

    def save_worker_parse_failure(
        self,
        *,
        plan_id: str,
        slice_id: str,
        payload: dict[str, Any],
    ) -> Path:
        path = self.active_root / "worker_failures" / plan_id / slice_id / f"{payload.get('failure_id', 'parse_failure')}.json"
        self._write_json(path, payload)
        return path

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(payload, "__dataclass_fields__"):
            from dataclasses import asdict

            data = asdict(payload)
        else:
            data = payload
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
