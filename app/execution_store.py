"""
Persistence for brokered execution state with hard cutover from the legacy runtime.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import MISSING, asdict, fields
from pathlib import Path
from typing import Any

from app.execution_models import (
    BaselineRef,
    BrokerHealth,
    ExecutionPlan,
    ExecutionStateV2,
    ExecutionTurn,
    PlanSlice,
    ToolPolicy,
    ToolResultEnvelope,
    WorkerAction,
    WorkerReportableIssue,
)
from app.run_context import build_state_pointer, ensure_current_run, extract_pointed_path, read_current_run_id, resolve_run_dir

_COMPLETED_TURN_HISTORY_LIMIT = 50
_COMPLETED_LEDGER_LIMIT = 100


class ExecutionStateStore:
    def __init__(self, state_path: str | Path, *, run_id: str = "") -> None:
        self.state_path = Path(state_path)
        self.run_id = run_id or read_current_run_id(self.state_path.parent)

    @property
    def current_state_path(self) -> Path:
        if not self.run_id:
            return self.state_path
        run_dir = ensure_current_run(self.state_path.parent, self.run_id)
        return run_dir / self.state_path.name

    def save(self, state: ExecutionStateV2) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state.touch()
        payload_dict = asdict(state)
        payload_dict["turn_history"] = [asdict(item) for item in _trim_turn_history(state)]
        payload_dict["tool_call_ledger"] = [asdict(item) for item in _trim_tool_call_ledger(state)]
        payload = json.dumps(payload_dict, ensure_ascii=False, indent=2)
        current_state_path = self.current_state_path
        current_state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(current_state_path.parent), suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            Path(tmp_path).rename(current_state_path)
            if current_state_path != self.state_path:
                pointer = build_state_pointer(run_id=self.run_id, state_path=current_state_path)
                self.state_path.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def load(self) -> ExecutionStateV2 | None:
        load_path = self._resolve_load_path()
        if load_path is None or not load_path.exists():
            return None
        payload = json.loads(load_path.read_text(encoding="utf-8"))
        return _deserialize_state(payload)

    def clear(self) -> None:
        current_state_path = self._resolve_load_path()
        if current_state_path and current_state_path.exists():
            current_state_path.unlink()
        if self.state_path.exists():
            self.state_path.unlink()

    def archive_legacy_runtime(self, *, legacy_state_path: str | Path, legacy_plan_dir: str | Path) -> Path | None:
        current_state = self._resolve_load_path()
        if current_state is not None and current_state.exists():
            return None
        legacy_state = Path(legacy_state_path)
        legacy_plan_root = Path(legacy_plan_dir)
        if not legacy_state.exists() and not legacy_plan_root.exists():
            return None
        archive_root = self.state_path.parent / "archive" / "broker_cutover"
        archive_root.mkdir(parents=True, exist_ok=True)
        target = archive_root / self._archive_stamp()
        target.mkdir(parents=True, exist_ok=True)
        if legacy_state.exists():
            shutil.copy2(legacy_state, target / legacy_state.name)
            legacy_state.unlink(missing_ok=True)
        if legacy_plan_root.exists():
            shutil.copytree(legacy_plan_root, target / "plans", dirs_exist_ok=True)
            self._clear_dir(legacy_plan_root)
        return target

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

    @staticmethod
    def _archive_stamp() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    @staticmethod
    def _clear_dir(root: Path) -> None:
        for path in root.iterdir():
            if path.name in {"runs", "current", "current_run.json"}:
                path.unlink(missing_ok=True) if not path.is_dir() else shutil.rmtree(path, ignore_errors=True)
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


def _deserialize_state(payload: dict[str, Any]) -> ExecutionStateV2:
    plans = [_deserialize_plan(item) for item in payload.get("plans", []) if isinstance(item, dict)]
    turns = [_deserialize_turn(item) for item in payload.get("turn_history", []) if isinstance(item, dict)]
    ledger = [_build_dataclass(ToolResultEnvelope, item) for item in payload.get("tool_call_ledger", []) if isinstance(item, dict)]
    broker_health = _build_dataclass(BrokerHealth, payload.get("broker_health", {}))
    state = ExecutionStateV2(
        goal=str(payload.get("goal", "") or ""),
        status=str(payload.get("status", "idle") or "idle"),
        plans=plans,
        turn_history=turns,
        tool_call_ledger=ledger,
        broker_health=broker_health,
        stop_reason=str(payload.get("stop_reason", "") or ""),
        current_plan_id=str(payload.get("current_plan_id", "") or ""),
        completed_plan_count=int(payload.get("completed_plan_count", 0) or 0),
        consecutive_failed_plans=int(payload.get("consecutive_failed_plans", 0) or 0),
        no_progress_cycles=int(payload.get("no_progress_cycles", 0) or 0),
        broker_failure_count=int(payload.get("broker_failure_count", 0) or 0),
        created_at=str(payload.get("created_at", "") or ""),
        updated_at=str(payload.get("updated_at", "") or ""),
    )
    return state


def _deserialize_plan(payload: dict[str, Any]) -> ExecutionPlan:
    baseline = _build_dataclass(BaselineRef, payload.get("baseline_ref", {}), required={"snapshot_id": "", "version": 1})
    slices = [_build_dataclass(PlanSlice, item) for item in payload.get("slices", []) if isinstance(item, dict)]
    return ExecutionPlan(
        plan_id=str(payload.get("plan_id", "") or ""),
        goal=str(payload.get("goal", "") or ""),
        baseline_ref=baseline,
        global_constraints=[str(item) for item in payload.get("global_constraints", []) or []],
        slices=slices,
        plan_source_kind=str(payload.get("plan_source_kind", "planner") or "planner"),
        source_sequence_id=str(payload.get("source_sequence_id", "") or ""),
        source_raw_plan=str(payload.get("source_raw_plan", "") or ""),
        source_manifest_path=str(payload.get("source_manifest_path", "") or ""),
        source_semantic_path=str(payload.get("source_semantic_path", "") or ""),
        source_compile_report_path=str(payload.get("source_compile_report_path", "") or ""),
        sequence_batch_index=int(payload.get("sequence_batch_index", 0) or 0),
        status=str(payload.get("status", "draft") or "draft"),
        created_at=str(payload.get("created_at", "") or ""),
        updated_at=str(payload.get("updated_at", "") or ""),
    )


def _deserialize_turn(payload: dict[str, Any]) -> ExecutionTurn:
    action_payload = payload.get("action", {}) or {}
    issues = [_build_dataclass(WorkerReportableIssue, item) for item in action_payload.get("reportable_issues", []) or [] if isinstance(item, dict)]
    action = WorkerAction(
        action_id=str(action_payload.get("action_id", "") or ""),
        action_type=str(action_payload.get("action_type", "") or ""),
        tool=str(action_payload.get("tool", "") or ""),
        arguments=dict(action_payload.get("arguments", {}) or {}),
        reason=str(action_payload.get("reason", "") or ""),
        expected_evidence=[str(item) for item in action_payload.get("expected_evidence", []) or []],
        status=str(action_payload.get("status", "") or ""),
        summary=str(action_payload.get("summary", "") or ""),
        facts=dict(action_payload.get("facts", {}) or {}),
        artifacts=[str(item) for item in action_payload.get("artifacts", []) or []],
        pending_questions=[str(item) for item in action_payload.get("pending_questions", []) or []],
        reportable_issues=issues,
        verdict=str(action_payload.get("verdict", "") or ""),
        key_metrics=dict(action_payload.get("key_metrics", {}) or {}),
        findings=[str(item) for item in action_payload.get("findings", []) or []],
        rejected_findings=[str(item) for item in action_payload.get("rejected_findings", []) or []],
        next_actions=[str(item) for item in action_payload.get("next_actions", []) or []],
        risks=[str(item) for item in action_payload.get("risks", []) or []],
        evidence_refs=[str(item) for item in action_payload.get("evidence_refs", []) or []],
        confidence=float(action_payload.get("confidence", 0.0) or 0.0),
        reason_code=str(action_payload.get("reason_code", "") or ""),
        retryable=bool(action_payload.get("retryable", False)),
    )
    tool_result = None
    if isinstance(payload.get("tool_result"), dict):
        tool_result = _build_dataclass(ToolResultEnvelope, payload["tool_result"])
    return ExecutionTurn(
        turn_id=str(payload.get("turn_id", "") or ""),
        plan_id=str(payload.get("plan_id", "") or ""),
        slice_id=str(payload.get("slice_id", "") or ""),
        worker_id=str(payload.get("worker_id", "") or ""),
        turn_index=int(payload.get("turn_index", 0) or 0),
        action=action,
        tool_result=tool_result,
        created_at=str(payload.get("created_at", "") or ""),
    )


def _build_dataclass(cls: Any, payload: Any, *, required: dict[str, Any] | None = None) -> Any:
    data = dict(payload) if isinstance(payload, dict) else {}
    for key, value in (required or {}).items():
        data.setdefault(key, value)
    allowed = {item.name for item in fields(cls)}
    filtered = {key: data[key] for key in allowed if key in data}
    for item in fields(cls):
        if item.name in filtered:
            continue
        if item.default is not MISSING:
            filtered[item.name] = item.default
        elif item.default_factory is not MISSING:  # type: ignore[attr-defined]
            filtered[item.name] = item.default_factory()  # type: ignore[misc]
    return cls(**filtered)


def _trim_turn_history(state: ExecutionStateV2) -> list[ExecutionTurn]:
    active_plan_id = state.current_plan_id
    keep_turn_ids = {turn.turn_id for turn in state.turn_history if turn.plan_id == active_plan_id}
    completed_turn_ids = [
        turn.turn_id
        for turn in state.turn_history
        if turn.plan_id != active_plan_id
    ][-_COMPLETED_TURN_HISTORY_LIMIT:]
    keep_turn_ids.update(completed_turn_ids)
    return [turn for turn in state.turn_history if turn.turn_id in keep_turn_ids]


def _trim_tool_call_ledger(state: ExecutionStateV2) -> list[ToolResultEnvelope]:
    active_plan_id = state.current_plan_id
    keep_call_ids = {item.call_id for item in state.tool_call_ledger if item.plan_id == active_plan_id}
    completed_call_ids = [
        item.call_id
        for item in state.tool_call_ledger
        if item.plan_id != active_plan_id
    ][-_COMPLETED_LEDGER_LIMIT:]
    keep_call_ids.update(completed_call_ids)
    return [item for item in state.tool_call_ledger if item.call_id in keep_call_ids]
