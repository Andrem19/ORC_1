"""Tests for minimal legacy state utility used by reset/archive paths."""

import json
import tempfile
from pathlib import Path

from app.plan_models import PlanReport
from app.models import DirectOrchestratorState, OrchestratorState, StopReason, TaskResult
from app.state_store import StateStore


def _tmp_store() -> tuple[StateStore, Path]:
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "test_state.json"
    return StateStore(path), path


def test_save_and_load_empty():
    store, path = _tmp_store()
    state = OrchestratorState(goal="test goal")
    store.save(state)
    assert path.exists()

    loaded = store.load()
    assert loaded is not None
    assert loaded.goal == "test goal"
    assert loaded.current_cycle == 0


def test_orchestrator_state_alias_points_to_direct_state():
    state = OrchestratorState(goal="test goal")

    assert isinstance(state, DirectOrchestratorState)

def test_save_and_load_with_results_only():
    store, path = _tmp_store()
    state = OrchestratorState(goal="test")
    result = TaskResult(
        task_id="t1",
        worker_id="w1",
        status="success",
        summary="Done",
        plan_report=PlanReport(
            task_id="t1",
            plan_id="p1",
            plan_version=1,
            worker_id="w1",
            status="success",
            what_was_done="Done",
        ),
    )
    state.results.append(result)
    store.save(state)

    loaded = store.load()
    assert loaded is not None
    assert len(loaded.results) == 1
    assert loaded.results[0].summary == "Done"
    assert loaded.results[0].plan_report is not None
    assert loaded.results[0].plan_report.plan_version == 1


def test_load_nonexistent():
    store = StateStore("/nonexistent/path/state.json")
    assert store.load() is None


def test_load_corrupt_file():
    store, path = _tmp_store()
    path.write_text("not valid json {{{")
    assert store.load() is None


def test_clear():
    store, path = _tmp_store()
    state = OrchestratorState(goal="test")
    store.save(state)
    assert path.exists()

    store.clear()
    assert not path.exists()


def test_state_stop_reason_persistence():
    store, path = _tmp_store()
    state = OrchestratorState(goal="test")
    state.stop_reason = StopReason.GOAL_REACHED
    store.save(state)

    loaded = store.load()
    assert loaded is not None
    assert loaded.stop_reason == StopReason.GOAL_REACHED

def test_save_refreshes_state_timestamps():
    store, path = _tmp_store()
    state = OrchestratorState(goal="test")
    original_updated_at = state.updated_at

    store.save(state)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["updated_at"] >= original_updated_at
    assert loaded["last_change_at"] is not None


def test_save_trims_large_result_raw_output():
    store, path = _tmp_store()
    state = OrchestratorState(goal="test")
    raw = "x" * 50000
    state.results.append(
        TaskResult(
            task_id="t1",
            worker_id="w1",
            status="error",
            raw_output=raw,
            plan_report=PlanReport(
                task_id="t1",
                plan_id="p1",
                plan_version=1,
                worker_id="w1",
                status="error",
                raw_output=raw,
            ),
        )
    )

    store.save(state)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    saved = loaded["results"][0]
    assert len(saved["raw_output"]) <= 24000
    assert len(saved["plan_report"]["raw_output"]) <= 24000

def test_state_store_uses_run_scoped_pointer():
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "state.json"
    store = StateStore(path, run_id="run-123")
    state = OrchestratorState(goal="test")

    store.save(state)

    pointer = json.loads(path.read_text(encoding="utf-8"))
    assert pointer["pointer_type"] == "run_state"
    run_state_path = Path(pointer["state_path"])
    assert run_state_path.exists()

    loaded = store.load()
    assert loaded is not None
    assert loaded.goal == "test"
