"""Tests for state persistence."""

import json
import tempfile
from pathlib import Path

from app.models import (
    MemoryEntry,
    OrchestratorState,
    PlannerDecision,
    StopReason,
    Task,
    TaskResult,
    TaskStatus,
    ProcessInfo,
)
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


def test_save_and_load_with_tasks():
    store, path = _tmp_store()
    state = OrchestratorState(goal="test")
    task = Task(description="Do something", status=TaskStatus.RUNNING)
    task.mark_waiting_result()
    state.add_task(task)

    result = TaskResult(
        task_id=task.task_id,
        worker_id="w1",
        status="success",
        summary="Done",
    )
    state.results.append(result)
    state.last_planner_decision = PlannerDecision.LAUNCH_WORKER
    store.save(state)

    loaded = store.load()
    assert loaded is not None
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].status == TaskStatus.WAITING_RESULT
    assert loaded.tasks[0].description == "Do something"
    assert len(loaded.results) == 1
    assert loaded.results[0].summary == "Done"
    assert loaded.last_planner_decision == PlannerDecision.LAUNCH_WORKER


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


def test_state_memory_persistence():
    store, path = _tmp_store()
    state = OrchestratorState(goal="test")
    state.add_memory(MemoryEntry(content="Important note", source="planner"))
    state.add_memory(MemoryEntry(content="Worker result", source="worker:w1"))
    store.save(state)

    loaded = store.load()
    assert loaded is not None
    assert len(loaded.memory) == 2
    assert loaded.memory[0].content == "Important note"
    assert loaded.memory[1].source == "worker:w1"


def test_state_stop_reason_persistence():
    store, path = _tmp_store()
    state = OrchestratorState(goal="test")
    state.stop_reason = StopReason.GOAL_REACHED
    store.save(state)

    loaded = store.load()
    assert loaded is not None
    assert loaded.stop_reason == StopReason.GOAL_REACHED


def test_state_process_info():
    store, path = _tmp_store()
    state = OrchestratorState(goal="test")
    state.processes.append(ProcessInfo(task_id="t1", worker_id="w1", pid=12345))
    store.save(state)

    loaded = store.load()
    assert len(loaded.processes) == 1
    assert loaded.processes[0].pid == 12345
