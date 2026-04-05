"""Integration test: restart recovery."""

import json
from pathlib import Path
import tempfile

from app.adapters.fake_planner import FakePlanner
from app.adapters.fake_worker import FakeWorker
from app.config import OrchestratorConfig
from app.models import (
    OrchestratorState,
    StopReason,
    Task,
    TaskStatus,
)
from app.orchestrator import Orchestrator
from app.state_store import StateStore


def test_state_persists_across_restart():
    """State saved to disk survives a restart."""
    tmp = Path(tempfile.mkdtemp())
    store = StateStore(tmp / "state.json")

    # First run — create some state
    state = OrchestratorState(goal="Persist test", current_cycle=7, empty_cycles=2)
    task = Task(description="My task", status=TaskStatus.WAITING_RESULT)
    state.add_task(task)
    store.save(state)

    # Second orchestrator loads state
    planner = FakePlanner(responses=[
        {"decision": "launch_worker", "target_worker_id": "qwen-1", "task_instruction": "Continue", "reason": "restart", "check_after_seconds": 1},
        {"decision": "finish", "should_finish": True, "final_summary": "Recovered", "reason": "done"},
    ])
    worker = FakeWorker(responses=[
        {"status": "success", "summary": "Continued work", "confidence": 0.9},
    ])

    config = OrchestratorConfig(goal="Persist test", poll_interval_seconds=0, max_empty_cycles=10)
    orch = Orchestrator(config=config, state_store=store, planner_adapter=planner, worker_adapter=worker)

    loaded = orch.load_state()
    assert loaded is True
    assert orch.state.current_cycle == 7
    assert orch.state.goal == "Persist test"


def test_running_tasks_stalled_after_restart():
    """Tasks that were running are marked stalled after restart recovery."""
    tmp = Path(tempfile.mkdtemp())
    store = StateStore(tmp / "state.json")

    state = OrchestratorState(goal="test", current_cycle=3)
    task1 = Task(description="Running task", status=TaskStatus.RUNNING)
    task2 = Task(description="Waiting task", status=TaskStatus.WAITING_RESULT)
    task3 = Task(description="Completed task", status=TaskStatus.COMPLETED)
    state.add_task(task1)
    state.add_task(task2)
    state.add_task(task3)
    store.save(state)

    planner = FakePlanner(responses=[
        {"decision": "finish", "should_finish": True, "final_summary": "Done", "reason": "done"},
    ])
    worker = FakeWorker(responses=[])

    config = OrchestratorConfig(goal="test", poll_interval_seconds=0, max_empty_cycles=10)
    orch = Orchestrator(config=config, state_store=store, planner_adapter=planner, worker_adapter=worker)
    orch.load_state()

    # Running and waiting tasks should be stalled
    stalled = [t for t in orch.state.tasks if t.status == TaskStatus.STALLED]
    assert len(stalled) == 2

    # Completed task should remain completed
    completed = [t for t in orch.state.tasks if t.status == TaskStatus.COMPLETED]
    assert len(completed) == 1


def test_memory_preserved_across_restart():
    """Memory entries survive restart."""
    tmp = Path(tempfile.mkdtemp())
    store = StateStore(tmp / "state.json")

    from app.models import MemoryEntry

    state = OrchestratorState(goal="test")
    state.add_memory(MemoryEntry(content="Key insight", source="planner"))
    state.add_memory(MemoryEntry(content="Worker result X", source="worker:w1"))
    store.save(state)

    planner = FakePlanner(responses=[
        {"decision": "finish", "should_finish": True, "final_summary": "Done", "reason": "done"},
    ])
    worker = FakeWorker(responses=[])

    config = OrchestratorConfig(goal="test", poll_interval_seconds=0, max_empty_cycles=10)
    orch = Orchestrator(config=config, state_store=store, planner_adapter=planner, worker_adapter=worker)
    orch.load_state()

    assert len(orch.state.memory) == 2
    assert orch.state.memory[0].content == "Key insight"
