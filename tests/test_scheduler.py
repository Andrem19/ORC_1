"""Tests for scheduler logic."""

from app.models import (
    OrchestratorState,
    PlannerDecision,
    Task,
    TaskResult,
    TaskStatus,
)
from app.scheduler import Scheduler


def test_should_call_planner_first_cycle():
    s = Scheduler()
    state = OrchestratorState(goal="test")
    assert s.should_call_planner(state, [])


def test_should_call_planner_with_new_results():
    s = Scheduler()
    state = OrchestratorState(goal="test")
    state.current_cycle = 5
    results = [TaskResult(task_id="t1", worker_id="w1", status="success", summary="done")]
    assert s.should_call_planner(state, results)


def test_should_call_planner_pending_no_active():
    s = Scheduler()
    state = OrchestratorState(goal="test")
    state.current_cycle = 3
    task = Task(status=TaskStatus.PENDING)
    state.add_task(task)
    assert s.should_call_planner(state, [])


def test_should_not_call_planner_active_running():
    s = Scheduler()
    state = OrchestratorState(goal="test")
    state.current_cycle = 3
    task = Task(status=TaskStatus.WAITING_RESULT)
    state.add_task(task)
    assert not s.should_call_planner(state, [])


def test_should_stop_max_errors():
    s = Scheduler(max_errors_total=5)
    state = OrchestratorState(goal="test")
    state.total_errors = 5
    assert s.should_stop_orchestrator(state) == "max_errors"


def test_should_stop_max_empty_cycles():
    s = Scheduler(max_empty_cycles=3)
    state = OrchestratorState(goal="test")
    state.empty_cycles = 3
    assert s.should_stop_orchestrator(state) == "no_progress"


def test_should_stop_planner_finish():
    s = Scheduler()
    state = OrchestratorState(goal="test")
    state.last_planner_decision = PlannerDecision.FINISH
    assert s.should_stop_orchestrator(state) == "goal_reached"


def test_should_not_stop_normal():
    s = Scheduler()
    state = OrchestratorState(goal="test")
    assert s.should_stop_orchestrator(state) is None


def test_sleep_interval_default():
    s = Scheduler(poll_interval_seconds=120)
    state = OrchestratorState(goal="test")
    assert s.sleep_interval(state) == 120


def test_should_call_planner_all_tasks_settled():
    s = Scheduler()
    state = OrchestratorState(goal="test")
    state.current_cycle = 5
    task = Task(status=TaskStatus.COMPLETED)
    state.add_task(task)
    # All tasks done, no pending, no active — planner must decide
    assert s.should_call_planner(state, [])
