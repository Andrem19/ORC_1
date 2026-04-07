"""Tests for MCP health inference from active worker results."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.models import OrchestratorState, Task, TaskResult, TaskStatus
from app.services.plan_orchestrator._task_health import TaskHealthMixin


def _make_mixin() -> TaskHealthMixin:
    """Create a TaskHealthMixin with mock state."""
    mixin = TaskHealthMixin()
    mixin.state = OrchestratorState()
    mixin._mcp_healthy = True
    mixin._mcp_check_cycle = 0
    mixin._cycles_since_last_real_health_check = 0
    mixin._mcp_reconnect_attempts = 0
    mixin.worker_service = MagicMock()
    mixin.config = MagicMock()
    mixin.config.worker_adapter.cli_path = "test-cli"
    mixin.config.mcp_health_check_interval_cycles = 5

    # Add an active task so the "active workers" path is taken
    task = Task(task_id="active-1", assigned_worker_id="w1")
    task.status = TaskStatus.RUNNING
    mixin.state.tasks.append(task)

    return mixin


def _result(status: str, summary: str = "", error: str = "") -> TaskResult:
    return TaskResult(
        task_id="t1",
        worker_id="w1",
        status=status,
        summary=summary,
        error=error,
    )


class TestInferMcpHealth:
    def test_healthy_from_recent_mcp_successes(self):
        mixin = _make_mixin()
        mixin.state.results = [
            _result("success", "Created backtest run with snapshot id"),
            _result("success", "Trained catboost model"),
        ]
        assert mixin._infer_mcp_health_from_active_workers() == "likely_healthy"

    def test_unhealthy_from_recent_mcp_failures(self):
        mixin = _make_mixin()
        mixin.state.results = [
            _result("partial", "backtest attempt", error="tool not found in registry"),
            _result("partial", "snapshot feature", error="mcp server not connected"),
            _result("partial", "model training", error="mcp tools are not available"),
        ]
        assert mixin._infer_mcp_health_from_active_workers() == "likely_unhealthy"

    def test_uncertain_with_no_mcp_evidence(self):
        mixin = _make_mixin()
        mixin.state.results = [
            _result("success", "Generic task completed"),
            _result("error", "Something went wrong"),
        ]
        assert mixin._infer_mcp_health_from_active_workers() == "uncertain"

    def test_uncertain_with_empty_results(self):
        mixin = _make_mixin()
        mixin.state.results = []
        assert mixin._infer_mcp_health_from_active_workers() == "uncertain"

    def test_healthy_inference_does_not_set_flag_true(self):
        """likely_healthy should NOT override an explicitly-set unhealthy flag."""
        mixin = _make_mixin()
        mixin.state.current_cycle = 5  # trigger periodic check
        mixin._mcp_healthy = False
        mixin.state.mcp_consecutive_failures = 2
        mixin.state.results = [
            _result("success", "backtest completed"),
        ]
        mixin._periodic_mcp_health_check()
        # Flag should remain False — only real probe sets it back to True
        assert mixin._mcp_healthy is False

    def test_unhealthy_inference_marks_mcp_unhealthy(self):
        mixin = _make_mixin()
        mixin.state.current_cycle = 5  # trigger periodic check
        mixin._mcp_healthy = True
        mixin.state.results = [
            _result("partial", "backtest", error="tool not found in registry"),
            _result("partial", "snapshot", error="mcp server not connected"),
        ]
        mixin._periodic_mcp_health_check()
        assert mixin._mcp_healthy is False
        assert mixin.state.mcp_consecutive_failures == 1

    def test_single_mcp_failure_is_uncertain(self):
        """Only 1 MCP failure — not enough to infer unhealthy."""
        mixin = _make_mixin()
        mixin.state.results = [
            _result("partial", "backtest", error="tool not found in registry"),
        ]
        assert mixin._infer_mcp_health_from_active_workers() == "uncertain"

    def test_success_overrides_failures(self):
        """Even with some failures, a success means likely_healthy."""
        mixin = _make_mixin()
        mixin.state.results = [
            _result("partial", "backtest", error="tool not found"),
            _result("partial", "snapshot", error="mcp error"),
            _result("success", "backtest completed successfully"),
        ]
        assert mixin._infer_mcp_health_from_active_workers() == "likely_healthy"
