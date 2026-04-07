"""Tests for MCP session auto-reconnect on 'tool not found in registry'."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.models import OrchestratorState, Task, TaskResult, TaskStatus
from app.plan_models import PlanTask, TaskReport
from app.services.plan_orchestrator._result_processing import ResultProcessingMixin


def _make_report(error: str = "", raw_output: str = "") -> TaskReport:
    return TaskReport(
        task_id="t1",
        worker_id="w1",
        status="partial",
        verdict="PENDING",
        plan_version=1,
        error=error,
        raw_output=raw_output,
    )


def _make_mixin() -> ResultProcessingMixin:
    """Create a ResultProcessingMixin with minimal mock state."""
    mixin = ResultProcessingMixin()
    mixin.orch = MagicMock()
    mixin.state = OrchestratorState()
    mixin._plan_store = MagicMock()
    mixin._current_plan = MagicMock()
    mixin._current_plan.version = 1
    mixin._current_plan.get_task_by_stage.return_value = PlanTask(
        stage_number=1, stage_name="test", depends_on=[],
    )
    mixin._mcp_healthy = True
    mixin._mcp_reconnect_attempts = 0
    mixin._stage_retry_counts = {}
    mixin._persist_current_plan = MagicMock()

    # Create a plan-mode task in state
    task = Task(
        task_id="t1",
        assigned_worker_id="w1",
        max_attempts=3,
    )
    task.metadata["plan_mode"] = True
    task.metadata["stage_number"] = 1
    task.metadata["plan_version"] = 1
    task.status = TaskStatus.RUNNING
    mixin.state.tasks.append(task)

    return mixin


class TestIsSessionError:
    def test_tool_not_found_is_session_error(self):
        report = _make_report(error="Error: tool not found in registry")
        assert ResultProcessingMixin._is_session_error(report) is True

    def test_mcp_server_not_connected_is_not_session_error(self):
        report = _make_report(error="mcp server not connected")
        assert ResultProcessingMixin._is_session_error(report) is False

    def test_clean_report_is_not_session_error(self):
        report = _make_report()
        assert ResultProcessingMixin._is_session_error(report) is False

    def test_in_raw_output(self):
        report = _make_report(raw_output="failed: tool not found in registry at step 3")
        assert ResultProcessingMixin._is_session_error(report) is True


class TestMcpReconnect:
    def test_session_error_retries_not_marked_unhealthy(self):
        mixin = _make_mixin()
        result = TaskResult(
            task_id="t1",
            worker_id="w1",
            status="partial",
            plan_report=_make_report(error="tool not found in registry"),
        )

        mixin._process_plan_results([result])
        # MCP should still be healthy after first session error
        assert mixin._mcp_healthy is True
        assert mixin._mcp_reconnect_attempts == 1
        # Task should be reset to PENDING for retry
        task = mixin.state.tasks[0]
        assert task.status == TaskStatus.PENDING

    def test_session_error_max_retries_marks_unhealthy(self):
        mixin = _make_mixin()
        mixin._mcp_reconnect_attempts = 2  # already tried twice
        result = TaskResult(
            task_id="t1",
            worker_id="w1",
            status="partial",
            plan_report=_make_report(error="tool not found in registry"),
        )

        mixin._process_plan_results([result])
        assert mixin._mcp_healthy is False
        assert mixin.state.mcp_consecutive_failures == 1

    def test_non_session_mcp_failure_immediate_unhealthy(self):
        mixin = _make_mixin()
        result = TaskResult(
            task_id="t1",
            worker_id="w1",
            status="partial",
            plan_report=_make_report(error="mcp server not connected"),
        )

        mixin._process_plan_results([result])
        assert mixin._mcp_healthy is False
        assert mixin.state.mcp_consecutive_failures == 1

    def test_reconnect_counter_resets_per_cycle(self):
        mixin = _make_mixin()
        mixin._mcp_reconnect_attempts = 2
        # Simulate cycle reset (what _core.py does)
        mixin._mcp_reconnect_attempts = 0
        assert mixin._mcp_reconnect_attempts == 0

        result = TaskResult(
            task_id="t1",
            worker_id="w1",
            status="partial",
            plan_report=_make_report(error="tool not found in registry"),
        )
        mixin._process_plan_results([result])
        # Should retry again since counter was reset
        assert mixin._mcp_healthy is True
        assert mixin._mcp_reconnect_attempts == 1
