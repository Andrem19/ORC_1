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
    mixin._mcp_reconnect_stage_counts = {}
    mixin._stage_retry_counts = {}
    mixin._persist_current_plan = MagicMock()
    mixin.worker_service = MagicMock()
    mixin.worker_service.check_mcp_health.return_value = True
    mixin.config = MagicMock()
    mixin.config.worker_adapter.cli_path = "test-cli"

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
    @patch("app.services.plan_orchestrator._result_processing._time.sleep")
    def test_session_error_retries_not_marked_unhealthy(self, mock_sleep):
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
        assert mixin._mcp_reconnect_stage_counts[1] == 1
        # Task should be reset to PENDING for retry
        task = mixin.state.tasks[0]
        assert task.status == TaskStatus.PENDING
        mock_sleep.assert_called_once_with(5)

    def test_session_error_max_retries_marks_unhealthy(self):
        mixin = _make_mixin()
        mixin._mcp_reconnect_stage_counts = {1: 2}  # already tried twice
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

    @patch("app.services.plan_orchestrator._result_processing._time.sleep")
    def test_reconnect_counter_persists_across_cycles(self, mock_sleep):
        """Counter should NOT reset between cycles — budget must deplete."""
        mixin = _make_mixin()
        mixin._mcp_reconnect_stage_counts = {1: 1}
        result = TaskResult(
            task_id="t1",
            worker_id="w1",
            status="partial",
            plan_report=_make_report(error="tool not found in registry"),
        )
        mixin._process_plan_results([result])
        # Second attempt should succeed (budget 2)
        assert mixin._mcp_healthy is True
        assert mixin._mcp_reconnect_stage_counts[1] == 2

        # Third attempt should exhaust budget and mark unhealthy
        mixin2 = _make_mixin()
        mixin2._mcp_reconnect_stage_counts = {1: 2}
        mixin2._process_plan_results([result])
        assert mixin2._mcp_healthy is False

    @patch("app.services.plan_orchestrator._result_processing._time.sleep")
    def test_different_stage_has_independent_budget(self, mock_sleep):
        """Each stage should have its own reconnect budget."""
        mixin = _make_mixin()
        mixin._mcp_reconnect_stage_counts = {1: 2}  # stage 1 exhausted

        result = TaskResult(
            task_id="t1",
            worker_id="w1",
            status="partial",
            plan_report=_make_report(error="tool not found in registry"),
        )
        mixin._process_plan_results([result])
        assert mixin._mcp_healthy is False  # stage 1 exhausted budget

        # A different stage should still have fresh budget
        mixin2 = _make_mixin()
        mixin2._mcp_reconnect_stage_counts = {2: 2}  # stage 2 exhausted
        mixin2.state.tasks[0].metadata["stage_number"] = 3
        mixin2._current_plan.get_task_by_stage.return_value = PlanTask(
            stage_number=3, stage_name="test3", depends_on=[],
        )
        result3 = TaskResult(
            task_id="t1",
            worker_id="w1",
            status="partial",
            plan_report=_make_report(error="tool not found in registry"),
        )
        mixin2._process_plan_results([result3])
        assert mixin2._mcp_healthy is True  # stage 3 has fresh budget


class TestMcpReconnectHealthProbe:
    """Tests for MCP health probe before reconnect retry."""

    def test_probe_success_allows_retry(self):
        """If health probe passes, task should be reset to PENDING."""
        mixin = _make_mixin()
        mixin.worker_service.check_mcp_health.return_value = True
        result = TaskResult(
            task_id="t1",
            worker_id="w1",
            status="partial",
            plan_report=_make_report(error="tool not found in registry"),
        )
        with patch("app.services.plan_orchestrator._result_processing._time") as mock_time:
            mixin._process_plan_results([result])
        assert mixin._mcp_healthy is True
        assert mixin.state.tasks[0].status == TaskStatus.PENDING

    def test_probe_failure_prevents_retry(self):
        """If health probe fails, MCP should be marked unhealthy, no retry."""
        mixin = _make_mixin()
        mixin.worker_service.check_mcp_health.return_value = False
        # Mark task as not running so no active workers → probe path is taken
        mixin.state.tasks[0].status = TaskStatus.COMPLETED
        result = TaskResult(
            task_id="t1",
            worker_id="w1",
            status="partial",
            plan_report=_make_report(error="tool not found in registry"),
        )
        mixin._process_plan_results([result])
        assert mixin._mcp_healthy is False
        assert mixin.state.mcp_consecutive_failures == 1

    def test_probe_skipped_when_workers_active(self):
        """When workers are still running, probe should be skipped."""
        mixin = _make_mixin()
        # Add an active RUNNING task (the default task is already RUNNING,
        # so probe is skipped). Verify check_mcp_health is NOT called.
        result = TaskResult(
            task_id="t1",
            worker_id="w1",
            status="partial",
            plan_report=_make_report(error="tool not found in registry"),
        )
        with patch("app.services.plan_orchestrator._result_processing._time") as mock_time:
            mixin._process_plan_results([result])
        # Task is still RUNNING → probe skipped → retry happens
        assert mixin._mcp_healthy is True
        assert mixin.state.tasks[0].status == TaskStatus.PENDING
        mixin.worker_service.check_mcp_health.assert_not_called()

    def test_probe_failure_increments_consecutive_failures(self):
        """Probe failure should increment mcp_consecutive_failures."""
        mixin = _make_mixin()
        mixin.worker_service.check_mcp_health.return_value = False
        # Remove the running task so probe path is taken
        mixin.state.tasks.clear()
        mixin.state.tasks.append(
            Task(task_id="t1", assigned_worker_id="w1", max_attempts=3)
        )
        mixin.state.tasks[0].status = TaskStatus.COMPLETED
        mixin.state.tasks[0].metadata["plan_mode"] = True
        mixin.state.tasks[0].metadata["stage_number"] = 1
        mixin.state.tasks[0].metadata["plan_version"] = 1

        result = TaskResult(
            task_id="t1",
            worker_id="w1",
            status="partial",
            plan_report=_make_report(error="tool not found in registry"),
        )
        mixin._process_plan_results([result])
        assert mixin.state.mcp_consecutive_failures == 1
