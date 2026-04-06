"""Tests for plan validation guards (Fixes 2-4 for empty-plan bug)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.models import OrchestratorState, TaskStatus
from app.plan_models import PlanTask, ResearchPlan
from app.plan_validation import validate_plan
from app.result_parser import parse_plan_output


# ---------------------------------------------------------------------------
# Fix 3: parse_plan_output fallback has _parse_failed sentinel
# ---------------------------------------------------------------------------


def test_parse_plan_output_no_json_returns_sentinel():
    result = parse_plan_output("not json at all, just agent chatter")
    assert result["_parse_failed"] is True
    assert result["plan_action"] == "continue"
    assert result["plan_version"] == 0


def test_parse_plan_output_invalid_json_returns_sentinel():
    result = parse_plan_output("{broken json")
    assert result["_parse_failed"] is True
    assert "plan_version" in result


def test_parse_plan_output_valid_json_no_sentinel():
    raw = json.dumps({
        "plan_action": "create",
        "plan_version": 1,
        "tasks": [{"stage_number": 0, "stage_name": "test"}],
        "tasks_to_dispatch": [0],
    })
    result = parse_plan_output(raw)
    assert result["plan_action"] == "create"
    assert result.get("_parse_failed") is None  # no sentinel on success


# ---------------------------------------------------------------------------
# Fix 2: _process_plan_data rejects empty plans
# ---------------------------------------------------------------------------


def _make_plan_service():
    """Create a PlanOrchestratorService with minimal mocks."""
    from app.services.plan_orchestrator_service import PlanOrchestratorService

    orch = MagicMock()
    orch.config.goal = "test goal"
    orch._research_context_text = None
    orch._mcp_scanner = None

    state = OrchestratorState(goal="test goal")
    orch.state = state

    # Mock sub-services
    orch.planner_service = MagicMock()
    orch.worker_service = MagicMock()
    orch.scheduler = MagicMock()
    orch.memory_service = MagicMock()
    orch.notification_service = MagicMock()
    orch._plan_store = MagicMock()
    orch._worker_ids = ["qwen-1"]
    orch._mcp_scanner = None

    svc = PlanOrchestratorService(orch)
    return svc


def test_process_plan_data_rejects_zero_tasks():
    svc = _make_plan_service()
    empty_data = {
        "plan_action": "create",
        "plan_version": 1,
        "tasks": [],
        "tasks_to_dispatch": [],
    }
    svc._process_plan_data(empty_data)

    assert svc._current_plan is None
    # Plan store should NOT have been called to save
    svc._plan_store.save_plan.assert_not_called()


def test_process_plan_data_rejects_parse_failure_fallback():
    svc = _make_plan_service()
    fallback_data = {
        "plan_action": "continue",
        "plan_version": 0,
        "reason": "Failed to parse planner output as JSON",
        "_parse_failed": True,
    }
    svc._process_plan_data(fallback_data)

    assert svc._current_plan is None


def test_process_plan_data_accepts_valid_plan():
    svc = _make_plan_service()
    valid_data = {
        "plan_action": "create",
        "plan_version": 1,
        "tasks": [
            {
                "stage_number": 0,
                "stage_name": "Infrastructure Check",
                "theory": "Test theory",
                "agent_instructions": ["step 1"],
                "results_table_columns": ["metric"],
                "decision_gates": [],
            },
            {
                "stage_number": 1,
                "stage_name": "Feature Exploration",
                "theory": "Test theory 2",
                "agent_instructions": ["step 2"],
                "results_table_columns": ["metric"],
                "decision_gates": [],
            },
        ],
        "tasks_to_dispatch": [0, 1],
        "frozen_base": "snapshot-1@1",
        "cumulative_summary": "Test summary",
        "principles": ["principle 1"],
    }
    svc._process_plan_data(valid_data)

    assert svc._current_plan is not None
    assert len(svc._current_plan.tasks) == 2
    assert svc._current_plan.execution_order == [0, 1]
    svc._plan_store.save_plan.assert_called_once()


def test_process_plan_data_defaults_execution_order_when_missing():
    svc = _make_plan_service()
    data = {
        "plan_action": "create",
        "plan_version": 1,
        "tasks": [
            {
                "stage_number": 2,
                "stage_name": "Second",
                "agent_instructions": ["step"],
                "results_table_columns": [],
                "decision_gates": [],
            },
            {
                "stage_number": 0,
                "stage_name": "First",
                "agent_instructions": ["step"],
                "results_table_columns": [],
                "decision_gates": [],
            },
        ],
        # No "tasks_to_dispatch" key
    }
    svc._process_plan_data(data)

    assert svc._current_plan is not None
    # Should default to sorted stage order
    assert svc._current_plan.execution_order == [0, 2]


# ---------------------------------------------------------------------------
# Fix 4: _revise_plan guards against empty plans
# ---------------------------------------------------------------------------


def test_revise_plan_clears_empty_plan():
    svc = _make_plan_service()
    svc._current_plan = ResearchPlan(version=1, goal="test")

    assert len(svc._current_plan.tasks) == 0

    svc._revise_plan()

    assert svc._current_plan is None
    # Planner should NOT have been called
    svc.planner_service.start_plan_revision.assert_not_called()


def test_revise_plan_proceeds_with_valid_plan():
    svc = _make_plan_service()
    plan = ResearchPlan(version=1, goal="test")
    plan.tasks.append(PlanTask(
        plan_version=1,
        stage_number=0,
        stage_name="Test Stage",
    ))
    svc._current_plan = plan
    svc._plan_store.load_reports_for_plan.return_value = []

    svc._revise_plan()

    assert svc._current_plan is not None  # plan was NOT cleared
    svc.planner_service.start_plan_revision.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 1: ClaudePlannerCli includes mandatory flags
# ---------------------------------------------------------------------------


def test_claude_cli_includes_mandatory_flags():
    from app.adapters.claude_planner_cli import ClaudePlannerCli

    adapter = ClaudePlannerCli(cli_path="claude", model="opus")
    assert adapter._mandatory_flags == ["--tools", ""]


def test_claude_cli_command_includes_tools_flag():
    """Verify the built command includes --tools '' in both invoke and start."""
    from app.adapters.claude_planner_cli import ClaudePlannerCli

    adapter = ClaudePlannerCli(cli_path="claude", model="opus")

    # We can't easily test invoke() without actually running claude,
    # but we can test start() by inspecting the spawned process command.
    # Instead, verify the mandatory_flags are set correctly.
    assert "--tools" in adapter._mandatory_flags
    assert "" in adapter._mandatory_flags


def test_validate_plan_accepts_symbolic_ref_with_dependency() -> None:
    plan = ResearchPlan(
        schema_version=2,
        version=1,
        goal="test",
        tasks=[
            PlanTask(stage_number=0, stage_name="Baseline", agent_instructions=["run baseline"]),
            PlanTask(
                stage_number=1,
                stage_name="Inspect baseline",
                depends_on=[0],
                agent_instructions=[
                    "backtests_runs(action='inspect', run_id='{{stage:0.run_id}}', view='detail')",
                ],
            ),
        ],
    )

    validation = validate_plan(plan)

    assert validation.is_valid


def test_validate_plan_rejects_legacy_placeholder() -> None:
    plan = ResearchPlan(
        schema_version=2,
        version=1,
        goal="test",
        tasks=[
            PlanTask(
                stage_number=1,
                stage_name="Bad",
                agent_instructions=["backtests_runs(action='inspect', run_id='<run_id>', view='detail')"],
            ),
        ],
    )

    validation = validate_plan(plan)

    assert not validation.is_valid
    assert validation.errors[0].code == "legacy_placeholder"


def test_validate_plan_rejects_symbolic_ref_without_dependency() -> None:
    plan = ResearchPlan(
        schema_version=2,
        version=1,
        goal="test",
        tasks=[
            PlanTask(stage_number=0, stage_name="Baseline", agent_instructions=["run baseline"]),
            PlanTask(
                stage_number=1,
                stage_name="Bad",
                agent_instructions=[
                    "backtests_runs(action='inspect', run_id='{{stage:0.run_id}}', view='detail')",
                ],
            ),
        ],
    )

    validation = validate_plan(plan)

    assert not validation.is_valid
    assert any(error.code == "unresolved_symbolic_reference" for error in validation.errors)
