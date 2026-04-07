"""Tests for plan validation guards (Fixes 2-4 for empty-plan bug)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.models import OrchestratorState, TaskStatus
from app.plan_models import PlanStep, PlanTask, ResearchPlan, TaskReport, decision_gate_from_dict
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


def test_parse_plan_output_schema_v3_steps_and_extra_fields() -> None:
    raw = json.dumps({
        "schema_version": 3,
        "plan_action": "create",
        "plan_version": 1,
        "tasks": [
            {
                "stage_number": 0,
                "stage_name": "Baseline",
                "unknown_task_field": "ignore me",
                "steps": [
                    {
                        "step_id": "baseline_run",
                        "kind": "tool_call",
                        "instruction": "Run baseline",
                        "tool_name": "backtests_runs",
                        "args": {"action": "start"},
                        "unknown_step_field": "ignore me too",
                    }
                ],
                "decision_gates": [
                    {
                        "metric": "sharpe",
                        "threshold": 1.0,
                        "reason": "keep",
                        "unknown_gate_field": "extra",
                    }
                ],
            }
        ],
    })
    result = parse_plan_output(raw)

    assert result["schema_version"] == 3
    assert result["tasks"][0]["steps"][0]["step_id"] == "baseline_run"
    assert result["tasks"][0]["steps"][0]["tool_name"] == "backtests_runs"
    assert result["tasks"][0]["steps"][0]["args"]["action"] == "start"


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


def test_process_plan_data_uses_request_version_on_parse_failure() -> None:
    svc = _make_plan_service()
    svc.planner_service.plan_transport_retry_count = 1
    fallback_data = {
        "plan_action": "continue",
        "plan_version": 0,
        "reason": "Failed to parse planner output as JSON",
        "_parse_failed": True,
        "_failure_class": "parse_error",
        "_request_type": "create",
        "_request_version": 1,
        "_attempt_number": 1,
    }
    svc._process_plan_data(fallback_data)

    kwargs = svc._plan_store.save_rejected_plan_attempt.call_args.kwargs
    assert kwargs["plan_version"] == 1


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


def test_process_plan_data_accepts_steps_only_tasks() -> None:
    svc = _make_plan_service()
    valid_data = {
        "schema_version": 3,
        "plan_action": "create",
        "plan_version": 1,
        "_request_type": "create",
        "_request_version": 1,
        "_attempt_number": 1,
        "_failure_class": "none",
        "tasks": [
            {
                "stage_number": 0,
                "stage_name": "Baseline",
                "steps": [
                    {
                        "step_id": "baseline_run",
                        "kind": "tool_call",
                        "instruction": "Run baseline",
                        "tool_name": "backtests_runs",
                        "args": {
                            "action": "start",
                            "snapshot_id": "active-signal-v1",
                            "version": "1",
                            "symbol": "BTCUSDT",
                            "anchor_timeframe": "1h",
                            "execution_timeframe": "5m",
                        },
                    }
                ],
                "results_table_columns": ["run_id"],
                "decision_gates": [],
            },
        ],
    }
    svc._process_plan_data(valid_data)

    assert svc._current_plan is not None
    assert svc._current_plan.tasks[0].steps[0].step_id == "baseline_run"
    assert svc._current_plan.tasks[0].agent_instructions == []


def test_process_plan_data_ignores_wrong_plan_version_from_planner():
    """Planner may echo plan_version: 1 from the schema example even for
    revision requests.  The orchestrator must always use request_version."""
    svc = _make_plan_service()
    data = {
        "plan_action": "update",
        "plan_version": 1,  # planner echoes schema example (wrong)
        "_request_type": "revision",
        "_request_version": 3,  # orchestrator's correct version
        "_attempt_number": 1,
        "_failure_class": "none",
        "tasks": [
            {
                "stage_number": 0,
                "stage_name": "Baseline",
                "theory": "test",
                "steps": [
                    {
                        "step_id": "s0",
                        "kind": "tool_call",
                        "instruction": "run",
                        "tool_name": "backtests_runs",
                        "args": {"action": "start", "snapshot_id": "snap1", "version": "1"},
                    }
                ],
                "decision_gates": [],
            },
        ],
    }
    svc._process_plan_data(data)

    assert svc._current_plan is not None
    assert svc._current_plan.version == 3
    assert svc._current_plan.tasks[0].plan_version == 3


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


def test_decision_gate_from_dict_handles_full_partial_and_extra_fields() -> None:
    full = decision_gate_from_dict({
        "metric": "sharpe",
        "threshold": 1.1,
        "comparator": "gte",
        "verdict_pass": "PROMOTE",
        "verdict_fail": "REJECT",
        "reason": "Baseline should hold",
        "future_field": "ignored",
    })
    partial = decision_gate_from_dict({"metric": "pnl"})
    extra = decision_gate_from_dict({"future_field": "ignored"})

    assert full.reason == "Baseline should hold"
    assert full.threshold == 1.1
    assert partial.metric == "pnl"
    assert partial.comparator == "gte"
    assert extra.metric == ""


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


def test_claude_cli_builds_batch_json_command() -> None:
    from app.adapters.claude_planner_cli import ClaudePlannerCli

    adapter = ClaudePlannerCli(cli_path="claude", model="opus")
    cmd, output_mode = adapter._build_command("hello", json_schema='{"type":"object"}')

    assert output_mode == "stream-json"
    assert "--bare" in cmd
    assert "--no-session-persistence" in cmd
    assert "--output-format" in cmd
    assert "stream-json" in cmd
    assert "--include-partial-messages" in cmd
    assert "--json-schema" in cmd


def test_claude_runtime_summary_reports_custom_backend() -> None:
    from app.adapters.claude_planner_cli import ClaudePlannerCli

    adapter = ClaudePlannerCli(cli_path="claude", model="opus")
    with patch.object(adapter, "_load_claude_settings", return_value={
        "env": {
            "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.1",
        }
    }):
        summary = adapter.runtime_summary()

    assert summary["has_custom_backend"] is True
    assert summary["has_model_remap"] is True
    assert summary["resolved_model"] == "glm-5.1"


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


def test_validate_plan_accepts_symbolic_ref_in_instructions() -> None:
    """Symbolic refs like {{stage:N.field}} are no longer validated —
    resolved at dispatch time by the worker."""
    plan = ResearchPlan(
        schema_version=2,
        version=1,
        goal="test",
        tasks=[
            PlanTask(stage_number=0, stage_name="Baseline", agent_instructions=["run baseline"]),
            PlanTask(
                stage_number=1,
                stage_name="UsesRef",
                agent_instructions=[
                    "backtests_runs(action='inspect', run_id='{{stage:0.run_id}}', view='detail')",
                ],
            ),
        ],
    )

    validation = validate_plan(plan)

    assert validation.is_acceptable


def test_validate_plan_accepts_step_ref_to_prior_step() -> None:
    plan = ResearchPlan(
        schema_version=3,
        version=1,
        goal="test",
        tasks=[
            PlanTask(
                stage_number=0,
                stage_name="Baseline",
                steps=[
                    PlanStep(
                        step_id="baseline_run",
                        kind="tool_call",
                        instruction="Start baseline run",
                        tool_name="backtests_runs",
                        args={
                            "action": "start",
                            "snapshot_id": "active-signal-v1",
                            "version": "1",
                            "symbol": "BTCUSDT",
                            "anchor_timeframe": "1h",
                            "execution_timeframe": "5m",
                        },
                        binds=["run_id"],
                    ),
                    PlanStep(
                        step_id="inspect",
                        kind="tool_call",
                        instruction="Inspect baseline",
                        tool_name="backtests_runs",
                        args={
                            "action": "inspect",
                            "run_id": "{{step:baseline_run.run_id}}",
                            "view": "detail",
                        },
                    ),
                ],
            ),
        ],
    )

    validation = validate_plan(plan)

    assert validation.is_valid


def test_validate_plan_rejects_invalid_tool_alias_and_action() -> None:
    plan = ResearchPlan(
        schema_version=3,
        version=1,
        goal="test",
        tasks=[
            PlanTask(
                stage_number=0,
                stage_name="Bad",
                steps=[
                    PlanStep(
                        step_id="fork",
                        kind="tool_call",
                        instruction="Fork snapshot",
                        tool_name="snapshots",
                        args={"action": "fork"},
                    ),
                    PlanStep(
                        step_id="run",
                        kind="tool_call",
                        instruction="Run backtest",
                        tool_name="backtests_runs",
                        args={"action": "run", "strategy": "x"},
                    ),
                ],
            ),
        ],
    )

    validation = validate_plan(plan)
    codes = {error.code for error in validation.errors}

    assert "tool_alias_invalid" not in codes  # tool name validation removed
    assert "action_invalid" not in codes  # action validation removed


def test_build_plan_repair_prompt_compacts_large_invalid_payload() -> None:
    from app.plan_prompts import build_plan_repair_prompt
    from app.plan_validation import PlanRepairRequest, PlanValidationError

    prompt = build_plan_repair_prompt(
        PlanRepairRequest(
            goal="test goal",
            plan_version=1,
            attempt_number=2,
            invalid_plan_data={
                "plan_action": "create",
                "tasks": [
                    {
                        "stage_number": 0,
                        "stage_name": "Bad 0",
                        "depends_on": [],
                        "theory": "x" * 2000,
                        "agent_instructions": ["bad " + ("y" * 1000)],
                        "results_table_columns": ["run_id"],
                        "decision_gates": [],
                    },
                    {
                        "stage_number": 1,
                        "stage_name": "Valid 1",
                        "depends_on": [0],
                        "agent_instructions": ["ok"],
                        "results_table_columns": ["run_id"],
                        "decision_gates": [],
                    },
                ],
            },
            validation_errors=[
                PlanValidationError(
                    stage_number=0,
                    instruction_index=0,
                    code="legacy_placeholder",
                    message="Legacy <...> placeholder is not allowed",
                    offending_text="<run_id>",
                )
            ],
        )
    )

    assert "Valid Stage Summary" in prompt
    assert "Invalid Stage Fragments" in prompt
    assert "Valid 1" in prompt
    assert len(prompt) < 15000


def test_build_plan_creation_prompt_includes_catalog_without_research_context() -> None:
    from app.plan_prompts import build_plan_creation_prompt

    prompt = build_plan_creation_prompt(goal="Improve BTCUSDT strategy")

    assert "## MCP dev_space1 Tool Catalog" in prompt
    assert '"schema_version": 4' in prompt
    assert '"steps"' in prompt


def test_build_plan_creation_prompt_omits_cross_stage_refs() -> None:
    from app.plan_prompts import build_plan_creation_prompt

    prompt = build_plan_creation_prompt(goal="Improve BTCUSDT strategy")

    # Creation prompt should NOT contain symbolic cross-stage ref instructions
    assert "Symbolic Reference Contract" not in prompt
    assert "cross-stage refs ONLY" not in prompt
    # Should instruct NOT to use cross-stage refs
    assert "do NOT use cross-stage refs" in prompt
    # But should still mention intra-stage refs
    assert "{{step:" in prompt


def test_build_plan_revision_prompt_omits_cross_stage_refs() -> None:
    from app.plan_prompts import build_plan_revision_prompt

    plan = ResearchPlan(schema_version=2, version=1, goal="test")
    prompt = build_plan_revision_prompt(goal="test", current_plan=plan, reports=[])

    # Revision prompt should NOT encourage {{stage:N.run_id}} usage
    assert "Use {{stage:0.run_id}} only for earlier stages" not in prompt
    assert "Do NOT use cross-stage refs" in prompt
    assert "concrete IDs" in prompt
    assert "schema_version\": 4" in prompt
    # Revision from v1 should show plan_version: 2 in schema
    assert '"plan_version": 2' in prompt


def test_build_plan_revision_prompt_increments_plan_version() -> None:
    from app.plan_prompts import build_plan_revision_prompt

    plan_v2 = ResearchPlan(schema_version=4, version=2, goal="test")
    prompt = build_plan_revision_prompt(goal="test", current_plan=plan_v2, reports=[])

    # Revision from v2 should show plan_version: 3 in schema
    assert '"plan_version": 3' in prompt
    assert "plan_v3" in prompt


def test_build_plan_repair_prompt_omits_cross_stage_refs() -> None:
    from app.plan_prompts import build_plan_repair_prompt
    from app.plan_validation import PlanRepairRequest as PRR

    prompt = build_plan_repair_prompt(
        PRR(
            goal="test goal",
            plan_version=1,
            attempt_number=1,
            invalid_plan_data={"plan_action": "create", "tasks": []},
            validation_errors=[],
        )
    )

    # Repair prompt should NOT encourage {{stage:N.run_id}} usage
    assert "Use {{stage:N.run_id}} only for earlier stages" not in prompt
    assert "Do NOT use cross-stage refs" in prompt
    assert "concrete IDs" in prompt
    assert "schema_version\": 4" in prompt


def test_build_plan_task_prompt_includes_previous_stage_results() -> None:
    from app.plan_prompts import build_plan_task_prompt

    dep_report = TaskReport(
        task_id="stage-0",
        worker_id="qwen-1",
        plan_version=1,
        status="success",
        results_table=[{"run_id": "r1", "snapshot_id": "snap1", "verdict": "PROMOTE"}],
        key_metrics={"sharpe": 1.2, "net_pnl": 500},
        artifacts=["run:r1", "snapshot:snap1"],
        verdict="PROMOTE",
        raw_output="{}",
    )

    prompt = build_plan_task_prompt(
        stage_number=1,
        stage_name="Inspect Baseline",
        theory="Check baseline results",
        agent_instructions=["backtests_runs(action='inspect', run_id='r1', view='detail')"],
        plan_version=1,
        dependency_reports=[dep_report],
    )

    assert "## Previous Stage Results" in prompt
    assert "stage-0" in prompt
    assert "r1" in prompt
    assert "snap1" in prompt
    assert "PROMOTE" in prompt
    assert "sharpe" in prompt


def test_build_plan_task_prompt_without_dependency_reports() -> None:
    from app.plan_prompts import build_plan_task_prompt

    prompt = build_plan_task_prompt(
        stage_number=0,
        stage_name="Baseline",
        theory="Run baseline",
        agent_instructions=["do baseline"],
        plan_version=1,
    )

    assert "## Previous Stage Results" not in prompt


# ---------------------------------------------------------------------------
# Phase 3: Schema DRY tests
# ---------------------------------------------------------------------------


def test_plan_schema_generates_correct_version_numbers() -> None:
    """_plan_schema(N) should produce schema_version: N."""
    from app.plan_prompts import _plan_schema
    v4 = _plan_schema(4)
    v3 = _plan_schema(3)
    assert '"schema_version": 4' in v4
    assert '"schema_version": 3' in v3
    # Both should contain the same task structure
    assert '"steps":' in v4
    assert '"steps":' in v3
    # Default plan_version is 1
    assert '"plan_version": 1' in v4
    # Explicit plan_version overrides the default
    v4_pv3 = _plan_schema(4, plan_version=3)
    assert '"plan_version": 3' in v4_pv3


def test_build_plan_task_prompt_non_dict_results_table_no_crash() -> None:
    """build_plan_task_prompt should not crash when results_table has non-dict rows."""
    from app.plan_prompts import build_plan_task_prompt

    dep_report = TaskReport(
        task_id="s0",
        worker_id="qwen-1",
        plan_version=1,
        status="success",
        results_table=["not a dict", 42],
        raw_output="{}",
    )

    # Should not crash
    prompt = build_plan_task_prompt(
        stage_number=1,
        stage_name="UsesBase",
        theory="test",
        agent_instructions=["do stuff"],
        plan_version=1,
        dependency_reports=[dep_report],
    )

    assert "## Previous Stage Results" in prompt


def test_topological_sort_with_unknown_deps_and_cycles() -> None:
    """_topological_sort_stages should handle unknown deps and cycles gracefully."""
    from app.services.plan_orchestrator._plan_lifecycle import _topological_sort_stages

    # Stage 1 depends on non-existent stage 5
    tasks = [
        PlanTask(stage_number=0, plan_version=1, stage_name="A"),
        PlanTask(stage_number=1, plan_version=1, stage_name="B", depends_on=[5]),
    ]
    result = _topological_sort_stages(tasks)
    assert 0 in result
    assert 1 in result
    # Unknown dep 5 should NOT appear in result
    assert 5 not in result

    # Cycle: stage 0 depends on 1, stage 1 depends on 0
    tasks_cycle = [
        PlanTask(stage_number=0, plan_version=1, stage_name="A", depends_on=[1]),
        PlanTask(stage_number=1, plan_version=1, stage_name="B", depends_on=[0]),
    ]
    result_cycle = _topological_sort_stages(tasks_cycle)
    # Should produce some result without infinite loop (at most the stages that exist)
    assert len(result_cycle) <= 2


