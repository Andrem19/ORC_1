from __future__ import annotations

from app.execution_models import BaselineRef, ExecutionPlan, PlanSlice
from app.services.direct_execution.budgeting import normalize_plan_budgets
from app.services.direct_execution.tool_preflight import preflight_direct_tool_call


def test_preflight_repairs_research_record_hypothesis_atlas_shape() -> None:
    result = preflight_direct_tool_call(
        "research_record",
        {
            "action": "create",
            "kind": "hypothesis",
            "project_id": "proj_1",
            "record": {
                "title": "H1",
                "summary": "s",
                "atlas": {
                    "statement": "x",
                    "expected_outcome": "y",
                    "falsification_criteria": "z",
                },
            },
            "coordinates": "{'market_regime': 'trend', 'timeframe': '1h'}",
        },
    )

    assert result.local_payload is None
    assert result.charge_budget is True
    assert result.arguments["atlas"]["coordinates"] == {"market_regime": "trend", "timeframe": "1h"}
    assert "atlas" not in result.arguments["record"]


def test_preflight_blocks_research_record_hypothesis_without_required_atlas_block() -> None:
    result = preflight_direct_tool_call(
        "research_record",
        {
            "action": "create",
            "kind": "hypothesis",
            "project_id": "proj_1",
            "record": {"title": "H1", "summary": "s"},
        },
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["error_class"] == "agent_contract_misuse"
    assert "top-level atlas block" in result.local_payload["summary"]


def test_preflight_blocks_research_search_without_query_without_spending_budget() -> None:
    result = preflight_direct_tool_call(
        "research_search",
        {"project_id": "proj_1", "limit": 20},
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["summary"] == "research_search requires a non-empty query"


def test_normalize_plan_budgets_scales_by_six_for_direct_runtime() -> None:
    plan = ExecutionPlan(
        plan_id="plan_1",
        goal="g",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=[],
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="t",
                hypothesis="h",
                objective="o",
                success_criteria=["done"],
                allowed_tools=["research_record"],
                evidence_requirements=["done"],
                policy_tags=["analysis"],
                max_turns=4,
                max_tool_calls=3,
                max_expensive_calls=0,
                parallel_slot=1,
            )
        ],
    )

    normalize_plan_budgets(plan)

    slice_obj = plan.slices[0]
    assert slice_obj.max_turns == 24
    assert slice_obj.max_tool_calls == 18
    assert slice_obj.max_expensive_calls >= 2
    assert slice_obj.budget_scale_applied == 6


# --- research_map ---

def test_preflight_blocks_research_map_without_project_id_no_budget_charge() -> None:
    result = preflight_direct_tool_call("research_map", {"action": "inspect", "node_id": "node_abc"})
    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["error_class"] == "agent_contract_misuse"
    assert "project_id" in result.local_payload["summary"]


def test_preflight_allows_research_map_with_project_id() -> None:
    result = preflight_direct_tool_call("research_map", {"action": "inspect", "project_id": "proj_1"})
    assert result.local_payload is None
    assert result.charge_budget is True


# --- features_catalog ---

def test_preflight_normalizes_features_catalog_numeric_timeframe() -> None:
    result = preflight_direct_tool_call("features_catalog", {"scope": "timeframe", "timeframe": 15})
    assert result.local_payload is None
    assert result.charge_budget is True
    assert result.arguments["timeframe"] == "15m"
    assert result.repair_notes


# --- experiments_read ---

def test_preflight_blocks_experiments_read_without_job_id_no_budget_charge() -> None:
    result = preflight_direct_tool_call("experiments_read", {"view": "list"})
    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["error_class"] == "agent_contract_misuse"
    assert "job_id" in result.local_payload["summary"]


def test_preflight_allows_experiments_read_with_job_id() -> None:
    result = preflight_direct_tool_call("experiments_read", {"view": "json", "job_id": "job_123"})
    assert result.local_payload is None
    assert result.charge_budget is True


# --- experiments_inspect ---

def test_preflight_blocks_experiments_inspect_status_without_job_id() -> None:
    result = preflight_direct_tool_call("experiments_inspect", {"view": "status"})
    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["error_class"] == "agent_contract_misuse"
    assert result.local_payload["details"]["required_field"] == "job_id"


def test_preflight_allows_experiments_inspect_list_without_job_id() -> None:
    result = preflight_direct_tool_call("experiments_inspect", {"view": "list"})
    assert result.local_payload is None
    assert result.charge_budget is True


# --- backtests_conditions ---

def test_preflight_blocks_backtests_conditions_without_snapshot_id_no_budget_charge() -> None:
    result = preflight_direct_tool_call(
        "backtests_conditions",
        {"action": "run", "symbol": "BTCUSDT", "start_at": "2024-01-01T00:00:00Z", "end_at": "2024-12-31T23:59:59Z"},
    )
    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["error_class"] == "agent_contract_misuse"
    assert "snapshot_id" in result.local_payload["summary"]


def test_preflight_allows_backtests_conditions_with_snapshot_id() -> None:
    result = preflight_direct_tool_call(
        "backtests_conditions",
        {"action": "run", "snapshot_id": "candidate_v1", "symbol": "BTCUSDT"},
    )
    assert result.local_payload is None
    assert result.charge_budget is True


def test_preflight_splits_backtests_conditions_snapshot_ref_with_version() -> None:
    result = preflight_direct_tool_call(
        "backtests_conditions",
        {"action": "run", "snapshot_id": "active-signal-v1@1", "symbol": "BTCUSDT"},
    )
    assert result.local_payload is None
    assert result.arguments["snapshot_id"] == "active-signal-v1"
    assert result.arguments["version"] == "1"
    assert result.repair_notes


def test_preflight_allows_backtests_conditions_list_without_snapshot_id() -> None:
    result = preflight_direct_tool_call("backtests_conditions", {"action": "list"})
    assert result.local_payload is None
    assert result.charge_budget is True


# --- backtests_analysis ---

def test_preflight_blocks_backtests_analysis_start_without_run_id_no_budget_charge() -> None:
    result = preflight_direct_tool_call("backtests_analysis", {"action": "start", "analysis": "diagnostics"})
    assert result.charge_budget is True
    assert result.local_payload is None
    assert result.arguments["action"] == "list"
    assert result.repair_notes


def test_preflight_allows_backtests_analysis_non_start_action_without_run_id() -> None:
    result = preflight_direct_tool_call("backtests_analysis", {"action": "inspect", "view": "status"})
    assert result.local_payload is None
    assert result.charge_budget is True


def test_preflight_rewrites_backtests_analysis_start_with_snapshot_like_run_id_to_list() -> None:
    result = preflight_direct_tool_call(
        "backtests_analysis",
        {"action": "start", "analysis": "diagnostics", "run_id": "active-signal-v1@1"},
    )
    assert result.local_payload is None
    assert result.charge_budget is True
    assert result.arguments["action"] == "list"
    assert "run_id" not in result.arguments
    assert result.repair_notes


# --- research_record(action='status') ---


def test_preflight_blocks_research_record_status_without_operation_id() -> None:
    result = preflight_direct_tool_call(
        "research_record",
        {"action": "status", "wait": "completed"},
    )
    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["error_class"] == "agent_contract_misuse"
    assert "operation_id" in result.local_payload["summary"]


def test_preflight_allows_research_record_status_with_operation_id() -> None:
    result = preflight_direct_tool_call(
        "research_record",
        {"action": "status", "operation_id": "op-123"},
    )
    assert result.local_payload is None
    assert result.charge_budget is True
