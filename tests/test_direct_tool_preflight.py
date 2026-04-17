from __future__ import annotations

from app.execution_models import BaselineRef, ExecutionPlan, PlanSlice
from app.services.direct_execution.budgeting import normalize_plan_budgets
from app.services.direct_execution.backtests_protocol import backtests_start_guard_payload
from app.services.direct_execution.tool_preflight import preflight_direct_tool_call

from tests.mcp_catalog_fixtures import make_catalog_snapshot


def test_preflight_blocks_missing_required_field_without_spending_budget() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "research_search",
        {"project_id": "proj_1"},
        catalog_snapshot=snapshot,
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["error_class"] == "agent_contract_misuse"
    assert result.local_payload["details"]["missing_field"] == "query"
    assert result.local_payload["details"]["tool_name"] == "research_search"


def test_preflight_blocks_wrong_enum_with_allowed_values() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "experiments_read",
        {"view": "binary", "job_id": "job_1"},
        catalog_snapshot=snapshot,
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert "allowed_values" in result.local_payload["details"]
    assert "text" in result.local_payload["details"]["allowed_values"]


def test_preflight_blocks_wrong_type() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "features_catalog",
        {"scope": "timeframe", "timeframe": 15},
        catalog_snapshot=snapshot,
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert "is not of type" in result.local_payload["summary"]
    assert result.local_payload["details"]["json_path"] == "timeframe"


def test_preflight_blocks_unknown_field_when_additional_properties_false() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "research_map",
        {"action": "inspect", "project_id": "proj_1", "bogus": True},
        catalog_snapshot=snapshot,
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert "Additional properties are not allowed" in result.local_payload["summary"]


def test_preflight_blocks_unknown_tool_against_live_catalog() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "not_a_real_tool",
        {"foo": "bar"},
        catalog_snapshot=snapshot,
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["details"]["tool_name"] == "not_a_real_tool"
    assert "available_tools" in result.local_payload["details"]


def test_preflight_allows_valid_call() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "research_map",
        {"action": "inspect", "project_id": "proj_1"},
        catalog_snapshot=snapshot,
    )

    assert result.local_payload is None
    assert result.charge_budget is True
    assert result.arguments == {"action": "inspect", "project_id": "proj_1"}


def test_preflight_blocks_features_analytics_without_feature_name() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "features_analytics",
        {"action": "analytics"},
        catalog_snapshot=snapshot,
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["details"]["missing_field"] == "feature_name"
    assert "requires feature_name" in result.local_payload["summary"]


def test_preflight_blocks_features_custom_detail_without_name() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "features_custom",
        {"action": "inspect", "view": "detail"},
        catalog_snapshot=snapshot,
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["details"]["missing_field"] == "name"
    assert "requires name" in result.local_payload["summary"]


def test_preflight_blocks_suspicious_backtests_run_id_without_spending_budget() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "backtests_runs",
        {"action": "status", "run_id": "1"},
        catalog_snapshot=snapshot,
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["details"]["reason_code"] == "suspicious_durable_handle"
    assert result.local_payload["details"]["field_name"] == "run_id"


def test_preflight_blocks_numeric_backtests_run_id_before_schema_validation() -> None:
    snapshot = make_catalog_snapshot()

    for value in (1, "3"):
        result = preflight_direct_tool_call(
            "backtests_runs",
            {"action": "status", "run_id": value},
            catalog_snapshot=snapshot,
        )

        assert result.charge_budget is False
        assert result.local_payload is not None
        assert result.local_payload["details"]["reason_code"] == "suspicious_durable_handle"
        assert result.local_payload["details"]["field_name"] == "run_id"


def test_preflight_blocks_cross_kind_backtests_run_id_without_spending_budget() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "backtests_runs",
        {"action": "detail", "run_id": "analysis-d1b90901b5be-rm-feature-short__feature-long"},
        catalog_snapshot=snapshot,
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["details"]["reason_code"] == "suspicious_durable_handle"
    assert result.local_payload["details"]["field_name"] == "run_id"


def test_preflight_allows_valid_backtests_run_id_observe_call() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "backtests_runs",
        {"action": "status", "run_id": "20260413-143910-3db43a14"},
        catalog_snapshot=snapshot,
    )

    assert result.local_payload is None
    assert result.charge_budget is True


def test_backtests_start_requires_successful_plan_in_current_attempt() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "backtests_runs",
        {"action": "start", "snapshot_id": "active-signal-v1"},
        catalog_snapshot=snapshot,
        runtime_profile="backtests_integration_analysis",
        transcript=[],
    )

    assert result.charge_budget is False
    assert result.local_payload is not None
    assert result.local_payload["details"]["reason_code"] == "backtests_plan_required_before_start"


def test_backtests_start_after_successful_plan_is_allowed() -> None:
    snapshot = make_catalog_snapshot()

    result = preflight_direct_tool_call(
        "backtests_runs",
        {"action": "start", "snapshot_id": "candidate-signal-v1"},
        catalog_snapshot=snapshot,
        runtime_profile="backtests_integration_analysis",
        transcript=[
            {
                "kind": "tool_result",
                "tool": "backtests_plan",
                "payload": {"ok": True, "payload": {"structuredContent": {"status": "ready"}}},
            }
        ],
        baseline_bootstrap={"baseline_snapshot_id": "active-signal-v1", "baseline_version": 1},
    )

    assert result.local_payload is None
    assert result.charge_budget is True


def test_backtests_start_after_handle_misuse_is_blocked_even_after_plan() -> None:
    payload = backtests_start_guard_payload(
        tool_name="backtests_runs",
        arguments={"action": "start", "snapshot_id": "candidate-signal-v1"},
        runtime_profile="backtests_integration_analysis",
        baseline_bootstrap={"baseline_snapshot_id": "active-signal-v1", "baseline_version": 1},
        transcript=[
            {
                "kind": "tool_result",
                "tool": "backtests_runs",
                "payload": {
                    "error_class": "agent_contract_misuse",
                    "details": {"reason_code": "suspicious_durable_handle", "field_name": "run_id"},
                },
            },
            {
                "kind": "tool_result",
                "tool": "backtests_plan",
                "payload": {"ok": True, "payload": {"structuredContent": {"status": "ready"}}},
            },
        ],
    )

    assert payload is not None
    assert payload["details"]["reason_code"] == "backtests_start_blocked_after_handle_misuse"


def test_duplicate_baseline_start_is_blocked_from_saved_run_list() -> None:
    payload = backtests_start_guard_payload(
        tool_name="backtests_runs",
        arguments={
            "action": "start",
            "snapshot_id": "active-signal-v1",
            "version": 1,
            "symbol": "BTCUSDT",
        },
        runtime_profile="backtests_integration_analysis",
        baseline_bootstrap={"baseline_snapshot_id": "active-signal-v1", "baseline_version": 1, "symbol": "BTCUSDT"},
        transcript=[
            {
                "kind": "tool_result",
                "tool": "backtests_plan",
                "payload": {"ok": True, "payload": {"structuredContent": {"status": "ready"}}},
            },
            {
                "kind": "tool_result",
                "tool": "backtests_runs",
                "arguments": {"action": "inspect", "view": "list"},
                "payload": {
                    "ok": True,
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {
                                "saved_runs": [
                                    {
                                        "run_id": "20260413-143910-3db43a14",
                                        "status": "completed",
                                        "symbol": "BTCUSDT",
                                        "strategy_snapshot_id": "active-signal-v1",
                                        "strategy_snapshot_version": 1,
                                    }
                                ]
                            },
                        }
                    },
                },
            },
        ],
    )

    assert payload is not None
    assert payload["details"]["reason_code"] == "duplicate_baseline_start_blocked"


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
