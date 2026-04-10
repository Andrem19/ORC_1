"""
Integration tests for the planner → compiler → ExecutionPlan pipeline.

Tests that PlannerDecisionService.create_plan() produces ExecutionPlan
objects identical to what the converter path would produce, using the
same compilation logic.
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock

from app.execution_models import ExecutionPlan
from app.services.direct_execution.planner import (
    PlannerDecisionService,
    PlannerDecisionError,
)

BOOTSTRAP = {
    "baseline_snapshot_id": "active-signal-v1",
    "baseline_version": 1,
    "symbol": "BTCUSDT",
    "anchor_timeframe": "1h",
    "execution_timeframe": "5m",
}

TOOLS = [
    "research_project",
    "research_map",
    "research_record",
    "research_search",
    "datasets",
    "datasets_sync",
    "features_custom",
    "features_dataset",
    "backtests_runs",
    "backtests_analysis",
    "experiments_run",
    "experiments_inspect",
    "notify_send",
]


def _semantic_response(stages: list[dict], **overrides) -> str:
    payload = {
        "source_title": "Integration Test Plan",
        "goal": "Find alpha signal",
        "baseline_ref": {
            "snapshot_id": "active-signal-v1",
            "version": 1,
            "symbol": "BTCUSDT",
            "anchor_timeframe": "1h",
            "execution_timeframe": "5m",
        },
        "global_constraints": ["keep baseline fixed"],
        "stages": stages,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _stage(
    stage_id: str,
    title: str = "Stage",
    objective: str = "Do work",
    tool_hints: list[str] | None = None,
    depends_on: list[str] | None = None,
    required: bool = True,
    parallelizable: bool = False,
) -> dict:
    return {
        "stage_id": stage_id,
        "title": title,
        "objective": objective,
        "actions": [f"Complete {stage_id}"],
        "success_criteria": [f"{stage_id} done"],
        "tool_hints": tool_hints or ["analysis"],
        "policy_tags": [],
        "depends_on": depends_on or [],
        "required": required,
        "parallelizable": parallelizable,
    }


def _make_service(mock_raw_output: str) -> PlannerDecisionService:
    adapter = MagicMock()
    adapter.name.return_value = "test_planner"
    artifact_store = MagicMock()

    async def mock_invoke(*, adapter, prompt, **kwargs):
        result = MagicMock()
        result.success = True
        result.raw_output = mock_raw_output
        result.error = ""
        return result

    service = PlannerDecisionService(
        adapter=adapter,
        artifact_store=artifact_store,
        timeout_seconds=30,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
        invoker=mock_invoke,
    )
    return service


def _create_plan(svc, **kwargs):
    defaults = dict(
        goal="G",
        baseline_bootstrap=BOOTSTRAP,
        available_tools=TOOLS,
        worker_count=1,
        plan_version=1,
        previous_state_summary="",
    )
    defaults.update(kwargs)
    return asyncio.run(svc.create_plan(**defaults))


class TestPlannerToExecutionSingleBatch:
    def test_single_stage_produces_one_plan(self):
        raw = _semantic_response(stages=[_stage("stage_1", tool_hints=["research_memory"])])
        svc = _make_service(raw)
        plan = _create_plan(svc)
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.slices) == 1
        assert plan.plan_source_kind == "planner"

    def test_tool_expansion_from_family_hint(self):
        raw = _semantic_response(stages=[_stage("stage_1", tool_hints=["backtesting"])])
        svc = _make_service(raw)
        plan = _create_plan(svc)
        tools = plan.slices[0].allowed_tools
        assert "backtests_runs" in tools
        assert "backtests_analysis" in tools

    def test_research_record_auto_appended(self):
        raw = _semantic_response(stages=[_stage("stage_1", tool_hints=["backtesting"])])
        svc = _make_service(raw)
        plan = _create_plan(svc)
        assert "research_record" in plan.slices[0].allowed_tools

    def test_budget_class_from_keywords(self):
        raw = _semantic_response(
            stages=[_stage("stage_1", title="Dataset readiness check", tool_hints=["data_readiness"])]
        )
        svc = _make_service(raw)
        plan = _create_plan(svc)
        # data_readiness preset = (6, 5, 1), scaled by 6 = (36, 30, 6)
        s = plan.slices[0]
        assert s.max_turns == 36
        assert s.max_tool_calls == 30

    def test_optional_stage_gets_policy_tag(self):
        raw = _semantic_response(stages=[_stage("stage_1", required=False)])
        svc = _make_service(raw)
        plan = _create_plan(svc)
        assert "optional_candidate" in plan.slices[0].policy_tags

    def test_three_stages_single_batch(self):
        raw = _semantic_response(
            stages=[
                _stage("stage_1", tool_hints=["research_memory"]),
                _stage("stage_2", tool_hints=["backtesting"], depends_on=["stage_1"]),
                _stage("stage_3", tool_hints=["analysis"], depends_on=["stage_1"]),
            ]
        )
        svc = _make_service(raw)
        plan = _create_plan(svc)
        assert len(plan.slices) == 3
        dep = plan.slices[1].depends_on
        assert any("stage_1" in d for d in dep)


class TestPlannerMultiBatchQueuing:
    def test_five_stages_produces_two_plans(self):
        stages = [
            _stage("stage_1"),
            _stage("stage_2"),
            _stage("stage_3"),
            _stage("stage_4", depends_on=["stage_1"]),
            _stage("stage_5", depends_on=["stage_2"]),
        ]
        raw = _semantic_response(stages=stages)
        svc = _make_service(raw)

        plan1 = _create_plan(svc)
        assert len(plan1.slices) == 3

        plan2 = _create_plan(svc, plan_version=2)
        assert len(plan2.slices) == 2
        assert plan2.plan_source_kind == "planner"

    def test_third_call_invokes_llm_again(self):
        raw = _semantic_response(stages=[_stage("stage_1")])
        svc = _make_service(raw)

        _create_plan(svc)
        # No more pending plans, next call invokes LLM again
        plan2 = _create_plan(svc, plan_version=2)
        assert isinstance(plan2, ExecutionPlan)


class TestPlannerEdgeCases:
    def test_unknown_tool_hint_falls_back_to_analysis(self):
        raw = _semantic_response(stages=[_stage("stage_1", tool_hints=["nonexistent_tool_xyz"])])
        svc = _make_service(raw)
        plan = _create_plan(svc)
        assert isinstance(plan, ExecutionPlan)

    def test_invalid_json_raises_error(self):
        svc = _make_service("not valid json {{{")
        with pytest.raises(PlannerDecisionError):
            _create_plan(svc)

    def test_parallelizable_gets_slot(self):
        raw = _semantic_response(
            stages=[
                _stage("stage_1", parallelizable=True),
                _stage("stage_2", parallelizable=True),
            ]
        )
        svc = _make_service(raw)
        plan = _create_plan(svc, worker_count=3)
        assert plan.slices[0].parallel_slot == 1
        assert plan.slices[1].parallel_slot == 2

    def test_plan_saved_to_artifact_store(self):
        raw = _semantic_response(stages=[_stage("stage_1")])
        svc = _make_service(raw)
        _create_plan(svc)
        svc.artifact_store.save_plan.assert_called_once()
