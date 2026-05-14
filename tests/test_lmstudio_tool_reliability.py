"""
Tests for LMStudio tool-call reliability improvements:
- tool_choice="required" on first turn
- Nudge retry on zero tool calls
- Temperature cap change
- Model crash detection and fast-fail
- Catalog-only slice readiness fix
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool
from app.services.direct_execution.lmstudio_tool_loop import LmStudioToolLoop
from app.services.direct_execution.temperature_config import get_adaptive_temperature
from app.services.direct_execution.slice_readiness import required_output_facts_for_slice
from app.execution_models import ExecutionPlan, PlanSlice
from tests.mcp_catalog_fixtures import make_catalog_snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_loop(
    *,
    zero_tool_retries: int = 2,
    first_turn_tool_choice: str = "required",
    allowed_tools: set[str] | None = None,
    runtime_profile: str = "",
    required_output_facts: list[str] | None = None,
    success_criteria: list[str] | None = None,
    slice_title: str = "test-slice",
    baseline_bootstrap: dict | None = None,
) -> LmStudioToolLoop:
    adapter = MagicMock()
    adapter.temperature = 0.7
    adapter.max_tokens = 512
    adapter.model = "test-model"
    adapter.reasoning_effort = ""
    adapter.extra_body = {}
    adapter.api_key = ""
    adapter.base_url = "http://localhost:9999"

    mcp_client = AsyncMock()
    mcp_client.close = AsyncMock()
    mcp_client.call_tool = AsyncMock(return_value={"status": "ok", "data": {}})

    incident_store = MagicMock()
    incident_store.record = MagicMock()

    loop = LmStudioToolLoop(
        adapter=adapter,
        mcp_client=mcp_client,
        incident_store=incident_store,
        catalog_snapshot=make_catalog_snapshot(),
        allowed_tools=allowed_tools or {"research_map"},
        slice_title=slice_title,
        success_criteria=list(success_criteria or []),
        required_output_facts=list(required_output_facts or []),
        runtime_profile=runtime_profile,
        max_tool_calls=24,
        max_expensive_tool_calls=6,
        safe_exclude_tools=set(),
        first_action_timeout_seconds=10,
        stalled_action_timeout_seconds=10,
        zero_tool_retries=zero_tool_retries,
        first_turn_tool_choice=first_turn_tool_choice,
        baseline_bootstrap=baseline_bootstrap or {
            "baseline_snapshot_id": "active-signal-v1",
            "baseline_version": 1,
            "symbol": "BTCUSDT",
            "anchor_timeframe": "1h",
            "execution_timeframe": "5m",
        },
    )
    loop._connection_pool.warm_up = MagicMock(return_value=True)
    loop._connection_pool.health_check = MagicMock(return_value=True)
    return loop


def _tool_call_response(tool_name: str, call_id: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ]
    }


def _text_only_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content, "tool_calls": []}}]}


def _make_slice(allowed_tools: list[str]) -> PlanSlice:
    runtime_profile = "catalog_contract_probe" if allowed_tools == ["features_catalog"] else "generic_read"
    required_output_facts = ["features_catalog.scopes"] if runtime_profile == "catalog_contract_probe" else []
    finalization_mode = "fact_based" if runtime_profile == "catalog_contract_probe" else "generic_salvage"
    return PlanSlice(
        slice_id="test_slice",
        title="test",
        hypothesis="test",
        objective="test",
        success_criteria=[],
        allowed_tools=allowed_tools,
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=24,
        max_expensive_calls=0,
        parallel_slot=1,
        runtime_profile=runtime_profile,
        required_output_facts=required_output_facts,
        finalization_mode=finalization_mode,
    )


def _make_plan(slices: list[PlanSlice] | None = None) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="test_plan",
        goal="test",
        baseline_ref={},
        global_constraints=[],
        slices=slices or [],
    )


# ---------------------------------------------------------------------------
# CHANGE 1: tool_choice="required" on first turn
# ---------------------------------------------------------------------------

class TestToolChoiceRequired:
    def test_first_turn_uses_required_tool_choice(self) -> None:
        """First chat call should pass tool_choice='required'."""
        loop = _make_tool_loop(first_turn_tool_choice="required")
        call_kwargs_list: list[dict] = []

        call_idx = 0

        def fake_sync_seq(**kwargs):
            nonlocal call_idx
            call_kwargs_list.append(kwargs)
            call_idx += 1
            if call_idx == 1:
                return _tool_call_response("research_map", "c1", {"action": "inspect", "project_id": "proj_1"})
            return _text_only_response(
                '```json\n{"type":"final_report","summary":"ok","verdict":"WATCHLIST",'
                '"findings":[],"facts":{},"evidence_refs":["transcript:1:research_map"],"confidence":0.7}\n```'
            )

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=fake_sync_seq):
                await loop.invoke(
                    prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1"
                )

        asyncio.run(_run())
        assert len(call_kwargs_list) >= 1
        # First call uses "required"
        assert call_kwargs_list[0]["tool_choice"] == "required"
        # After a tool has been called (tool_call_count > 0), subsequent calls use "auto"
        if len(call_kwargs_list) > 1:
            for kw in call_kwargs_list[1:]:
                assert kw["tool_choice"] == "auto"

    def test_required_fallback_to_auto_on_error(self) -> None:
        """If tool_choice=required causes API error, retry with auto."""
        loop = _make_tool_loop(first_turn_tool_choice="required")
        call_kwargs_list: list[dict] = []
        call_idx = 0

        def fake_sync(**kwargs):
            nonlocal call_idx
            call_kwargs_list.append(kwargs)
            call_idx += 1
            if call_idx == 1:
                # Simulate API error for "required"
                return {"error": "tool_choice 'required' is not supported"}
            if call_idx == 2:
                return _tool_call_response("research_map", "c1", {"action": "inspect", "project_id": "proj_1"})
            return _text_only_response(
                '```json\n{"type":"final_report","summary":"ok","verdict":"WATCHLIST",'
                '"findings":[],"facts":{},"evidence_refs":["x"],"confidence":0.7}\n```'
            )

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=fake_sync):
                result = await loop.invoke(
                    prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1"
                )
            return result

        asyncio.run(_run())
        assert call_kwargs_list[0]["tool_choice"] == "required"
        assert call_kwargs_list[1]["tool_choice"] == "auto"

    def test_auto_mode_skips_required(self) -> None:
        """When first_turn_tool_choice='auto', never use 'required'."""
        loop = _make_tool_loop(first_turn_tool_choice="auto")
        call_kwargs_list: list[dict] = []

        def fake_sync(**kwargs):
            call_kwargs_list.append(kwargs)
            return _text_only_response("some text")

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=fake_sync):
                await loop.invoke(
                    prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1"
                )

        asyncio.run(_run())
        assert all(kw["tool_choice"] == "auto" for kw in call_kwargs_list)


class TestMixedDomainLateContextLoop:
    def test_late_research_memory_loop_auto_finalizes_after_domain_evidence(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_memory", "features_catalog", "features_analytics"},
            runtime_profile="generic_mutation",
            success_criteria=["Only plausible hypotheses remain."],
            slice_title="quick plausibility filter",
        )
        loop.runtime_profile = "generic_mutation"
        loop.finalization_mode = "none"
        loop.required_output_facts = []

        responses = iter(
            [
                _tool_call_response("research_memory", "c1", {"action": "search", "query": "ctx-1"}),
                _tool_call_response("research_memory", "c2", {"action": "search", "query": "ctx-2"}),
                _tool_call_response("features_catalog", "c3", {"scope": "available"}),
                _tool_call_response(
                    "features_analytics",
                    "c4",
                    {"action": "analytics", "symbol": "BTCUSDT", "anchor_timeframe": "1h"},
                ),
                _tool_call_response("research_memory", "c5", {"action": "search", "query": "loop-1"}),
                _tool_call_response("research_memory", "c6", {"action": "search", "query": "loop-2"}),
                _tool_call_response("research_memory", "c7", {"action": "search", "query": "loop-3"}),
                _tool_call_response("research_memory", "c8", {"action": "search", "query": "loop-4"}),
            ]
        )

        def _tool_payload(tool_name: str) -> dict:
            summary = "Loaded research search results." if tool_name == "research_memory" else "Loaded live domain evidence."
            return {
                "structuredContent": {
                    "status": "ok",
                    "summary": summary,
                    "data": {"project_id": "proj_1"},
                }
            }

        async def _call_tool(tool_name: str, arguments: dict) -> dict:
            return _tool_payload(tool_name)

        async def _run():
            with (
                patch.object(loop, "_chat_sync", side_effect=lambda **_: next(responses)),
                patch(
                    "app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call",
                    side_effect=lambda tool_name, arguments, catalog_snapshot=None: SimpleNamespace(
                        arguments=arguments,
                        local_payload=None,
                        repair_notes=[],
                        charge_budget=True,
                    ),
                ),
            ):
                loop.mcp_client.call_tool = AsyncMock(side_effect=_call_tool)
                return await loop.invoke(
                    prompt="test",
                    timeout_seconds=30,
                    plan_id="p1",
                    slice_id="s1",
                )

        result = asyncio.run(_run())

        assert '"type": "final_report"' in result.response.raw_output
        assert "direct.auto_finalized_from_late_context_loop_salvage" in result.response.raw_output
        assert result.tool_call_count == 0


class TestFeatureContractRuntime:
    def test_feature_contract_exploration_auto_finalizes_after_distinct_live_probes(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_memory", "events", "datasets"},
            runtime_profile="generic_mutation",
            success_criteria=[
                "Leakage verified for every contract.",
                "Event alignment verified for event hypotheses.",
            ],
            slice_title="Data contract and feature contract exploration",
        )
        loop.slice_objective = "Explore managed features, events, and datasets for leakage-safe feature contracts."
        loop.policy_tags = ["data_readiness", "feature_contract"]
        loop.finalization_mode = "none"
        loop.required_output_facts = []

        responses = iter(
            [
                _tool_call_response("research_memory", "c1", {"action": "search", "query": "ctx"}),
                _tool_call_response("events", "c2", {"view": "catalog", "family": "funding"}),
                _tool_call_response("datasets", "c3", {"view": "catalog"}),
            ]
        )

        async def _call_tool(tool_name: str, arguments: dict) -> dict:
            return {
                "structuredContent": {
                    "status": "ok",
                    "summary": f"{tool_name} ok",
                    "data": {"project_id": "proj_1"},
                },
            }

        async def _run():
            with (
                patch.object(loop, "_chat_sync", side_effect=lambda **_: next(responses)),
                patch(
                    "app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call",
                    side_effect=lambda tool_name, arguments, catalog_snapshot=None: SimpleNamespace(
                        arguments=arguments,
                        local_payload=None,
                        repair_notes=[],
                        charge_budget=True,
                    ),
                ),
            ):
                loop.mcp_client.call_tool = AsyncMock(side_effect=_call_tool)
                return await loop.invoke(
                    prompt="test",
                    timeout_seconds=30,
                    plan_id="p1",
                    slice_id="s1",
                )

        result = asyncio.run(_run())

        assert '"type": "final_report"' in result.response.raw_output
        assert '"verdict": "COMPLETE"' in result.response.raw_output
        assert "feature_contract.exploration_completed" in result.response.raw_output
        assert result.tool_call_count == 3

    def test_incomplete_feature_contract_exploration_does_not_stall_salvage(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_memory", "features_catalog", "events", "datasets"},
            runtime_profile="generic_mutation",
            success_criteria=[
                "Leakage verified for every contract.",
                "Event alignment verified for event hypotheses.",
            ],
            slice_title="Data contract and feature contract exploration",
        )
        loop.slice_objective = "Explore managed features, events, and datasets for leakage-safe feature contracts."
        loop.policy_tags = ["data_readiness", "feature_contract"]

        transcript = [
            {
                "kind": "tool_result",
                "tool": "research_memory",
                "arguments": {"action": "search", "query": "ctx"},
                "payload": {"ok": True, "payload": {"structuredContent": {"status": "ok", "data": {"project_id": "proj_1"}}}},
            },
            {
                "kind": "tool_result",
                "tool": "features_catalog",
                "arguments": {"scope": "timeframe", "timeframe": "1h"},
                "payload": {"ok": True, "payload": {"structuredContent": {"status": "ok", "data": {"project_id": "proj_1"}}}},
            },
        ]

        assert loop._maybe_auto_finalize_after_stall(transcript) is None

    def test_feature_contract_identifier_misuse_auto_finalizes_after_probe_evidence(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_memory", "features_dataset", "features_custom", "features_analytics"},
            runtime_profile="generic_mutation",
            success_criteria=[
                "Custom features validated and published.",
                "Leakage verified for every contract.",
            ],
            slice_title="Data contract and feature contract construction",
        )
        loop.slice_objective = "Define precise feature contracts and leakage checks."
        loop.policy_tags = ["data_readiness", "feature_contract"]
        loop.finalization_mode = "none"
        loop.required_output_facts = []

        responses = iter(
            [
                _tool_call_response("research_memory", "c1", {"action": "search", "query": "ctx"}),
                _tool_call_response(
                    "features_dataset",
                    "c2",
                    {"action": "inspect", "view": "columns", "symbol": "BTCUSDT", "timeframe": "1h"},
                ),
                _tool_call_response("features_custom", "c3", {"action": "inspect", "view": "contract"}),
                _tool_call_response(
                    "features_analytics",
                    "c4",
                    {"action": "analytics", "symbol": "BTCUSDT", "anchor_timeframe": "1h"},
                ),
            ]
        )

        async def _call_tool(tool_name: str, arguments: dict) -> dict:
            if tool_name == "research_memory":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Loaded research search results.",
                        "data": {"project_id": "proj_1"},
                    },
                }
            if tool_name == "features_dataset":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Columns loaded.",
                        "data": {"columns": ["atr_1", "rsi_1", "cf_vol_term_spread"]},
                    },
                }
            if tool_name == "features_custom":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Custom feature contract loaded.",
                        "data": {"view": "contract"},
                    },
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=lambda **_: next(responses)):
                loop.mcp_client.call_tool = AsyncMock(side_effect=_call_tool)
                return await loop.invoke(
                    prompt="test",
                    timeout_seconds=30,
                    plan_id="p1",
                    slice_id="s1",
                )

        result = asyncio.run(_run())

        assert '"type": "final_report"' in result.response.raw_output
        assert "feature_contract.explicit_identifier_required" in result.response.raw_output
        assert '"WATCHLIST"' in result.response.raw_output

    def test_feature_contract_unknown_custom_feature_name_auto_finalizes_after_live_list(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_memory", "features_dataset", "features_custom", "features_analytics"},
            runtime_profile="generic_mutation",
            success_criteria=[
                "Custom features validated and published.",
                "Leakage verified for every contract.",
            ],
            slice_title="Data contract and feature contract construction",
        )
        loop.slice_objective = "Define precise feature contracts and leakage checks."
        loop.policy_tags = ["data_readiness", "feature_contract"]
        loop.finalization_mode = "none"
        loop.required_output_facts = []

        responses = iter(
            [
                _tool_call_response("research_memory", "c1", {"action": "search", "query": "ctx"}),
                _tool_call_response(
                    "features_dataset",
                    "c2",
                    {"action": "inspect", "view": "columns", "symbol": "BTCUSDT", "timeframe": "1h"},
                ),
                _tool_call_response("features_custom", "c3", {"action": "inspect", "view": "contract"}),
                _tool_call_response("features_custom", "c4", {"action": "inspect", "view": "list"}),
                _tool_call_response(
                    "features_custom",
                    "c5",
                    {"action": "inspect", "view": "detail", "name": "cf_fake_signal"},
                ),
            ]
        )

        async def _call_tool(tool_name: str, arguments: dict) -> dict:
            if tool_name == "research_memory":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Loaded research search results.",
                        "data": {"project_id": "proj_1"},
                    },
                }
            if tool_name == "features_dataset":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Columns loaded.",
                        "data": {"columns": ["atr_1", "rsi_1", "cf_vol_term_spread"]},
                    },
                }
            if tool_name == "features_custom" and arguments.get("view") == "contract":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Custom feature contract loaded.",
                        "data": {"view": "contract"},
                    },
                }
            if tool_name == "features_custom" and arguments.get("view") == "list":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Custom feature list loaded.",
                        "data": {"features": [{"name": "cf_vol_term_spread"}]},
                    },
                }
            if tool_name == "features_custom" and arguments.get("view") == "detail":
                return {
                    "structuredContent": {
                        "status": "error",
                        "message": "Unknown custom feature: cf_fake_signal",
                        "data": {},
                    },
                }
            raise AssertionError(f"unexpected tool call: {tool_name} {arguments}")

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=lambda **_: next(responses)):
                loop.mcp_client.call_tool = AsyncMock(side_effect=_call_tool)
                return await loop.invoke(
                    prompt="test",
                    timeout_seconds=30,
                    plan_id="p1",
                    slice_id="s1",
                )

        result = asyncio.run(_run())

        assert '"type": "final_report"' in result.response.raw_output
        assert "feature_contract.invalid_custom_feature_name" in result.response.raw_output
        assert "cf_fake_signal" in result.response.raw_output

    def test_feature_contract_construction_auto_finalizes_before_expensive_budget_salvage(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_memory", "features_dataset", "features_custom", "features_analytics"},
            runtime_profile="generic_mutation",
            success_criteria=[
                "Clean data and feature contract exists for each shortlisted family.",
                "Leakage verified for every contract.",
            ],
            slice_title="Data contract and feature contract construction",
        )
        loop.slice_objective = "Fix exact feature contracts and leakage-safe construction paths."
        loop.policy_tags = ["data_readiness", "feature_contract"]
        loop.finalization_mode = "none"
        loop.required_output_facts = []

        responses = iter(
            [
                _tool_call_response("research_memory", "c1", {"action": "search", "query": "ctx"}),
                _tool_call_response(
                    "features_dataset",
                    "c2",
                    {"action": "inspect", "view": "columns", "symbol": "BTCUSDT", "timeframe": "1h"},
                ),
                _tool_call_response("features_custom", "c3", {"action": "inspect", "view": "list"}),
                _tool_call_response("features_custom", "c4", {"action": "inspect", "view": "contract"}),
                _tool_call_response(
                    "features_analytics",
                    "c5",
                    {"action": "analytics", "symbol": "BTCUSDT", "feature_name": "rsi_1", "anchor_timeframe": "1h"},
                ),
                _tool_call_response(
                    "features_analytics",
                    "c6",
                    {"action": "heatmap", "symbol": "BTCUSDT", "feature_name": "rsi_1", "anchor_timeframe": "1h"},
                ),
            ]
        )

        async def _call_tool(tool_name: str, arguments: dict) -> dict:
            if tool_name == "research_memory":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Loaded research shortlist context.",
                        "data": {"project_id": "proj_1"},
                    },
                }
            if tool_name == "features_dataset":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Columns loaded.",
                        "data": {"columns": ["atr_1", "rsi_1", "cf_vol_term_spread"]},
                    },
                }
            if tool_name == "features_custom" and arguments.get("view") == "list":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Custom feature list loaded.",
                        "data": {"features": [{"name": "cf_vol_term_spread"}]},
                    },
                }
            if tool_name == "features_custom" and arguments.get("view") == "contract":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Custom feature contract loaded.",
                        "data": {"view": "contract"},
                    },
                }
            if tool_name == "features_analytics":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": f"{arguments['action']} ok",
                        "data": {"feature_name": arguments["feature_name"]},
                    },
                }
            raise AssertionError(f"unexpected tool call: {tool_name} {arguments}")

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=lambda **_: next(responses)):
                loop.mcp_client.call_tool = AsyncMock(side_effect=_call_tool)
                return await loop.invoke(
                    prompt="test",
                    timeout_seconds=30,
                    plan_id="p1",
                    slice_id="s1",
                )

        result = asyncio.run(_run())

        assert '"type": "final_report"' in result.response.raw_output
        assert "feature_contract.construction_completed" in result.response.raw_output
        assert "direct.auto_finalized_from_expensive_budget_salvage" not in result.response.raw_output
        assert '"COMPLETE"' in result.response.raw_output

    def test_feature_profitability_filter_auto_finalizes_after_catalog_and_analytics_probes(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_memory", "features_catalog", "features_analytics"},
            runtime_profile="generic_mutation",
            success_criteria=["Only hypotheses with preliminary plausibility and orthogonality remain."],
            slice_title="Fast filter via signal plausibility and feature profitability",
        )
        loop.slice_objective = "Screen hypotheses before backtests using live feature profitability evidence."
        loop.policy_tags = ["cheap_first", "filter"]
        loop.finalization_mode = "none"
        loop.required_output_facts = []

        responses = iter(
            [
                _tool_call_response("research_memory", "c1", {"action": "search", "query": "ctx"}),
                _tool_call_response("features_catalog", "c2", {"scope": "timeframe", "timeframe": "1h"}),
                _tool_call_response(
                    "features_analytics",
                    "c3",
                    {"action": "analytics", "symbol": "BTCUSDT", "feature_name": "rsi_1", "anchor_timeframe": "1h"},
                ),
                _tool_call_response(
                    "features_analytics",
                    "c4",
                    {"action": "portability", "symbol": "BTCUSDT", "feature_name": "rsi_1", "anchor_timeframe": "1h"},
                ),
            ]
        )

        async def _call_tool(tool_name: str, arguments: dict) -> dict:
            if tool_name == "research_memory":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Loaded shortlist context.",
                        "data": {"project_id": "proj_1"},
                    },
                }
            if tool_name == "features_catalog":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Catalog loaded.",
                        "data": {"columns": ["rsi_1", "atr_1"]},
                    },
                }
            if tool_name == "features_analytics":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": f"{arguments['action']} ok",
                        "data": {"feature_name": arguments["feature_name"]},
                    },
                }
            raise AssertionError(f"unexpected tool call: {tool_name} {arguments}")

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=lambda **_: next(responses)):
                loop.mcp_client.call_tool = AsyncMock(side_effect=_call_tool)
                return await loop.invoke(
                    prompt="test",
                    timeout_seconds=30,
                    plan_id="p1",
                    slice_id="s1",
                )

        result = asyncio.run(_run())

        assert '"type": "final_report"' in result.response.raw_output
        assert "direct.auto_finalized_from_generic_salvage" not in result.response.raw_output
        assert '"COMPLETE"' in result.response.raw_output
        assert (
            "feature_filter.preliminary_screen_completed" in result.response.raw_output
            or "Catalog-contract probe completed from live MCP tool evidence." in result.response.raw_output
        )

    def test_filter_slice_reuses_recent_feature_name_after_explicit_selection(self) -> None:
        from app.services.direct_execution.feature_contract_runtime import (
            repair_feature_analytics_identifier_from_transcript,
        )

        transcript = [
            {
                "kind": "tool_result",
                "tool": "features_analytics",
                "arguments": {
                    "action": "analytics",
                    "symbol": "BTCUSDT",
                    "anchor_timeframe": "1h",
                    "column_name": "signal_strength",
                },
                "payload": {
                    "ok": True,
                    "payload": {
                        "structuredContent": {
                            "status": "error",
                            "message": "Stored analytics are not ready for feature 'signal_strength' on BTCUSDT 1h.",
                            "error": {"details": {"feature_name": "signal_strength"}},
                        }
                    },
                },
            },
            {
                "kind": "tool_result",
                "tool": "features_analytics",
                "arguments": {
                    "action": "backfill",
                    "symbol": "BTCUSDT",
                    "anchor_timeframe": "1h",
                },
                "payload": {
                    "ok": True,
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "summary": "Backfill queued.",
                        }
                    },
                },
            },
        ]

        repaired, notes = repair_feature_analytics_identifier_from_transcript(
            transcript=transcript,
            tool_name="features_analytics",
            arguments={"action": "heatmap", "symbol": "BTCUSDT", "anchor_timeframe": "1h"},
            slice_title="Fast filter via signal plausibility and feature profitability",
            slice_objective="Screen hypotheses before backtests using live feature profitability evidence.",
            success_criteria=["Only hypotheses with preliminary plausibility and orthogonality remain."],
            policy_tags=["cheap_first", "filter"],
        )

        assert repaired["feature_name"] == "signal_strength"
        assert "reused_recent_feature_name:signal_strength" in notes

    def test_filter_slice_selects_live_catalog_feature_when_identifier_missing(self) -> None:
        from app.services.direct_execution.feature_contract_runtime import (
            repair_feature_analytics_identifier_from_transcript,
        )

        transcript = [
            {
                "kind": "tool_result",
                "tool": "features_catalog",
                "arguments": {"scope": "available"},
                "payload": {
                    "ok": True,
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {"columns": ["cf_eth_lead", "atr_1", "rsi_1", "cl_1h"]},
                        }
                    },
                },
            }
        ]

        repaired, notes = repair_feature_analytics_identifier_from_transcript(
            transcript=transcript,
            tool_name="features_analytics",
            arguments={"action": "analytics", "symbol": "BTCUSDT", "anchor_timeframe": "1h"},
            slice_title="Fast filter via signal plausibility and feature profitability",
            slice_objective="Screen hypotheses before backtests using live feature profitability evidence.",
            success_criteria=["Only hypotheses with preliminary plausibility and orthogonality remain."],
            policy_tags=["cheap_first", "filter"],
        )

        assert repaired["feature_name"] == "rsi_1"
        assert "reused_recent_feature_name:rsi_1" in notes


# ---------------------------------------------------------------------------
# CHANGE 1 (connection level): tool_choice param in LMStudioConnectionPool
# ---------------------------------------------------------------------------

class TestConnectionPoolToolChoice:
    def test_tool_choice_passed_to_body(self) -> None:
        """Verify tool_choice parameter is included in the request body."""
        pool = LMStudioConnectionPool(
            base_url="http://localhost:9999",
            timeout=5,
        )
        captured_body: list[dict] = []

        def fake_urlopen(method, url, body=None, headers=None, timeout=None, retries=None):
            captured_body.append(json.loads(body))
            resp = MagicMock()
            resp.status = 200
            resp.data = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
            return resp

        pool._pool.urlopen = fake_urlopen
        pool.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "test", "parameters": {}}}],
            tool_choice="required",
        )
        assert captured_body[0]["tool_choice"] == "required"

    def test_tool_choice_defaults_to_auto(self) -> None:
        """Default tool_choice should be 'auto' when tools are present."""
        pool = LMStudioConnectionPool(
            base_url="http://localhost:9999",
            timeout=5,
        )
        captured_body: list[dict] = []

        def fake_urlopen(method, url, body=None, headers=None, timeout=None, retries=None):
            captured_body.append(json.loads(body))
            resp = MagicMock()
            resp.status = 200
            resp.data = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
            return resp

        pool._pool.urlopen = fake_urlopen
        pool.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "test", "parameters": {}}}],
        )
        assert captured_body[0]["tool_choice"] == "auto"

    def test_reasoning_effort_none_is_sent_explicitly(self) -> None:
        """Direct LM Studio path must send reasoning_effort='none' explicitly."""
        pool = LMStudioConnectionPool(
            base_url="http://localhost:9999",
            timeout=5,
        )
        captured_body: list[dict] = []

        def fake_urlopen(method, url, body=None, headers=None, timeout=None, retries=None):
            captured_body.append(json.loads(body))
            resp = MagicMock()
            resp.status = 200
            resp.data = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
            return resp

        pool._pool.urlopen = fake_urlopen
        pool.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "test", "parameters": {}}}],
            tool_choice="required",
            reasoning_effort="none",
        )
        assert captured_body[0]["reasoning_effort"] == "none"

    def test_reasoning_effort_off_normalizes_to_none(self) -> None:
        pool = LMStudioConnectionPool(
            base_url="http://localhost:9999",
            timeout=5,
        )
        captured_body: list[dict] = []

        def fake_urlopen(method, url, body=None, headers=None, timeout=None, retries=None):
            captured_body.append(json.loads(body))
            resp = MagicMock()
            resp.status = 200
            resp.data = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
            return resp

        pool._pool.urlopen = fake_urlopen
        pool.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "test", "parameters": {}}}],
            reasoning_effort="off",
        )
        assert captured_body[0]["reasoning_effort"] == "none"


# ---------------------------------------------------------------------------
# CHANGE 2: Nudge retry on zero tool calls
# ---------------------------------------------------------------------------

class TestNudgeRetry:
    def test_nudge_retry_sends_nudge_and_retries(self) -> None:
        """Model returns text first, gets nudged, then calls tool."""
        loop = _make_tool_loop(zero_tool_retries=2, first_turn_tool_choice="auto")
        call_idx = 0
        messages_seen: list[list[dict]] = []

        def fake_sync(**kwargs):
            nonlocal call_idx
            messages_seen.append(list(kwargs.get("messages", [])))
            call_idx += 1
            if call_idx == 1:
                # First: return text without tool calls
                return _text_only_response("I'll analyze the data...")
            if call_idx == 2:
                # After nudge: call tool
                return _tool_call_response("research_map", "c1", {"action": "inspect"})
            # Final: return report
            return _text_only_response(
                '```json\n{"type":"final_report","summary":"done","verdict":"WATCHLIST",'
                '"findings":[],"facts":{},"evidence_refs":["transcript:1:research_map"],"confidence":0.7}\n```'
            )

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=fake_sync):
                result = await loop.invoke(
                    prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1"
                )
            return result

        result = asyncio.run(_run())
        # After nudge, model should have been called again
        assert call_idx >= 2
        # Check that nudge message was sent (look for "user" role with nudge text)
        nudge_found = False
        for msgs in messages_seen[1:]:
            for m in msgs:
                if m.get("role") == "user" and "MUST call" in m.get("content", ""):
                    nudge_found = True
        assert nudge_found
        # Transcript should contain nudge entry
        nudge_entries = [
            e for e in result.transcript if e.get("kind") == "zero_tool_nudge"
        ]
        assert len(nudge_entries) == 1
        assert nudge_entries[0]["attempt"] == 1

    def test_nudge_exhaustion_returns_without_tool_calls(self) -> None:
        """After exhausting nudge retries, return text-only result."""
        loop = _make_tool_loop(zero_tool_retries=1, first_turn_tool_choice="auto")

        def fake_sync(**kwargs):
            return _text_only_response("I can't call tools")

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=fake_sync):
                result = await loop.invoke(
                    prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1"
                )
            return result

        result = asyncio.run(_run())
        assert result.tool_call_count == 0
        # Should have 1 nudge entry in transcript
        nudge_entries = [
            e for e in result.transcript if e.get("kind") == "zero_tool_nudge"
        ]
        assert len(nudge_entries) == 1

    def test_zero_retries_returns_immediately(self) -> None:
        """With zero_tool_retries=0, no nudging happens."""
        loop = _make_tool_loop(zero_tool_retries=0, first_turn_tool_choice="auto")
        call_count = 0

        def fake_sync(**kwargs):
            nonlocal call_count
            call_count += 1
            return _text_only_response("text only")

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=fake_sync):
                result = await loop.invoke(
                    prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1"
                )
            return result

        result = asyncio.run(_run())
        # Only one call — no retries
        assert call_count == 1
        assert result.tool_call_count == 0
        nudge_entries = [
            e for e in result.transcript if e.get("kind") == "zero_tool_nudge"
        ]
        assert len(nudge_entries) == 0

    def test_backtests_zero_tool_nudge_recommends_backtests_plan(self) -> None:
        loop = _make_tool_loop(
            zero_tool_retries=1,
            first_turn_tool_choice="auto",
            allowed_tools={"backtests_plan", "backtests_runs", "research_memory"},
            runtime_profile="generic_mutation",
            slice_title="Standalone backtests новых сигналов",
            success_criteria=["Есть shortlist standalone-кандидатов, которые не пусты сами по себе."],
        )
        messages_seen: list[list[dict]] = []
        call_idx = 0

        def fake_sync(**kwargs):
            nonlocal call_idx
            messages_seen.append(list(kwargs.get("messages", [])))
            call_idx += 1
            return _text_only_response("still thinking")

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=fake_sync):
                return await loop.invoke(
                    prompt="test",
                    timeout_seconds=30,
                    plan_id="p1",
                    slice_id="s1",
                )

        result = asyncio.run(_run())
        assert call_idx == 2
        nudge_messages = [
            m.get("content", "")
            for msgs in messages_seen[1:]
            for m in msgs
            if m.get("role") == "user"
        ]
        assert any("backtests_plan(" in msg for msg in nudge_messages)
        assert any("active-signal-v1" in msg for msg in nudge_messages)
        nudge_entries = [
            e for e in result.transcript if e.get("kind") == "zero_tool_nudge"
        ]
        assert len(nudge_entries) == 1


# ---------------------------------------------------------------------------
# CHANGE 3: Temperature cap
# ---------------------------------------------------------------------------

class TestTemperatureCap:
    def test_lmstudio_temperature_capped_at_055(self) -> None:
        assert get_adaptive_temperature("lmstudio", 0.7) == 0.55

    def test_lmstudio_temperature_below_cap_unchanged(self) -> None:
        assert get_adaptive_temperature("lmstudio", 0.3) == 0.3

    def test_lmstudio_temperature_floor_at_01(self) -> None:
        assert get_adaptive_temperature("lmstudio", 0.05) == 0.1

    def test_non_lmstudio_temperature_unchanged(self) -> None:
        assert get_adaptive_temperature("qwen_cli", 0.7) == 0.7
        assert get_adaptive_temperature("claude_cli", 0.9) == 0.9


# ---------------------------------------------------------------------------
# CHANGE 5: Model crash detection
# ---------------------------------------------------------------------------

class TestCrashDetection:
    def test_connection_pool_detects_model_crash(self) -> None:
        """HTTP 400 with 'model has crashed' should return crash error immediately."""
        pool = LMStudioConnectionPool(
            base_url="http://localhost:9999",
            timeout=5,
        )

        def fake_urlopen(method, url, body=None, headers=None, timeout=None, retries=None):
            resp = MagicMock()
            resp.status = 400
            resp.data = b'{"error":"The model has crashed without additional information. (Exit code: null)"}'
            return resp

        pool._pool.urlopen = fake_urlopen
        result = pool.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "lmstudio_model_crash" in result.get("error", "")

    def test_connection_pool_exit_code_crash(self) -> None:
        """HTTP 400 with 'exit code: 1' should also be detected as crash."""
        pool = LMStudioConnectionPool(
            base_url="http://localhost:9999",
            timeout=5,
        )

        def fake_urlopen(method, url, body=None, headers=None, timeout=None, retries=None):
            resp = MagicMock()
            resp.status = 400
            resp.data = b'{"error":"Model process exited with Exit code: 1"}'
            return resp

        pool._pool.urlopen = fake_urlopen
        result = pool.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "lmstudio_model_crash" in result.get("error", "")

    def test_non_crash_400_not_treated_as_crash(self) -> None:
        """Regular HTTP 400 errors should NOT be treated as crashes."""
        pool = LMStudioConnectionPool(
            base_url="http://localhost:9999",
            timeout=5,
        )

        def fake_urlopen(method, url, body=None, headers=None, timeout=None, retries=None):
            resp = MagicMock()
            resp.status = 400
            resp.data = b'{"error":"Invalid request format"}'
            return resp

        pool._pool.urlopen = fake_urlopen
        result = pool.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "lmstudio_model_crash" not in result.get("error", "")
        assert "HTTP 400" in result.get("error", "")

    def test_crash_skips_retry_in_chat(self) -> None:
        """Model crash should not be retried in _chat method."""
        loop = _make_tool_loop(first_turn_tool_choice="auto")
        call_count = 0

        def fake_sync(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"error": "lmstudio_model_crash: The model has crashed"}

        async def _run():
            with patch.object(loop, "_chat_sync", side_effect=fake_sync):
                return await loop.invoke(
                    prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1"
                )

        asyncio.run(_run())
        # Should only call once — no retries for crash
        assert call_count == 1

    def test_health_check_detects_no_models_loaded(self) -> None:
        """Health check should fail if no models are loaded (post-crash)."""
        pool = LMStudioConnectionPool(
            base_url="http://localhost:9999",
            timeout=5,
        )

        def fake_urlopen(method, url, body=None, headers=None, timeout=None, retries=None):
            resp = MagicMock()
            resp.status = 200
            resp.data = json.dumps({"data": []}).encode()
            return resp

        pool._pool.urlopen = fake_urlopen
        assert pool.health_check() is False

    def test_health_check_passes_with_model_loaded(self) -> None:
        """Health check should pass when at least one model is loaded."""
        pool = LMStudioConnectionPool(
            base_url="http://localhost:9999",
            timeout=5,
        )

        def fake_urlopen(method, url, body=None, headers=None, timeout=None, retries=None):
            resp = MagicMock()
            resp.status = 200
            resp.data = json.dumps({"data": [{"id": "qwen-35b"}]}).encode()
            return resp

        pool._pool.urlopen = fake_urlopen
        assert pool.health_check() is True


# ---------------------------------------------------------------------------
# CHANGE 6: Catalog-only slice readiness fix
# ---------------------------------------------------------------------------

class TestCatalogSliceReadiness:
    def test_catalog_only_requires_scopes_not_research(self) -> None:
        """Catalog-only slices should require features_catalog.scopes, not research facts."""
        plan = _make_plan()
        s = _make_slice(["features_catalog"])
        facts = required_output_facts_for_slice(plan, s)
        assert facts == ["features_catalog.scopes"]
        assert "research.project_id" not in facts
        assert "research.shortlist_families" not in facts

    def test_non_catalog_slices_unaffected(self) -> None:
        """Non-catalog slices should still return empty list."""
        plan = _make_plan()
        s = _make_slice(["research_map", "research_search"])
        facts = required_output_facts_for_slice(plan, s)
        assert facts == []

    def test_multi_tool_slices_unaffected(self) -> None:
        """Slices with more than just features_catalog should return empty."""
        plan = _make_plan()
        s = _make_slice(["features_catalog", "features_analytics"])
        facts = required_output_facts_for_slice(plan, s)
        assert facts == []


# ---------------------------------------------------------------------------
# CHANGE 7: Config params
# ---------------------------------------------------------------------------

class TestConfigParams:
    def test_direct_execution_config_has_new_fields(self) -> None:
        from app.config import DirectExecutionConfig
        cfg = DirectExecutionConfig()
        assert cfg.lmstudio_zero_tool_retries == 2
        assert cfg.lmstudio_first_turn_tool_choice == "required"

    def test_config_loads_from_dict(self) -> None:
        from app.config import load_config_from_dict
        data = {
            "direct_execution": {
                "lmstudio_zero_tool_retries": 3,
                "lmstudio_first_turn_tool_choice": "auto",
            }
        }
        cfg = load_config_from_dict(data)
        assert cfg.direct_execution.lmstudio_zero_tool_retries == 3
        assert cfg.direct_execution.lmstudio_first_turn_tool_choice == "auto"


# ---------------------------------------------------------------------------
# CHANGE 4: System message improvement
# ---------------------------------------------------------------------------

class TestSystemMessage:
    def test_first_action_guide_contains_protocol(self) -> None:
        loop = _make_tool_loop()
        guide = loop._build_first_action_guide()
        assert "PROTOCOL:" in guide
        assert "Step 1:" in guide
        assert "MUST call at least one tool" in guide

    def test_first_action_guide_contains_example(self) -> None:
        loop = _make_tool_loop()
        guide = loop._build_first_action_guide()
        assert "Example:" in guide


class TestResearchShortlistLoop:
    def test_research_shortlist_inserts_nudge_then_blocks_without_terminal_write(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_map", "research_record"},
            runtime_profile="research_shortlist",
            required_output_facts=[
                "research.project_id",
                "research.memory_node_id",
                "research.shortlist_families",
                "research.novelty_justification_present",
            ],
            success_criteria=["Shortlist exists"],
            slice_title="Form shortlist",
        )
        call_messages: list[list[dict]] = []
        call_count = 0

        def fake_sync(**kwargs):
            nonlocal call_count
            messages = kwargs["messages"]
            call_messages.append(messages)
            call_count += 1
            if call_count <= 5:
                return _tool_call_response("research_map", f"c{call_count}", {"action": "inspect", "project_id": "proj_1"})
            return _text_only_response("unused")

        loop.mcp_client.call_tool = AsyncMock(
            return_value={
                "payload": {
                    "structuredContent": {
                        "status": "ok",
                        "data": {"project_id": "proj_1"},
                    }
                }
            }
        )

        async def _run():
            with (
                patch.object(loop, "_chat_sync", side_effect=fake_sync),
                patch(
                    "app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call",
                    return_value=SimpleNamespace(
                        arguments={"action": "search", "project_id": "proj_1", "query": "feature contract"},
                        local_payload=None,
                        repair_notes=[],
                        charge_budget=True,
                    ),
                ),
            ):
                return await loop.invoke(prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1")

        result = asyncio.run(_run())

        assert "\"reason_code\": \"direct_missing_terminal_write\"" in result.response.raw_output
        assert result.tool_call_count == 5
        assert any(
            any(
                msg.get("role") == "user"
                and "Research shortlist protocol reminder" in str(msg.get("content") or "")
                for msg in messages
            )
            for messages in call_messages
        )

    def test_weak_provider_header_tool_mandate_at_end(self) -> None:
        from app.services.direct_execution.prompt import _build_weak_provider_header
        header = _build_weak_provider_header(24, 6)
        # The CRITICAL tool-call mandate should be in the last few lines
        critical_idx = None
        budget_idx = None
        for i, line in enumerate(header):
            if "CRITICAL" in line:
                critical_idx = i
            if "Tool budget" in line:
                budget_idx = i
        assert critical_idx is not None
        assert budget_idx is not None
        # CRITICAL should come after budget line
        assert critical_idx > budget_idx


class TestMixedDomainLoop:
    def test_mixed_domain_slice_inserts_nudge_then_blocks_research_only_loop(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_memory", "features_custom", "features_dataset"},
            runtime_profile="generic_mutation",
            success_criteria=["Custom features validated and published."],
            slice_title="Data contract and feature contract",
        )
        call_messages: list[list[dict]] = []
        call_count = 0

        def fake_sync(**kwargs):
            nonlocal call_count
            messages = kwargs["messages"]
            call_messages.append(messages)
            call_count += 1
            if call_count <= 6:
                return _tool_call_response(
                    "research_memory",
                    f"c{call_count}",
                    {"action": "search", "project_id": "proj_1", "query": "feature contract"},
                )
            return _text_only_response("unused")

        loop.mcp_client.call_tool = AsyncMock(
            return_value={
                "ok": True,
                "payload": {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Loaded research search results.",
                        "data": {"project_id": "proj_1"},
                    }
                },
            }
        )

        async def _run():
            with (
                patch.object(loop, "_chat_sync", side_effect=fake_sync),
                patch(
                    "app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call",
                    return_value=SimpleNamespace(
                        arguments={"action": "search", "project_id": "proj_1", "query": "feature contract"},
                        local_payload=None,
                        repair_notes=[],
                        charge_budget=True,
                    ),
                ),
            ):
                return await loop.invoke(prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1")

        result = asyncio.run(_run())

        assert "\"reason_code\": \"direct_mixed_domain_exploration_loop\"" in result.response.raw_output
        assert any(
            any(
                msg.get("role") == "user"
                and "Feature contract protocol reminder" in str(msg.get("content") or "")
                for msg in messages
            )
            for messages in call_messages
        )
        assert any(
            msg.get("role") == "user"
            and "Feature contract protocol reminder" in str(msg.get("content") or "")
            for msg in call_messages[1]
        )

    def test_mixed_domain_exploration_slice_blocks_research_only_loop(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_memory", "features_catalog", "events", "datasets"},
            runtime_profile="generic_mutation",
            success_criteria=["Event alignment verified for event hypotheses."],
            slice_title="Data contract and feature contract exploration",
        )
        loop.runtime_profile = "generic_mutation"
        loop.finalization_mode = "none"
        call_messages: list[list[dict]] = []
        call_count = 0

        def fake_sync(**kwargs):
            nonlocal call_count
            messages = kwargs["messages"]
            call_messages.append(messages)
            call_count += 1
            if call_count <= 6:
                return _tool_call_response(
                    "research_memory",
                    f"c{call_count}",
                    {"action": "search", "project_id": "proj_1", "query": "event alignment contract"},
                )
            return _text_only_response("unused")

        loop.mcp_client.call_tool = AsyncMock(
            return_value={
                "ok": True,
                "payload": {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Loaded research search results.",
                        "data": {"project_id": "proj_1"},
                    }
                },
            }
        )

        async def _run():
            with (
                patch.object(loop, "_chat_sync", side_effect=fake_sync),
                patch(
                    "app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call",
                    return_value=SimpleNamespace(
                        arguments={"action": "search", "project_id": "proj_1", "query": "event alignment contract"},
                        local_payload=None,
                        repair_notes=[],
                        charge_budget=True,
                    ),
                ),
            ):
                return await loop.invoke(prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1")

        result = asyncio.run(_run())

        assert "\"reason_code\": \"direct_mixed_domain_exploration_loop\"" in result.response.raw_output
        assert "features_catalog" in result.response.raw_output or "events" in result.response.raw_output or "datasets" in result.response.raw_output

    def test_backtests_analysis_slice_inserts_nudge_then_blocks_research_only_loop(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_memory", "backtests_conditions", "backtests_analysis"},
            runtime_profile="generic_read",
            success_criteria=["Condition stability measured."],
            slice_title="Stability и condition analysis",
        )
        call_messages: list[list[dict]] = []
        call_count = 0

        def fake_sync(**kwargs):
            nonlocal call_count
            messages = kwargs["messages"]
            call_messages.append(messages)
            call_count += 1
            if call_count <= 6:
                return _tool_call_response(
                    "research_memory",
                    f"c{call_count}",
                    {
                        "action": "search",
                        "project_id": "proj_1",
                        "query": "signal family shortlist stability condition regime specialist",
                    },
                )
            return _text_only_response("unused")

        loop.mcp_client.call_tool = AsyncMock(
            return_value={
                "ok": True,
                "payload": {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Loaded shortlist context.",
                        "data": {"project_id": "proj_1"},
                    }
                },
            }
        )

        async def _run():
            with (
                patch.object(loop, "_chat_sync", side_effect=fake_sync),
                patch(
                    "app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call",
                    return_value=SimpleNamespace(
                        arguments={
                            "action": "search",
                            "project_id": "proj_1",
                            "query": "signal family shortlist stability condition regime specialist",
                        },
                        local_payload=None,
                        repair_notes=[],
                        charge_budget=True,
                    ),
                ),
            ):
                return await loop.invoke(prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1")

        result = asyncio.run(_run())

        assert "\"reason_code\": \"direct_mixed_domain_exploration_loop\"" in result.response.raw_output
        assert "backtests_conditions" in result.response.raw_output or "backtests_analysis" in result.response.raw_output
        assert any(
            any(
                msg.get("role") == "user"
                and "Backtests protocol reminder" in str(msg.get("content") or "")
                for msg in messages
            )
            for messages in call_messages
        )

    def test_backtests_slice_inserts_nudge_then_blocks_research_only_loop(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_memory", "backtests_plan", "backtests_runs"},
            runtime_profile="generic_mutation",
            success_criteria=["По каждому есть полная метрическая картина."],
            slice_title="Standalone backtests новых сигналов",
        )
        call_messages: list[list[dict]] = []
        call_count = 0

        def fake_sync(**kwargs):
            nonlocal call_count
            messages = kwargs["messages"]
            call_messages.append(messages)
            call_count += 1
            if call_count <= 6:
                return _tool_call_response(
                    "research_memory",
                    f"c{call_count}",
                    {
                        "action": "search",
                        "project_id": "proj_1",
                        "query": "shortlist candidates signal families",
                    },
                )
            return _text_only_response("unused")

        loop.mcp_client.call_tool = AsyncMock(
            return_value={
                "ok": True,
                "payload": {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Loaded shortlist context.",
                        "data": {"project_id": "proj_1"},
                    }
                },
            }
        )

        async def _run():
            with (
                patch.object(loop, "_chat_sync", side_effect=fake_sync),
                patch(
                    "app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call",
                    return_value=SimpleNamespace(
                        arguments={
                            "action": "search",
                            "project_id": "proj_1",
                            "query": "shortlist candidates signal families",
                        },
                        local_payload=None,
                        repair_notes=[],
                        charge_budget=True,
                    ),
                ),
            ):
                return await loop.invoke(prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1")

        result = asyncio.run(_run())

        assert "\"reason_code\": \"direct_mixed_domain_exploration_loop\"" in result.response.raw_output
        assert "backtests_plan" in result.response.raw_output or "backtests_runs" in result.response.raw_output
        assert any(
            any(
                msg.get("role") == "user"
                and "Backtests protocol reminder" in str(msg.get("content") or "")
                for msg in messages
            )
            for messages in call_messages
        )


class TestResearchSetupProjectSelection:
    def test_research_setup_list_nudges_then_blocks_ambiguous_project_selection(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_project", "research_map", "research_memory"},
            runtime_profile="generic_mutation",
            success_criteria=["Project opened and setup recorded."],
            slice_title="Запуск цикла и фиксация инвариантов",
        )
        call_messages: list[list[dict]] = []
        call_count = 0

        def fake_sync(**kwargs):
            nonlocal call_count
            messages = kwargs["messages"]
            call_messages.append(messages)
            call_count += 1
            if call_count <= 2:
                return _tool_call_response("research_project", f"c{call_count}", {"action": "list"})
            return _text_only_response("unused")

        loop.mcp_client.call_tool = AsyncMock(
            return_value={
                "structuredContent": {
                    "status": "ok",
                    "summary": "Research project payload loaded.",
                    "data": {
                        "projects": [
                            {
                                "project_id": "cycle-research-project-68023013",
                                "name": "Cycle Research Project",
                                "tags": [],
                                "metadata": {},
                            },
                            {
                                "project_id": "v1-cycle-invariants-7750a80d",
                                "name": "v1-cycle-invariants",
                                "tags": [],
                                "metadata": {},
                            },
                        ]
                    },
                }
            }
        )

        async def _run():
            with (
                patch.object(loop, "_chat_sync", side_effect=fake_sync),
                patch(
                    "app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call",
                    return_value=SimpleNamespace(
                        arguments={"action": "list"},
                        local_payload=None,
                        repair_notes=[],
                        charge_budget=True,
                    ),
                ),
            ):
                return await loop.invoke(prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1")

        result = asyncio.run(_run())

        assert "\"reason_code\": \"direct_setup_project_selection_ambiguous\"" in result.response.raw_output
        assert any(
            any(
                msg.get("role") == "user"
                and "project list is discovery only" in str(msg.get("content") or "")
                for msg in messages
            )
            for messages in call_messages[1:]
        )

    def test_research_setup_rewrites_transient_open_to_create(self) -> None:
        loop = _make_tool_loop(
            allowed_tools={"research_project", "research_map", "research_memory"},
            runtime_profile="generic_mutation",
            success_criteria=["Project opened and setup recorded."],
            slice_title="Запуск цикла и фиксация инвариантов",
        )
        call_count = 0

        def fake_sync(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _tool_call_response("research_project", "c1", {"action": "list"})
            if call_count == 2:
                return _tool_call_response(
                    "research_project",
                    "c2",
                    {"action": "open", "project_id": "f322aedc810d46e1b92c7c0ad47784ff"},
                )
            return _text_only_response("done")

        observed_arguments: list[dict] = []

        async def _call_tool(tool_name: str, arguments: dict) -> dict:
            observed_arguments.append({"tool_name": tool_name, "arguments": dict(arguments)})
            if arguments.get("action") == "create":
                return {
                    "structuredContent": {
                        "status": "ok",
                        "summary": "Research project created.",
                        "data": {
                            "project": {
                                "project_id": "research-project-8582ea30",
                                "name": "Запуск цикла и фиксация инвариантов",
                            }
                        },
                    }
                }
            return {
                "structuredContent": {
                    "status": "ok",
                    "summary": "Research project payload loaded.",
                    "data": {
                        "projects": [
                            {
                                "project_id": "cycle-research-project-68023013",
                                "name": "Cycle Research Project",
                                "tags": [],
                                "metadata": {},
                            },
                            {
                                "project_id": "v1-cycle-invariants-7750a80d",
                                "name": "v1-cycle-invariants",
                                "tags": [],
                                "metadata": {},
                            },
                        ]
                    },
                }
            }

        async def _run():
            with (
                patch.object(loop, "_chat_sync", side_effect=fake_sync),
                patch(
                    "app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call",
                    side_effect=lambda tool_name, arguments, catalog_snapshot=None: SimpleNamespace(
                        arguments=arguments,
                        local_payload=None,
                        repair_notes=[],
                        charge_budget=True,
                    ),
                ),
            ):
                loop.mcp_client.call_tool = AsyncMock(side_effect=_call_tool)
                return await loop.invoke(prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1")

        asyncio.run(_run())

        assert observed_arguments[0]["arguments"]["action"] == "list"
        assert observed_arguments[1]["arguments"]["action"] == "create"
        assert "project_id" not in observed_arguments[1]["arguments"]
