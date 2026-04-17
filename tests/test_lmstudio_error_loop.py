"""
Tests for contract_error_streak detection in LmStudioToolLoop.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.direct_execution.lmstudio_tool_loop import (
    LmStudioToolLoop,
    _is_contract_issue_payload,
)
from app.services.direct_execution.issue_classification import classify_issue_payload
from app.services.direct_execution.service import _SOFT_ABORT_REASON_CODES
from tests.mcp_catalog_fixtures import make_catalog_snapshot


# ---------------------------------------------------------------------------
# Unit tests: _is_contract_issue_payload helper
# ---------------------------------------------------------------------------

def test_is_contract_issue_payload_detects_agent_contract_misuse() -> None:
    payload = {
        "ok": False,
        "payload": {
            "content": [{"type": "text", "text": '{"error": {"code": "agent_contract_misuse", "message": "missing field"}}'}]
        },
    }
    assert _is_contract_issue_payload(payload) is True


def test_is_contract_issue_payload_detects_resource_not_found() -> None:
    payload = {
        "ok": True,
        "payload": {
            "content": [{"type": "text", "text": '{"status": "error", "error": {"code": "resource_not_found"}}'}]
        },
    }
    assert _is_contract_issue_payload(payload) is True


def test_is_contract_issue_payload_returns_false_for_clean_payload() -> None:
    payload = {"ok": True, "payload": {"structuredContent": {"status": "completed"}}}
    assert _is_contract_issue_payload(payload) is False


def test_is_contract_issue_payload_returns_false_for_budget_error() -> None:
    payload = {"error": "direct_expensive_tool_budget_exhausted", "max_expensive_tool_calls": 3}
    assert _is_contract_issue_payload(payload) is False


def test_issue_classifier_marks_agent_contract_misuse_as_contract() -> None:
    payload = {
        "ok": False,
        "payload": {
            "content": [{"type": "text", "text": '{"error": {"code": "agent_contract_misuse"}}'}]
        },
    }
    assert classify_issue_payload(payload) == "contract_misuse"


def test_issue_classifier_marks_tools_unavailable_as_infra() -> None:
    payload = {"error": "dev_space1_tools_unavailable"}
    assert classify_issue_payload(payload) == "infra_unavailable"



def test_issue_classifier_marks_auto_salvage_reject_as_contract_misuse() -> None:
    payload = {"error": "auto_salvage_stub_rejected"}
    assert classify_issue_payload(payload) == "contract_misuse"


# ---------------------------------------------------------------------------
# Integration: direct_error_loop_detected is a soft abort code
# ---------------------------------------------------------------------------

def test_direct_error_loop_detected_is_in_soft_abort_reason_codes() -> None:
    assert "direct_error_loop_detected" in _SOFT_ABORT_REASON_CODES
    assert "feature_data_unavailable" in _SOFT_ABORT_REASON_CODES
    assert "no_features_available" in _SOFT_ABORT_REASON_CODES


# ---------------------------------------------------------------------------
# Helpers for async loop tests
# ---------------------------------------------------------------------------

def _make_tool_loop() -> LmStudioToolLoop:
    adapter = MagicMock()
    adapter.temperature = 0.1
    adapter.max_tokens = 512
    adapter.model = "test-model"
    adapter.reasoning_effort = ""
    adapter.extra_body = {}
    adapter.api_key = ""
    adapter.base_url = "http://localhost:9999"

    mcp_client = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            {
                "name": "research_map",
                "description": "map",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client.call_tool = AsyncMock(return_value={"status": "ok", "data": {}})

    incident_store = MagicMock()
    incident_store.record = MagicMock()

    loop = LmStudioToolLoop(
        adapter=adapter,
        mcp_client=mcp_client,
        incident_store=incident_store,
        catalog_snapshot=make_catalog_snapshot(),
        allowed_tools={"research_map"},
        slice_title="test-slice",
        success_criteria=[],
        required_output_facts=[],
        max_tool_calls=24,
        max_expensive_tool_calls=6,
        safe_exclude_tools=set(),
        first_action_timeout_seconds=10,
        stalled_action_timeout_seconds=10,
        zero_tool_retries=0,
        first_turn_tool_choice="auto",
    )
    # Pre-flight health check must be mocked since no real LM Studio server
    # is running during tests.
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


def _terminal_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content, "tool_calls": []}}]}


# ---------------------------------------------------------------------------
# Async loop behavior tests
# ---------------------------------------------------------------------------

def test_two_contract_errors_do_not_trigger_checkpoint() -> None:
    """Two consecutive contract errors — streak=2 — should NOT force a checkpoint yet."""
    loop = _make_tool_loop()

    call_seq = [
        _tool_call_response("research_map", "c1", {"action": "inspect"}),
        _tool_call_response("research_map", "c2", {"action": "inspect"}),
        _terminal_response(
            "```json\n"
            '{"type": "checkpoint", "status": "partial", "summary": "partial done"}'
            "\n```"
        ),
    ]
    idx = 0

    def fake_sync(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal idx
        r = call_seq[idx]
        idx += 1
        return r

    async def _run() -> str:
        with patch.object(loop, "_chat_sync", side_effect=fake_sync):
            result = await loop.invoke(
                prompt="test", timeout_seconds=30, plan_id="plan_1", slice_id="slice_1"
            )
        return result.response.raw_output or ""

    raw = asyncio.run(_run())
    assert "direct_error_loop_detected" not in raw


def test_three_consecutive_contract_errors_return_checkpoint() -> None:
    """Three consecutive local contract errors trigger direct_error_loop_detected checkpoint."""
    loop = _make_tool_loop()

    call_seq = [
        _tool_call_response("research_map", "c1", {"action": "inspect"}),
        _tool_call_response("research_map", "c2", {"action": "inspect"}),
        _tool_call_response("research_map", "c3", {"action": "inspect"}),
        _terminal_response('```json\n{"type": "checkpoint", "status": "partial", "summary": "nope"}\n```'),
    ]
    idx = 0

    def fake_sync(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal idx
        r = call_seq[idx]
        idx += 1
        return r

    async def _run() -> tuple[str, MagicMock]:
        with patch.object(loop, "_chat_sync", side_effect=fake_sync):
            result = await loop.invoke(
                prompt="test", timeout_seconds=30, plan_id="plan_1", slice_id="slice_1"
            )
        return result.response.raw_output or "", loop.incident_store

    raw, incident_store = asyncio.run(_run())
    assert "direct_error_loop_detected" in raw
    assert incident_store.record.called
    call_kwargs = incident_store.record.call_args.kwargs
    assert call_kwargs["severity"] == "medium"
    assert "contract" in call_kwargs["summary"].lower()
    assert call_kwargs["metadata"]["streak"] == 3


def test_streak_resets_after_successful_call() -> None:
    """Two errors, then a successful call, then another error: streak never reaches 3."""
    loop = _make_tool_loop()

    call_seq = [
        _tool_call_response("research_map", "c1", {"action": "inspect"}),
        _tool_call_response("research_map", "c2", {"action": "inspect"}),
        _tool_call_response("research_map", "c3", {"action": "inspect", "project_id": "proj_1"}),
        _tool_call_response("research_map", "c4", {"action": "inspect"}),
        _terminal_response(
            "```json\n"
            '{"type": "checkpoint", "status": "partial", "summary": "done"}'
            "\n```"
        ),
    ]
    idx = 0

    def fake_sync(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal idx
        r = call_seq[idx]
        idx += 1
        return r

    async def _run() -> str:
        with patch.object(loop, "_chat_sync", side_effect=fake_sync):
            result = await loop.invoke(
                prompt="test", timeout_seconds=30, plan_id="plan_1", slice_id="slice_1"
            )
        return result.response.raw_output or ""

    raw = asyncio.run(_run())
    assert "direct_error_loop_detected" not in raw


def test_tool_budget_exhaustion_returns_blocked_checkpoint() -> None:
    loop = _make_tool_loop()
    loop.max_tool_calls = 1

    call_seq = [
        _tool_call_response("research_map", "c1", {"action": "inspect", "project_id": "proj_1"}),
        _tool_call_response("research_map", "c2", {"action": "inspect", "project_id": "proj_1"}),
    ]
    idx = 0

    def fake_sync(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal idx
        r = call_seq[idx]
        idx += 1
        return r

    async def _run() -> tuple[str, MagicMock]:
        with patch.object(loop, "_chat_sync", side_effect=fake_sync):
            result = await loop.invoke(
                prompt="test", timeout_seconds=30, plan_id="plan_1", slice_id="slice_1"
            )
        return result.response.raw_output or "", loop.incident_store

    raw, incident_store = asyncio.run(_run())
    assert "direct_tool_budget_exhausted" in raw
    assert incident_store.record.called


def test_catalog_only_slice_auto_finalizes_after_repeated_successful_reads() -> None:
    adapter = MagicMock()
    adapter.temperature = 0.1
    adapter.max_tokens = 512
    adapter.model = "test-model"
    adapter.reasoning_effort = ""
    adapter.extra_body = {}
    adapter.api_key = ""
    adapter.base_url = "http://localhost:9999"

    mcp_client = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            {
                "name": "features_catalog",
                "description": "catalog",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client.call_tool = AsyncMock(return_value={"status": "ok", "data": {}})

    incident_store = MagicMock()
    incident_store.record = MagicMock()

    loop = LmStudioToolLoop(
        adapter=adapter,
        mcp_client=mcp_client,
        incident_store=incident_store,
        catalog_snapshot=make_catalog_snapshot(),
        allowed_tools={"features_catalog"},
        slice_title="catalog-contract-slice",
        success_criteria=["Each hypothesis has a clean data and feature contract."],
        required_output_facts=[],
        max_tool_calls=24,
        max_expensive_tool_calls=0,
        safe_exclude_tools=set(),
        first_action_timeout_seconds=10,
        stalled_action_timeout_seconds=10,
        zero_tool_retries=0,
        first_turn_tool_choice="auto",
    )
    loop._connection_pool.warm_up = MagicMock(return_value=True)
    loop._connection_pool.health_check = MagicMock(return_value=True)

    call_seq = [
        _tool_call_response("features_catalog", "c1", {"scope": "available"}),
        _tool_call_response("features_catalog", "c2", {"scope": "timeframe", "timeframe": "1h"}),
        _tool_call_response("features_catalog", "c3", {"scope": "timeframe", "timeframe": "5m"}),
    ]
    idx = 0

    def fake_sync(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal idx
        r = call_seq[idx]
        idx += 1
        return r

    async def _run() -> str:
        with patch.object(loop, "_chat_sync", side_effect=fake_sync):
            result = await loop.invoke(
                prompt="test", timeout_seconds=30, plan_id="plan_1", slice_id="slice_1"
            )
        return result.response.raw_output or ""

    raw = asyncio.run(_run())
    assert "\"type\": \"final_report\"" in raw
    assert "features_catalog.timeframes" in raw


def test_research_budget_salvage_auto_finalizes_from_successful_transcript() -> None:
    adapter = MagicMock()
    adapter.temperature = 0.1
    adapter.max_tokens = 512
    adapter.model = "test-model"
    adapter.reasoning_effort = ""
    adapter.extra_body = {}
    adapter.api_key = ""
    adapter.base_url = "http://localhost:9999"

    mcp_client = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            {"name": "research_search", "description": "search", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "research_map", "description": "map", "inputSchema": {"type": "object", "properties": {}}},
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client.call_tool = AsyncMock(return_value={"status": "ok", "data": {}})

    incident_store = MagicMock()
    incident_store.record = MagicMock()

    loop = LmStudioToolLoop(
        adapter=adapter,
        mcp_client=mcp_client,
        incident_store=incident_store,
        catalog_snapshot=make_catalog_snapshot(),
        allowed_tools={"research_search", "research_map"},
        slice_title="research-slice",
        success_criteria=["Wave-1 shortlist exists"],
        required_output_facts=[],
        max_tool_calls=1,
        max_expensive_tool_calls=0,
        safe_exclude_tools=set(),
        first_action_timeout_seconds=10,
        stalled_action_timeout_seconds=10,
        zero_tool_retries=0,
        first_turn_tool_choice="auto",
    )
    loop._connection_pool.warm_up = MagicMock(return_value=True)
    loop._connection_pool.health_check = MagicMock(return_value=True)

    call_seq = [
        _tool_call_response("research_search", "c1", {"project_id": "proj_1", "query": "BTCUSDT"}),
        _tool_call_response("research_map", "c2", {"action": "inspect", "project_id": "proj_1"}),
    ]
    idx = 0

    def fake_sync(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal idx
        r = call_seq[idx]
        idx += 1
        return r

    async def _run() -> str:
        with patch.object(loop, "_chat_sync", side_effect=fake_sync):
            result = await loop.invoke(
                prompt="test", timeout_seconds=30, plan_id="plan_1", slice_id="slice_1"
            )
        return result.response.raw_output or ""

    raw = asyncio.run(_run())
    assert "\"type\": \"final_report\"" in raw
    assert "\"research.project_id\": \"proj_1\"" in raw
    assert "direct.auto_finalized_from_budget_salvage" in raw


def test_expensive_budget_salvage_auto_finalizes_from_backtests_transcript() -> None:
    adapter = MagicMock()
    adapter.temperature = 0.1
    adapter.max_tokens = 512
    adapter.model = "test-model"
    adapter.reasoning_effort = ""
    adapter.extra_body = {}
    adapter.api_key = ""
    adapter.base_url = "http://localhost:9999"

    mcp_client = AsyncMock()
    mcp_client.list_tools = AsyncMock(
        return_value=[
            {"name": "backtests_conditions", "description": "cond", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "backtests_analysis", "description": "analysis", "inputSchema": {"type": "object", "properties": {}}},
        ]
    )
    mcp_client.close = AsyncMock()
    mcp_client.call_tool = AsyncMock(return_value={"structuredContent": {"status": "ok", "data": {"jobs": []}}})

    incident_store = MagicMock()
    incident_store.record = MagicMock()

    loop = LmStudioToolLoop(
        adapter=adapter,
        mcp_client=mcp_client,
        incident_store=incident_store,
        catalog_snapshot=make_catalog_snapshot(),
        allowed_tools={"backtests_conditions", "backtests_analysis"},
        slice_title="expensive-slice",
        success_criteria=["Standalone and integrated quality evidence exists"],
        required_output_facts=[],
        max_tool_calls=24,
        max_expensive_tool_calls=2,
        safe_exclude_tools=set(),
        first_action_timeout_seconds=10,
        stalled_action_timeout_seconds=10,
        zero_tool_retries=0,
        first_turn_tool_choice="auto",
    )
    loop._connection_pool.warm_up = MagicMock(return_value=True)
    loop._connection_pool.health_check = MagicMock(return_value=True)

    call_seq = [
        _tool_call_response("backtests_conditions", "c1", {"action": "run"}),
        _tool_call_response("backtests_conditions", "c2", {"action": "run"}),
        _tool_call_response("backtests_conditions", "c3", {"action": "run"}),
    ]
    idx = 0

    def fake_sync(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal idx
        r = call_seq[idx]
        idx += 1
        return r

    async def _run() -> str:
        with patch.object(loop, "_chat_sync", side_effect=fake_sync):
            result = await loop.invoke(
                prompt="test", timeout_seconds=30, plan_id="plan_1", slice_id="slice_1"
            )
        return result.response.raw_output or ""

    raw = asyncio.run(_run())
    assert "\"type\": \"final_report\"" in raw
    assert "direct.auto_finalized_from_expensive_budget_salvage" in raw


def test_startup_snapshot_path_does_not_fetch_live_schemas_per_slice() -> None:
    loop = _make_tool_loop()
    loop.mcp_client.list_tools = AsyncMock(side_effect=TimeoutError("schema timeout"))

    async def _run() -> str:
        with patch.object(
            loop,
            "_chat_sync",
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "```json\n{\"type\":\"final_report\",\"summary\":\"ok\",\"verdict\":\"COMPLETE\",\"findings\":[],\"facts\":{},\"evidence_refs\":[],\"confidence\":0.8}\n```",
                            "tool_calls": [],
                        }
                    }
                ]
            },
        ):
            result = await loop.invoke(
                prompt="test", timeout_seconds=30, plan_id="plan_1", slice_id="slice_1"
            )
        return result.response.raw_output or ""

    raw = asyncio.run(_run())
    loop.mcp_client.list_tools.assert_not_called()
    assert "direct_model_stalled_before_first_action" not in raw
    assert "\"type\":\"final_report\"" in raw
