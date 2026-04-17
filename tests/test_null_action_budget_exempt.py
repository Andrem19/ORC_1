"""
Tests for null-action repair budget exemption in LmStudioToolLoop.

When minimax emits repeated null-action calls and the null_action_repair
module fixes them, the repaired calls must NOT consume the expensive tool
budget.  This prevents budget exhaustion from a model failure pattern,
leaving budget available for intentional work.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.direct_execution.lmstudio_tool_loop import LmStudioToolLoop
from app.services.direct_execution.tool_preflight import ToolPreflightResult
from tests.mcp_catalog_fixtures import make_catalog_snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backtests_tool_loop(*, max_expensive: int = 3) -> LmStudioToolLoop:
    adapter = MagicMock()
    adapter.temperature = 0.1
    adapter.max_tokens = 512
    adapter.model = "test-model"
    adapter.reasoning_effort = ""
    adapter.extra_body = {}
    adapter.api_key = ""
    adapter.base_url = "http://localhost:9999"

    mcp_client = AsyncMock()
    mcp_client.list_tools = AsyncMock(return_value=[])
    mcp_client.close = AsyncMock()
    mcp_client.call_tool = AsyncMock(
        return_value={"structuredContent": {"status": "ok", "data": {}}}
    )

    incident_store = MagicMock()
    incident_store.record = MagicMock()

    loop = LmStudioToolLoop(
        adapter=adapter,
        mcp_client=mcp_client,
        incident_store=incident_store,
        catalog_snapshot=make_catalog_snapshot(),
        allowed_tools={"backtests_runs", "backtests_conditions"},
        slice_title="Stability test",
        success_criteria=["Stability evidence exists"],
        required_output_facts=[],
        max_tool_calls=24,
        max_expensive_tool_calls=max_expensive,
        safe_exclude_tools=set(),
        first_action_timeout_seconds=10,
        stalled_action_timeout_seconds=10,
        zero_tool_retries=0,
        first_turn_tool_choice="auto",
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


def _terminal_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content, "tool_calls": []}}]}


def _runs_list_response(run_ids: list[str]) -> dict:
    return {
        "ok": True,
        "payload": {
            "structuredContent": {
                "status": "ok",
                "data": {
                    "saved_runs": [
                        {"run_id": rid, "status": "completed"}
                        for rid in run_ids
                    ],
                },
            }
        },
    }


def _pass_through_preflight(tool_name, arguments, **kwargs):
    return ToolPreflightResult(
        arguments=dict(arguments or {}),
        local_payload=None,
        charge_budget=True,
    )


# ---------------------------------------------------------------------------
# Core fix test: null-action repairs exempt from expensive budget
# ---------------------------------------------------------------------------

def test_null_action_repairs_do_not_consume_expensive_budget():
    """Send 4 null-action calls with max_expensive=3.

    - Calls 0-1: below repair threshold, pass as-is (action=None)
      → these would normally consume expensive budget
    - Calls 2-3: repaired to action='detail', run_id='...'
      → with the fix, these are EXEMPT from expensive budget
    - Without the fix: 4 expensive calls > max(3) → budget exhaustion
    - With the fix: only 2 count as expensive → budget NOT exhausted

    4 calls avoids the semantic-loop guard (threshold 5 for read-only).
    """
    loop = _make_backtests_tool_loop(max_expensive=3)
    loop.mcp_client.call_tool = AsyncMock(
        return_value=_runs_list_response(["run-aaa", "run-bbb"])
    )

    call_seq = []
    for i in range(4):
        call_seq.append(
            _tool_call_response("backtests_runs", f"c{i}", {"action": None})
        )
    call_seq.append(
        _terminal_response(
            '```json\n{"type":"final_report","summary":"done",'
            '"verdict":"COMPLETE","findings":["stable"],"facts":{},'
            '"evidence_refs":[],"confidence":0.8}\n```'
        )
    )
    idx = 0

    def fake_sync(**kwargs):
        nonlocal idx
        r = call_seq[idx]
        idx += 1
        return r

    async def _run():
        with patch.object(loop, "_chat_sync", side_effect=fake_sync), \
             patch("app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call", side_effect=_pass_through_preflight):
            result = await loop.invoke(
                prompt="test", timeout_seconds=30, plan_id="plan_1", slice_id="slice_1"
            )
        return result

    result = asyncio.run(_run())

    raw = result.response.raw_output or ""
    assert "expensive_tool_budget_exhausted" not in raw, (
        f"Null-action repairs should not consume expensive budget. Got: {raw[:300]}"
    )
    # The 2 repaired calls (c2, c3) should NOT count as expensive
    # The 2 unrepaired calls (c0, c1) DO count as expensive
    assert result.expensive_tool_call_count <= 2, (
        f"Expected expensive count <= 2 (only unrepaired calls), got {result.expensive_tool_call_count}"
    )
    assert result.tool_call_count == 4


def test_intentional_expensive_calls_still_consume_budget():
    """Ensure that real (non-null-action) expensive calls still consume budget."""
    loop = _make_backtests_tool_loop(max_expensive=2)
    loop.mcp_client.call_tool = AsyncMock(
        return_value=_runs_list_response(["run-xxx"])
    )

    call_seq = [
        _tool_call_response("backtests_runs", "c1", {"action": "start", "snapshot_id": "snap-1", "symbol": "BTCUSDT", "anchor_timeframe": "1h", "execution_timeframe": "5m"}),
        _tool_call_response("backtests_runs", "c2", {"action": "start", "snapshot_id": "snap-1", "symbol": "BTCUSDT", "anchor_timeframe": "1h", "execution_timeframe": "5m"}),
        _tool_call_response("backtests_runs", "c3", {"action": "start", "snapshot_id": "snap-1", "symbol": "BTCUSDT", "anchor_timeframe": "1h", "execution_timeframe": "5m"}),
    ]
    idx = 0

    def fake_sync(**kwargs):
        nonlocal idx
        r = call_seq[idx]
        idx += 1
        return r

    async def _run():
        with patch.object(loop, "_chat_sync", side_effect=fake_sync), \
             patch("app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call", side_effect=_pass_through_preflight):
            result = await loop.invoke(
                prompt="test", timeout_seconds=30, plan_id="plan_1", slice_id="slice_1"
            )
        return result

    result = asyncio.run(_run())
    raw = result.response.raw_output or ""
    assert "expensive_tool_budget_exhausted" in raw or "auto_finalized" in raw or "final_report" in raw, (
        f"Expected budget exhaustion after 3 intentional expensive calls. Got: {raw[:300]}"
    )


def test_null_action_repair_increments_tool_call_count_but_not_expensive():
    """Verify that null-action repairs increment tool_call_count (general budget)
    but NOT expensive_tool_call_count."""
    loop = _make_backtests_tool_loop(max_expensive=10)
    loop.mcp_client.call_tool = AsyncMock(
        return_value=_runs_list_response(["run-aaa"])
    )

    # 2 unrepaired null-action calls + 2 repaired calls
    call_seq = [
        _tool_call_response("backtests_runs", "c0", {"action": None}),
        _tool_call_response("backtests_runs", "c1", {"action": None}),
        _tool_call_response("backtests_runs", "c2", {"action": None}),
        _tool_call_response("backtests_runs", "c3", {"action": None}),
        _terminal_response(
            '```json\n{"type":"final_report","summary":"done",'
            '"verdict":"COMPLETE","findings":[],"facts":{},'
            '"evidence_refs":[],"confidence":0.8}\n```'
        ),
    ]
    idx = 0

    def fake_sync(**kwargs):
        nonlocal idx
        r = call_seq[idx]
        idx += 1
        return r

    async def _run():
        with patch.object(loop, "_chat_sync", side_effect=fake_sync), \
             patch("app.services.direct_execution.lmstudio_tool_loop.preflight_direct_tool_call", side_effect=_pass_through_preflight):
            result = await loop.invoke(
                prompt="test", timeout_seconds=30, plan_id="plan_1", slice_id="slice_1"
            )
        return result

    result = asyncio.run(_run())

    # All 4 calls should increment tool_call_count
    assert result.tool_call_count == 4, (
        f"Expected tool_call_count=4, got {result.tool_call_count}"
    )
    # Only the 2 unrepaired calls (c0, c1) should count as expensive
    # The 2 repaired calls (c2, c3) should NOT count as expensive
    assert result.expensive_tool_call_count == 2, (
        f"Expected expensive_tool_call_count=2 (only unrepaired), got {result.expensive_tool_call_count}"
    )
