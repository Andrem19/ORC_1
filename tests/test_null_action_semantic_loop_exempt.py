"""
Tests that null-action-repaired calls do not trigger semantic loop detection.

Regression test for compiled_plan_v1_stage_7: minimax entered a null-action
loop on backtests_runs, the null_action_repair module fixed every call to
the same action='detail' + run_id='...' arguments.  The semantic loop
detector saw 5 identical *repaired* signatures and fired an abort.  The
abort was then salvaged into a final_report with auto_finalized_from_*
facts, which the quality gate rejected as auto_salvage_stub_rejected.

Fix: null-action-repaired calls must not update the semantic loop counter,
because the repaired signature is an artifact of the repair picking the
same extractable ID every time, not the model deliberately repeating a call.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.direct_execution.lmstudio_tool_loop import LmStudioToolLoop
from app.services.direct_execution.tool_preflight import ToolPreflightResult
from tests.mcp_catalog_fixtures import make_catalog_snapshot


def _make_backtests_tool_loop(*, max_tool_calls: int = 24, max_expensive: int = 12) -> LmStudioToolLoop:
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
        allowed_tools={"backtests_runs"},
        slice_title="Integration test slice",
        success_criteria=["Integration evidence exists"],
        required_output_facts=[],
        max_tool_calls=max_tool_calls,
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


def test_null_action_repairs_do_not_trigger_semantic_loop():
    """Send 7 null-action calls — enough to trigger semantic loop at threshold 5.

    Without the fix: the null-action repair converts all calls from call 2 onward
    to the same action='detail' + run_id. The semantic loop detector sees 5
    consecutive identical signatures → fires abort with direct_semantic_loop_detected.

    With the fix: null-action-repaired calls are excluded from the semantic loop
    counter. The model continues to the terminal response.
    """
    loop = _make_backtests_tool_loop(max_tool_calls=20, max_expensive=20)
    loop.mcp_client.call_tool = AsyncMock(
        return_value=_runs_list_response(["analysis-d1b90901b5be"])
    )

    call_seq = []
    for i in range(7):
        call_seq.append(
            _tool_call_response("backtests_runs", f"c{i}", {"action": None})
        )
    call_seq.append(
        _terminal_response(
            '```json\n{"type":"final_report","summary":"Integration complete",'
            '"verdict":"COMPLETE","findings":["all candidates tested"],"facts":{},'
            '"evidence_refs":["transcript:1:backtests_runs"],"confidence":0.85}\n```'
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
                prompt="test", timeout_seconds=60, plan_id="plan_1", slice_id="slice_7"
            )
        return result

    result = asyncio.run(_run())

    raw = result.response.raw_output or ""
    assert "direct_semantic_loop_detected" not in raw, (
        f"Null-action-repaired calls must not trigger semantic loop. Got: {raw[:400]}"
    )
    assert "final_report" in raw, (
        f"Expected final_report after null-action repairs, got: {raw[:400]}"
    )
    assert result.tool_call_count == 7, (
        f"Expected 7 tool calls, got {result.tool_call_count}"
    )


def test_intentional_identical_calls_still_trigger_semantic_loop():
    """Ensure that real (non-null-action) identical calls still trigger semantic loop.

    This verifies the fix didn't break legitimate semantic loop detection.
    """
    loop = _make_backtests_tool_loop(max_tool_calls=20, max_expensive=20)
    loop.mcp_client.call_tool = AsyncMock(
        return_value=_runs_list_response(["run-aaa"])
    )

    # 6 identical intentional calls with action='detail' and the same run_id
    call_seq = []
    for i in range(6):
        call_seq.append(
            _tool_call_response("backtests_runs", f"c{i}", {
                "action": "detail",
                "run_id": "run-aaa",
            })
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
                prompt="test", timeout_seconds=60, plan_id="plan_1", slice_id="slice_test"
            )
        return result

    result = asyncio.run(_run())

    raw = result.response.raw_output or ""
    assert "direct_semantic_loop_detected" in raw, (
        f"Intentional identical calls should still trigger semantic loop. Got: {raw[:400]}"
    )


def test_null_action_loop_with_alternating_real_calls():
    """Mixed scenario: null-action calls interleaved with real different calls.

    The null-action repairs should not interfere with real semantic loop detection
    on the interspersed different calls.
    """
    loop = _make_backtests_tool_loop(max_tool_calls=20, max_expensive=20)
    loop.mcp_client.call_tool = AsyncMock(
        return_value=_runs_list_response(["run-xxx"])
    )

    call_seq = [
        # Real call 1
        _tool_call_response("backtests_runs", "c0", {"action": "inspect", "view": "list"}),
        # Null-action call 1
        _tool_call_response("backtests_runs", "c1", {"action": None}),
        # Null-action call 2
        _tool_call_response("backtests_runs", "c2", {"action": None}),
        # Null-action call 3 (repaired)
        _tool_call_response("backtests_runs", "c3", {"action": None}),
        # Null-action call 4 (repaired)
        _tool_call_response("backtests_runs", "c4", {"action": None}),
        # Real call 2 (different from call 1)
        _tool_call_response("backtests_runs", "c5", {"action": "detail", "run_id": "run-xxx"}),
        # Terminal
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
                prompt="test", timeout_seconds=60, plan_id="plan_1", slice_id="slice_mix"
            )
        return result

    result = asyncio.run(_run())

    raw = result.response.raw_output or ""
    assert "direct_semantic_loop_detected" not in raw, (
        f"Mixed null-action + real calls should not trigger semantic loop. Got: {raw[:400]}"
    )
    assert result.tool_call_count == 6
