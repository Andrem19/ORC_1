"""Debug script for tool_choice behavior."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.direct_execution.lmstudio_tool_loop import LmStudioToolLoop
from tests.mcp_catalog_fixtures import make_catalog_snapshot


def _tool_call_response(tool_name, call_id, arguments):
    return {
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments),
                    },
                }],
            }
        }]
    }


def _text_only_response(content):
    return {"choices": [{"message": {"content": content, "tool_calls": []}}]}


def _make_tool_loop():
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
        adapter=adapter, mcp_client=mcp_client, incident_store=incident_store,
        catalog_snapshot=make_catalog_snapshot(), allowed_tools={"research_map"},
        slice_title="test-slice", success_criteria=[], required_output_facts=[],
        max_tool_calls=24, max_expensive_tool_calls=6, safe_exclude_tools=set(),
        first_action_timeout_seconds=10, stalled_action_timeout_seconds=10,
        zero_tool_retries=2, first_turn_tool_choice="required",
    )
    loop._connection_pool.warm_up = MagicMock(return_value=True)
    loop._connection_pool.health_check = MagicMock(return_value=True)
    return loop


state = {"call_idx": 0, "call_kwargs_list": []}


def fake_sync_seq(**kwargs):
    state["call_kwargs_list"].append(kwargs)
    state["call_idx"] += 1
    ci = state["call_idx"]
    tc = kwargs.get("tool_choice")
    print(f"  call_idx={ci}, tool_choice={tc}")
    if ci == 1:
        return _tool_call_response("research_map", "c1", {"action": "inspect"})
    return _text_only_response(
        '```json\n{"type":"final_report","summary":"ok","verdict":"WATCHLIST",'
        '"findings":[],"facts":{},"evidence_refs":["transcript:1:research_map"],"confidence":0.7}\n```'
    )


async def _run():
    loop = _make_tool_loop()
    with patch.object(loop, "_chat_sync", side_effect=fake_sync_seq):
        result = await loop.invoke(
            prompt="test", timeout_seconds=30, plan_id="p1", slice_id="s1"
        )
    print(f"Result: tool_call_count={result.tool_call_count}")
    print(f"Transcript kinds: {[e.get('kind') for e in result.transcript]}")


asyncio.run(_run())
print(f"Total calls: {len(state['call_kwargs_list'])}")
for i, kw in enumerate(state["call_kwargs_list"]):
    print(f"  Call {i+1}: tool_choice={kw['tool_choice']}")
