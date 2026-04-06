"""Tests for clean planner stream rendering and structured extraction."""

from __future__ import annotations

import json

from app.planner_stream import consume_stream_fragment
from app.planner_structured_output import extract_planner_structured_output


def test_consume_stream_fragment_ignores_thinking_and_metadata() -> None:
    lines = [
        json.dumps({"type": "message_start", "message": {"id": "abc"}}),
        json.dumps({"type": "content_block_delta", "delta": {"type": "thinking_delta", "text": "Let me think"}}),
        json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": '{"plan_action":"create"'}}),
        json.dumps({"type": "message_stop"}),
    ]

    rendered, buffer, event_count = consume_stream_fragment("\n".join(lines) + "\n", "")

    assert rendered.startswith('{"plan_action":"create"')
    assert "thinking" not in rendered
    assert buffer == ""
    assert event_count == 4


def test_consume_stream_fragment_keeps_final_plain_text_when_not_json() -> None:
    rendered, buffer, event_count = consume_stream_fragment("final plain text\n", "")

    assert rendered == "final plain text"
    assert buffer == ""
    assert event_count == 0


def test_extract_structured_output_from_tool_use_input() -> None:
    payload = {
        "schema_version": 3,
        "plan_action": "create",
        "plan_version": 1,
        "tasks": [{"stage_number": 0, "stage_name": "Baseline"}],
    }
    transcript = "\n".join([
        json.dumps({"type": "system", "tools": ["StructuredOutput"]}),
        json.dumps({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "summary"},
                    {"type": "tool_use", "name": "StructuredOutput", "input": payload},
                ]
            },
        }),
        json.dumps({
            "type": "user",
            "message": {"content": [{"type": "tool_result", "content": "Structured output provided successfully"}]},
        }),
    ])

    extracted = extract_planner_structured_output(transcript, rendered_text="Structured output provided successfully")

    assert extracted.structured_payload == payload
    assert extracted.structured_payload_source == "tool_use_input"
    assert extracted.saw_tool_result_success is True


def test_extract_structured_output_from_input_json_delta() -> None:
    transcript = "\n".join([
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "StructuredOutput", "input": {}},
            },
        }),
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "{\"schema_version\":3,\"plan_action\":\"create\",\"plan_version\":1,\"tasks\":[",
                },
            },
        }),
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "{\"stage_number\":0,\"stage_name\":\"Baseline\"}]}",
                },
            },
        }),
    ])

    extracted = extract_planner_structured_output(transcript, rendered_text="")

    assert extracted.structured_payload is not None
    assert extracted.structured_payload["plan_version"] == 1
    assert extracted.structured_payload["tasks"][0]["stage_number"] == 0
    assert extracted.structured_payload_source == "input_json_delta"
