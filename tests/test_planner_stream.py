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


def test_extract_structured_output_ignores_tool_result_without_payload() -> None:
    transcript = "\n".join([
        json.dumps({"type": "system", "tools": ["StructuredOutput"]}),
        json.dumps({
            "type": "user",
            "message": {"content": [{"type": "tool_result", "content": "Structured output provided successfully"}]},
        }),
    ])

    extracted = extract_planner_structured_output(
        transcript,
        rendered_text="Structured output provided successfully",
    )

    assert extracted.structured_payload is None
    assert extracted.structured_payload_source == "none"
    assert extracted.saw_tool_result_success is True


def test_extract_structured_output_marks_transport_error_when_activity_has_no_payload() -> None:
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
                "delta": {"type": "input_json_delta", "partial_json": '{"schema_version":3'},
            },
        }),
    ])

    extracted = extract_planner_structured_output(transcript, rendered_text="")

    assert extracted.structured_payload is None
    assert extracted.structured_payload_source == "none"
    assert extracted.structured_delta_bytes > 0
    assert extracted.transport_errors
    assert "StructuredOutput activity detected" in extracted.transport_errors[-1]


def test_extract_structured_output_falls_back_to_rendered_text_json() -> None:
    rendered_text = json.dumps({
        "schema_version": 3,
        "plan_action": "create",
        "plan_version": 1,
        "tasks": [{"stage_number": 0, "stage_name": "Baseline"}],
    })

    extracted = extract_planner_structured_output("", rendered_text=rendered_text)

    assert extracted.structured_payload is not None
    assert extracted.structured_payload_source == "rendered_text"


def test_extract_structured_output_handles_real_regression_shape() -> None:
    payload = {
        "schema_version": 3,
        "plan_action": "create",
        "plan_version": 1,
        "tasks": [{"stage_number": 0, "stage_name": "Baseline"}],
    }
    partial = json.dumps(payload, ensure_ascii=False)
    transcript = "\n".join([
        json.dumps({
            "type": "system",
            "subtype": "init",
            "tools": ["StructuredOutput"],
            "model": "glm-5.1",
        }),
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "call_1", "name": "StructuredOutput", "input": {}},
            },
        }),
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": partial},
            },
        }),
        json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg_1",
                "content": [{"type": "tool_use", "id": "call_1", "name": "StructuredOutput", "input": payload}],
            },
        }),
        json.dumps({
            "type": "user",
            "message": {
                "content": [{"tool_use_id": "call_1", "type": "tool_result", "content": "Structured output provided successfully"}],
            },
            "tool_use_result": "Structured output provided successfully",
        }),
    ])

    extracted = extract_planner_structured_output(
        transcript,
        rendered_text="Structured output provided successfully",
    )

    assert extracted.structured_payload == payload
    assert extracted.structured_payload_source == "tool_use_input"
    assert extracted.saw_tool_result_success is True
