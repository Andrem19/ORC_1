"""Tests for clean planner stream rendering."""

from __future__ import annotations

import json

from app.planner_stream import consume_stream_fragment


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
