"""Tests for planner stream rendering and classification."""

from __future__ import annotations

import json

from app.planner_stream import (
    consume_stream_fragment,
    extract_markdown_from_thinking_transcript,
    inspect_stream_transcript,
)


def test_consume_stream_fragment_ignores_thinking_and_keeps_text() -> None:
    lines = [
        json.dumps({"type": "message_start", "message": {"id": "abc"}}),
        json.dumps({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Let me think"}}),
        json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "# Plan v1"}}),
        json.dumps({"type": "message_stop"}),
    ]

    rendered, buffer, event_count = consume_stream_fragment("\n".join(lines) + "\n", "")

    assert rendered == "# Plan v1"
    assert buffer == ""
    assert event_count == 4


def test_consume_stream_fragment_extracts_stream_event_wrapper_text() -> None:
    lines = [
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": "# Plan v1"},
            },
        }),
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "\n## ETAP 1"},
            },
        }),
    ]

    rendered, buffer, event_count = consume_stream_fragment("\n".join(lines) + "\n", "")

    assert rendered == "# Plan v1\n## ETAP 1"
    assert buffer == ""
    assert event_count == 2


def test_inspect_stream_transcript_detects_tool_use_without_visible_text() -> None:
    transcript = "\n".join([
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""},
            },
        }),
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Agent", "input": {}},
            },
        }),
    ])

    state = inspect_stream_transcript(transcript)

    assert state["tool_use_name"] == "Agent"
    assert state["has_visible_text"] is False
    assert state["thinking_only"] is True


def test_inspect_stream_transcript_detects_visible_text() -> None:
    transcript = "\n".join([
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""},
            },
        }),
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": "# Plan v1"},
            },
        }),
    ])

    state = inspect_stream_transcript(transcript)

    assert state["has_visible_text"] is True
    assert state["thinking_only"] is False


def test_extract_markdown_from_thinking_transcript_recovers_plan_text() -> None:
    transcript = "\n".join([
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": "# Plan v3\n\n## Status and Frame"},
            },
        }),
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "\n\n## ETAP 1: Test"},
            },
        }),
    ])

    recovered = extract_markdown_from_thinking_transcript(transcript)

    assert recovered.startswith("# Plan v3")
    assert "## ETAP 1: Test" in recovered


def test_extract_markdown_from_thinking_transcript_trims_post_plan_reasoning() -> None:
    transcript = "\n".join([
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {
                    "type": "thinking",
                    "thinking": (
                        "# Plan v1\n\n## Status and Frame\nBaseline.\n\n## Goal\nGoal.\n\n## Baseline\nBase.\n\n"
                        "## Research Principles\n- One.\n\n## dev_space1 Capabilities\nWorkers.\n\n"
                        "## ETAP 1: A\nGoal: one.\n1. Step.\n2. Step.\n3. Step.\n4. Step.\nCompletion criteria: ok.\n"
                        "| a | b |\n| --- | --- |\n| x | y |\n\n"
                        "## ETAP 2: B\nGoal: two.\n1. Step.\n2. Step.\n3. Step.\n4. Step.\nCompletion criteria: ok.\n"
                        "| a | b |\n| --- | --- |\n| x | y |\n\n"
                        "## ETAP 3: C\nGoal: three.\n1. Step.\n2. Step.\n3. Step.\n4. Step.\nCompletion criteria: ok.\n"
                        "| a | b |\n| --- | --- |\n| x | y |\n\n"
                        "Hmm, wait. I should reconsider the code parameter."
                    ),
                },
            },
        }),
    ])

    recovered = extract_markdown_from_thinking_transcript(transcript)

    assert recovered.startswith("# Plan v1")
    assert recovered.endswith("| x | y |")
    assert "Hmm, wait." not in recovered
