"""Unit tests for stream_tool_counter module."""

from __future__ import annotations

import json

from app.services.direct_execution.stream_tool_counter import (
    ToolCallCountResult,
    count_tool_calls_from_single_event,
    count_tool_calls_from_stream_json,
)


# ---------------------------------------------------------------------------
# Fixtures: sample stream-json lines
# ---------------------------------------------------------------------------

def _claude_tool_use_line(tool_name: str) -> str:
    return json.dumps({
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "toolu_123", "name": tool_name, "input": {}},
    })


def _qwen_tool_use_line(tool_name: str) -> str:
    return json.dumps({
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "thinking..."},
                {"type": "tool_use", "id": "call_1", "name": tool_name, "input": {}},
            ],
        },
    })


def _stream_event_wrapper(inner: dict) -> str:
    return json.dumps({"type": "stream_event", "event": inner})


def _text_line() -> str:
    return json.dumps({
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "some text"},
    })


# ---------------------------------------------------------------------------
# Tests: count_tool_calls_from_stream_json
# ---------------------------------------------------------------------------


class TestCountToolCallsFromStreamJson:
    def test_empty_input(self) -> None:
        result = count_tool_calls_from_stream_json("", "claude_cli")
        assert result == ToolCallCountResult(tool_call_count=0, tool_names=[])

    def test_no_tool_calls(self) -> None:
        raw = "\n".join([_text_line(), _text_line()])
        result = count_tool_calls_from_stream_json(raw, "claude_cli")
        assert result.tool_call_count == 0
        assert result.tool_names == []

    def test_claude_single_tool_call(self) -> None:
        raw = "\n".join([
            _text_line(),
            _claude_tool_use_line("research_project"),
            _text_line(),
        ])
        result = count_tool_calls_from_stream_json(raw, "claude_cli")
        assert result.tool_call_count == 1
        assert result.tool_names == ["research_project"]

    def test_claude_multiple_tool_calls(self) -> None:
        raw = "\n".join([
            _claude_tool_use_line("research_project"),
            _text_line(),
            _claude_tool_use_line("research_map"),
            _claude_tool_use_line("features_catalog"),
        ])
        result = count_tool_calls_from_stream_json(raw, "claude_cli")
        assert result.tool_call_count == 3
        assert result.tool_names == ["research_project", "research_map", "features_catalog"]

    def test_qwen_tool_call_with_mcp_prefix(self) -> None:
        raw = _qwen_tool_use_line("mcp__dev_space1__research_project")
        result = count_tool_calls_from_stream_json(raw, "qwen_cli")
        assert result.tool_call_count == 1
        assert result.tool_names == ["research_project"]

    def test_qwen_multiple_tool_calls_with_prefix(self) -> None:
        raw = "\n".join([
            _qwen_tool_use_line("mcp__dev_space1__events"),
            _qwen_tool_use_line("mcp__dev_space1__datasets"),
        ])
        result = count_tool_calls_from_stream_json(raw, "qwen_cli")
        assert result.tool_call_count == 2
        assert result.tool_names == ["events", "datasets"]

    def test_local_tools_are_excluded(self) -> None:
        """Bash, read_file, etc. should NOT be counted."""
        raw = "\n".join([
            _claude_tool_use_line("Bash"),
            _claude_tool_use_line("read_file"),
            _claude_tool_use_line("write_file"),
            _claude_tool_use_line("research_project"),
        ])
        result = count_tool_calls_from_stream_json(raw, "claude_cli")
        assert result.tool_call_count == 1
        assert result.tool_names == ["research_project"]

    def test_qwen_local_tools_with_prefix_excluded(self) -> None:
        """Non-dev_space1 tools even with prefix should be excluded (won't match catalog)."""
        raw = "\n".join([
            _qwen_tool_use_line("mcp__dev_space1__research_record"),
            _claude_tool_use_line("grep_search"),
        ])
        result = count_tool_calls_from_stream_json(raw, "qwen_cli")
        assert result.tool_call_count == 1
        assert result.tool_names == ["research_record"]

    def test_stream_event_wrapper(self) -> None:
        inner = {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "backtests_runs"},
        }
        raw = _stream_event_wrapper(inner)
        result = count_tool_calls_from_stream_json(raw, "claude_cli")
        assert result.tool_call_count == 1
        assert result.tool_names == ["backtests_runs"]

    def test_malformed_json_lines_skipped(self) -> None:
        raw = "\n".join([
            "not json at all",
            "{broken json",
            _claude_tool_use_line("research_project"),
            "",
        ])
        result = count_tool_calls_from_stream_json(raw, "claude_cli")
        assert result.tool_call_count == 1

    def test_mixed_text_and_tool_events(self) -> None:
        raw = "\n".join([
            '{"type":"system","subtype":"init"}',
            _text_line(),
            _claude_tool_use_line("features_catalog"),
            _text_line(),
            _claude_tool_use_line("research_search"),
            '{"type":"result","subtype":"success","result":"ok"}',
        ])
        result = count_tool_calls_from_stream_json(raw, "claude_cli")
        assert result.tool_call_count == 2
        assert "features_catalog" in result.tool_names
        assert "research_search" in result.tool_names

    def test_tool_without_name_ignored(self) -> None:
        raw = json.dumps({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": ""},
        })
        result = count_tool_calls_from_stream_json(raw, "claude_cli")
        assert result.tool_call_count == 0

    def test_all_dev_space1_tools_recognized(self) -> None:
        """Spot-check that common dev_space1 tools are counted."""
        tools = ["research_project", "backtests_runs", "features_custom", "models_train", "events"]
        raw = "\n".join(_claude_tool_use_line(t) for t in tools)
        result = count_tool_calls_from_stream_json(raw, "claude_cli")
        assert result.tool_call_count == len(tools)


# ---------------------------------------------------------------------------
# Tests: count_tool_calls_from_single_event
# ---------------------------------------------------------------------------


class TestCountToolCallsFromSingleEvent:
    def test_empty_line(self) -> None:
        assert count_tool_calls_from_single_event("") == []

    def test_single_tool_use(self) -> None:
        line = _claude_tool_use_line("research_project")
        names = count_tool_calls_from_single_event(line)
        assert names == ["research_project"]

    def test_no_tool_use(self) -> None:
        line = _text_line()
        assert count_tool_calls_from_single_event(line) == []

    def test_malformed_json(self) -> None:
        assert count_tool_calls_from_single_event("{broken") == []

    def test_qwen_prefixed_tool(self) -> None:
        line = _qwen_tool_use_line("mcp__dev_space1__features_catalog")
        names = count_tool_calls_from_single_event(line)
        assert names == ["features_catalog"]

    def test_local_tool_excluded(self) -> None:
        line = _claude_tool_use_line("Bash")
        assert count_tool_calls_from_single_event(line) == []
