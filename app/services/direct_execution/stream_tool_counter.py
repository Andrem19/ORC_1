"""
Extract tool call counts from CLI adapter stream-json output.

Both Claude CLI and Qwen CLI produce line-delimited stream-json that
contains tool_use events.  This module parses those events and counts
only dev_space1 MCP tools (ignoring local tools like Bash, read_file,
etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.services.direct_execution.tool_catalog import DEFAULT_DEV_SPACE1_TOOLS

_DEV_SPACE1_TOOL_SET: frozenset[str] = frozenset(DEFAULT_DEV_SPACE1_TOOLS)

_MCP_PREFIX = "mcp__dev_space1__"


@dataclass(frozen=True)
class ToolCallCountResult:
    """Result of counting tool calls from stream-json output."""

    tool_call_count: int = 0
    tool_names: list[str] = field(default_factory=list)


def count_tool_calls_from_stream_json(
    raw_stdout: str,
    provider: str = "",
) -> ToolCallCountResult:
    """Parse line-delimited stream-json and count dev_space1 tool_use events.

    Handles both Claude CLI and Qwen CLI stream-json formats:

    * Claude: ``content_block_start`` with ``content_block.type == "tool_use"``
    * Qwen:   ``assistant`` messages with ``tool_use`` content blocks
              (tool names prefixed with ``mcp__dev_space1__``)
    * Both:   ``stream_event`` wrappers around the above

    Only tool names present in ``DEFAULT_DEV_SPACE1_TOOLS`` (after prefix
    stripping) are counted.  Local tools (Bash, read_file, etc.) are ignored.
    """
    if not raw_stdout:
        return ToolCallCountResult()

    names: list[str] = []
    for raw_line in raw_stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        _collect_tool_names(payload, names)

    return ToolCallCountResult(tool_call_count=len(names), tool_names=names)


def count_tool_calls_from_single_event(
    raw_line: str,
) -> list[str]:
    """Count dev_space1 tool names in a single stream-json line.

    Used for incremental counting in async ``_consume_fragment()`` paths.
    Returns list of normalised tool names found (may be empty).
    """
    line = raw_line.strip()
    if not line:
        return []
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return []
    names: list[str] = []
    _collect_tool_names(payload, names)
    return names


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _collect_tool_names(payload: Any, out: list[str]) -> None:
    """Walk a single stream-json payload and append dev_space1 tool names."""
    if not isinstance(payload, dict):
        return

    event_type = str(payload.get("type", "")).lower()

    # Unwrap ``stream_event`` wrapper (Claude stream-json often wraps events)
    if event_type == "stream_event":
        inner = payload.get("event")
        if isinstance(inner, dict):
            _collect_tool_names(inner, out)
        return

    # Claude format: content_block_start → content_block.type == "tool_use"
    if event_type == "content_block_start":
        content_block = payload.get("content_block")
        if isinstance(content_block, dict) and str(content_block.get("type", "")).lower() == "tool_use":
            name = _normalize_tool_name(content_block.get("name", ""))
            if name and name in _DEV_SPACE1_TOOL_SET:
                out.append(name)
        return

    # Claude / Qwen format: assistant message with content blocks
    if event_type == "assistant":
        message = payload.get("message")
        if isinstance(message, dict):
            _scan_content_blocks(message.get("content"), out)
        # Also check top-level content (some Qwen variants)
        _scan_content_blocks(payload.get("content"), out)
        return

    # Generic message with content blocks
    if "content" in payload and isinstance(payload.get("content"), list):
        _scan_content_blocks(payload["content"], out)


def _scan_content_blocks(content: Any, out: list[str]) -> None:
    """Scan a list of content blocks for tool_use entries."""
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if str(block.get("type", "")).lower() == "tool_use":
            name = _normalize_tool_name(block.get("name", ""))
            if name and name in _DEV_SPACE1_TOOL_SET:
                out.append(name)


def _normalize_tool_name(raw_name: Any) -> str:
    """Strip ``mcp__dev_space1__`` prefix and return canonical tool name."""
    name = str(raw_name or "").strip()
    if name.startswith(_MCP_PREFIX):
        name = name[len(_MCP_PREFIX):]
    return name


__all__ = [
    "ToolCallCountResult",
    "count_tool_calls_from_stream_json",
    "count_tool_calls_from_single_event",
]
