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

_MCP_PREFIX = "mcp__dev_space1__"
_GENERIC_DEVSPACE1_PREFIXES = {
    "backtests",
    "datasets",
    "events",
    "experiments",
    "features",
    "incidents",
    "models",
    "notify",
    "research",
    "signal",
    "system",
}
@dataclass(frozen=True)
class ToolCallCountResult:
    """Result of counting tool calls from stream-json output."""

    tool_call_count: int = 0
    tool_names: list[str] = field(default_factory=list)


def count_tool_calls_from_stream_json(
    raw_stdout: str,
    provider: str = "",
    allowed_tool_names: set[str] | None = None,
) -> ToolCallCountResult:
    """Parse line-delimited stream-json and count dev_space1 tool_use events.

    Handles both Claude CLI and Qwen CLI stream-json formats:

    * Claude: ``content_block_start`` with ``content_block.type == "tool_use"``
    * Qwen:   ``assistant`` messages with ``tool_use`` content blocks
              (tool names prefixed with ``mcp__dev_space1__``)
    * Both:   ``stream_event`` wrappers around the above

    When a live tool set is supplied, only those tool names are counted.
    Without a live tool set this parser degrades to MCP-prefix evidence only,
    which is suitable for postmortem stream inspection but not authoritative
    runtime accounting.
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
        _collect_tool_names(payload, names, allowed_tool_names=allowed_tool_names)

    return ToolCallCountResult(tool_call_count=len(names), tool_names=names)


def count_tool_calls_from_single_event(
    raw_line: str,
    *,
    allowed_tool_names: set[str] | None = None,
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
    _collect_tool_names(payload, names, allowed_tool_names=allowed_tool_names)
    return names


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _collect_tool_names(payload: Any, out: list[str], *, allowed_tool_names: set[str] | None) -> None:
    """Walk a single stream-json payload and append dev_space1 tool names."""
    if not isinstance(payload, dict):
        return

    event_type = str(payload.get("type", "")).lower()

    # Unwrap ``stream_event`` wrapper (Claude stream-json often wraps events)
    if event_type == "stream_event":
        inner = payload.get("event")
        if isinstance(inner, dict):
            _collect_tool_names(inner, out, allowed_tool_names=allowed_tool_names)
        return

    # Claude format: content_block_start → content_block.type == "tool_use"
    if event_type == "content_block_start":
        content_block = payload.get("content_block")
        if isinstance(content_block, dict) and str(content_block.get("type", "")).lower() == "tool_use":
            raw_name = str(content_block.get("name", "") or "")
            name = _normalize_tool_name(raw_name)
            if _is_dev_space1_tool_name(name, raw_name=raw_name, allowed_tool_names=allowed_tool_names):
                out.append(name)
        return

    # Claude / Qwen format: assistant message with content blocks
    if event_type == "assistant":
        message = payload.get("message")
        if isinstance(message, dict):
            _scan_content_blocks(message.get("content"), out, allowed_tool_names=allowed_tool_names)
        # Also check top-level content (some Qwen variants)
        _scan_content_blocks(payload.get("content"), out, allowed_tool_names=allowed_tool_names)
        return

    # Generic message with content blocks
    if "content" in payload and isinstance(payload.get("content"), list):
        _scan_content_blocks(payload["content"], out, allowed_tool_names=allowed_tool_names)


def _scan_content_blocks(content: Any, out: list[str], *, allowed_tool_names: set[str] | None) -> None:
    """Scan a list of content blocks for tool_use entries."""
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if str(block.get("type", "")).lower() == "tool_use":
            raw_name = str(block.get("name", "") or "")
            name = _normalize_tool_name(raw_name)
            if _is_dev_space1_tool_name(name, raw_name=raw_name, allowed_tool_names=allowed_tool_names):
                out.append(name)


def _normalize_tool_name(raw_name: Any) -> str:
    """Strip ``mcp__dev_space1__`` prefix and return canonical tool name."""
    name = str(raw_name or "").strip()
    if name.startswith(_MCP_PREFIX):
        name = name[len(_MCP_PREFIX):]
    return name


def _is_dev_space1_tool_name(name: str, *, raw_name: str = "", allowed_tool_names: set[str] | None) -> bool:
    normalized = str(name or "").strip()
    if not normalized:
        return False
    if allowed_tool_names is not None:
        return normalized in allowed_tool_names
    raw_text = str(raw_name or "").strip()
    if raw_text.startswith(_MCP_PREFIX):
        return True
    prefix = normalized.split("_", 1)[0].strip().lower()
    return prefix in _GENERIC_DEVSPACE1_PREFIXES


__all__ = [
    "ToolCallCountResult",
    "count_tool_calls_from_stream_json",
    "count_tool_calls_from_single_event",
]
