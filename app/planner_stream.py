"""
Helpers for parsing Claude stream-json planner output.
"""

from __future__ import annotations

import json
from typing import Any


_IGNORED_EVENT_TYPES = {
    "message_start",
    "message_stop",
    "content_block_stop",
    "ping",
    "system",
    "error",
}
_TEXT_BLOCK_TYPES = {"text", "output_text"}
_TEXT_DELTA_TYPES = {"text_delta", "output_text_delta"}
_IGNORED_BLOCK_TYPES = {"thinking", "thinking_delta", "signature"}


def consume_stream_fragment(fragment: str, buffer: str) -> tuple[str, str, int]:
    """Consume line-delimited stream-json and return rendered text.

    Returns:
      (rendered_text_delta, remaining_buffer, event_count)
    """
    rendered: list[str] = []
    event_count = 0
    buffer += fragment

    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            # Keep plain-text fallback for non-stream modes or malformed tail chunks.
            _append_rendered(rendered, line)
            continue
        event_count += 1
        text = extract_stream_text(payload)
        if text:
            _append_rendered(rendered, text)

    return "".join(rendered), buffer, event_count


def extract_stream_text(payload: Any) -> str:
    """Extract only assistant-visible text from one Claude stream-json event."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return "".join(extract_stream_text(item) for item in payload)
    if not isinstance(payload, dict):
        return ""

    event_type = str(payload.get("type", "")).lower()
    if event_type in _IGNORED_EVENT_TYPES:
        return ""
    result = payload.get("result")
    if isinstance(result, str) and event_type in {"result", "final"}:
        return result

    if event_type == "content_block_delta":
        return _extract_delta_text(payload.get("delta"))

    if event_type == "content_block_start":
        return _extract_content_block_text(payload.get("content_block"))

    if event_type == "message_delta":
        return ""

    if event_type == "content":
        return _extract_content_block_text(payload)

    if "delta" in payload:
        delta_text = _extract_delta_text(payload.get("delta"))
        if delta_text:
            return delta_text

    if "content_block" in payload:
        block_text = _extract_content_block_text(payload.get("content_block"))
        if block_text:
            return block_text

    if "content" in payload:
        content_text = _extract_content_block_text(payload.get("content"))
        if content_text:
            return content_text

    if "message" in payload:
        return extract_stream_text(payload.get("message"))

    text = payload.get("text")
    if isinstance(text, str):
        return text

    output = payload.get("output")
    if isinstance(output, str):
        return output

    completion = payload.get("completion")
    if isinstance(completion, str):
        return completion

    return ""


def _append_rendered(rendered: list[str], text: str) -> None:
    if not text:
        return
    current = "".join(rendered)
    if current.endswith(text):
        return
    if len(text) >= 120 and text in current:
        return
    rendered.append(text)


def _extract_delta_text(delta: Any) -> str:
    if isinstance(delta, str):
        return delta
    if not isinstance(delta, dict):
        return ""
    delta_type = str(delta.get("type", "")).lower()
    if delta_type in _IGNORED_BLOCK_TYPES:
        return ""
    if delta_type in _TEXT_DELTA_TYPES:
        text = delta.get("text")
        return text if isinstance(text, str) else ""
    if "text" in delta and isinstance(delta["text"], str):
        return delta["text"]
    return ""


def _extract_content_block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, list):
        return "".join(_extract_content_block_text(item) for item in block)
    if not isinstance(block, dict):
        return ""

    block_type = str(block.get("type", "")).lower()
    if block_type in _IGNORED_BLOCK_TYPES:
        return ""
    if block_type in _TEXT_BLOCK_TYPES:
        text = block.get("text")
        return text if isinstance(text, str) else ""

    if "text" in block and isinstance(block["text"], str):
        return block["text"]

    nested_content = block.get("content")
    if nested_content is not None:
        return _extract_content_block_text(nested_content)

    return ""
