"""
Helpers for parsing Claude stream-json planner output.
"""

from __future__ import annotations

import json
from typing import Any


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
            rendered.append(line)
            continue
        event_count += 1
        text = extract_stream_text(payload)
        if text:
            rendered.append(text)

    return "".join(rendered), buffer, event_count


def extract_stream_text(payload: Any) -> str:
    """Best-effort extraction of user-visible text from one stream-json event."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return "".join(extract_stream_text(item) for item in payload)
    if not isinstance(payload, dict):
        return ""

    result = payload.get("result")
    if isinstance(result, str):
        return result

    event_type = str(payload.get("type", "")).lower()
    if event_type in {"message_stop", "content_block_stop", "ping", "system"}:
        return ""

    for key in ("delta", "message", "content", "content_block", "completion", "output"):
        if key in payload:
            nested = extract_stream_text(payload[key])
            if nested:
                return nested

    text = payload.get("text")
    if isinstance(text, str):
        return text

    if isinstance(payload.get("content"), list):
        parts: list[str] = []
        for item in payload["content"]:
            if isinstance(item, dict):
                item_text = item.get("text")
                if isinstance(item_text, str):
                    parts.append(item_text)
                    continue
            parts.append(extract_stream_text(item))
        return "".join(parts)

    parts: list[str] = []
    for value in payload.values():
        extracted = extract_stream_text(value)
        if extracted:
            parts.append(extracted)
    return "".join(parts)
