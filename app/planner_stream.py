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


def inspect_stream_transcript(transcript: str) -> dict[str, Any]:
    """Classify a Claude stream-json transcript for planner watchdog logic."""
    has_visible_text = False
    has_thinking = False
    event_count = 0
    tool_use_name = ""
    trailing_buffer = ""

    for payload in iter_stream_payloads(transcript):
        event_count += 1
        text = extract_stream_text(payload)
        if text:
            has_visible_text = True
        tool_name = _extract_tool_use_name(payload)
        if tool_name and not tool_use_name:
            tool_use_name = tool_name
        if _payload_has_thinking(payload):
            has_thinking = True

    if transcript and not transcript.endswith("\n"):
        trailing_buffer = transcript.rsplit("\n", 1)[-1]

    return {
        "event_count": event_count,
        "has_visible_text": has_visible_text,
        "has_thinking": has_thinking,
        "tool_use_name": tool_use_name,
        "thinking_only": has_thinking and not has_visible_text,
        "raw_stdout_tail": transcript[-1000:],
        "stream_buffer_tail": trailing_buffer[-500:],
    }


def extract_markdown_from_thinking_transcript(transcript: str) -> str:
    """Recover plan markdown when a backend emits it inside thinking deltas."""
    thinking_text = "".join(_extract_thinking_text(payload) for payload in iter_stream_payloads(transcript))
    if not thinking_text:
        return ""
    return extract_plan_markdown_candidate(thinking_text)


def extract_plan_markdown_candidate(text: str) -> str:
    """Extract the most plausible standalone plan block from a larger text blob."""
    markers = [idx for idx in range(len(text)) if text.startswith("# Plan v", idx)]
    if not markers:
        return ""

    candidates: list[str] = []
    for marker in markers:
        candidate = _trim_plan_tail(text[marker:])
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return ""

    def _score(item: str) -> tuple[int, int, int]:
        etap_count = item.count("\n## ETAP ")
        section_count = sum(1 for section in (
            "## Status and Frame",
            "## Goal",
            "## Baseline",
            "## Research Principles",
            "## dev_space1 Capabilities",
        ) if section in item)
        return (etap_count, section_count, len(item))

    return max(candidates, key=_score).strip()


def iter_stream_payloads(transcript: str) -> list[Any]:
    """Parse line-delimited stream-json transcript into payload objects."""
    payloads: list[Any] = []
    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return payloads


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
    if event_type == "stream_event":
        return extract_stream_text(payload.get("event"))
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


def _trim_plan_tail(text: str) -> str:
    lines = text.splitlines()
    collected: list[str] = []
    etap_count = 0
    seen_terminal_table_row_after_etap3 = False
    markdown_prefixes = (
        "#",
        "##",
        "**Goal**:",
        "Goal:",
        "Completion criteria",
        "|",
        "- ",
        "* ",
    )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## ETAP "):
            etap_count += 1
        if etap_count >= 3 and _looks_like_terminal_table_row(stripped):
            seen_terminal_table_row_after_etap3 = True
        if (
            seen_terminal_table_row_after_etap3
            and stripped
            and not stripped.startswith(markdown_prefixes)
            and not _looks_like_numbered_step(stripped)
        ):
            break
        collected.append(line)

    return "\n".join(collected).strip()


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


def _looks_like_numbered_step(line: str) -> bool:
    if not line:
        return False
    digits = []
    for ch in line:
        if ch.isdigit():
            digits.append(ch)
            continue
        break
    if not digits:
        return False
    return line[len(digits):].startswith(". ")


def _looks_like_terminal_table_row(line: str) -> bool:
    if not (line.startswith("|") and line.endswith("|")):
        return False
    inner = line.strip("|").strip()
    if not inner:
        return False
    return not all(ch in "-: |" for ch in inner)


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


def _extract_thinking_text(payload: Any) -> str:
    if isinstance(payload, str):
        return ""
    if isinstance(payload, list):
        return "".join(_extract_thinking_text(item) for item in payload)
    if not isinstance(payload, dict):
        return ""

    event_type = str(payload.get("type", "")).lower()
    if event_type == "stream_event":
        return _extract_thinking_text(payload.get("event"))
    if event_type == "content_block_start":
        return _extract_thinking_block(payload.get("content_block"))
    if event_type == "content_block_delta":
        return _extract_thinking_delta(payload.get("delta"))

    fragments: list[str] = []
    for key in ("event", "message", "content", "content_block", "delta"):
        value = payload.get(key)
        if value is not None:
            fragments.append(_extract_thinking_text(value))
    return "".join(fragments)


def _extract_thinking_block(block: Any) -> str:
    if isinstance(block, list):
        return "".join(_extract_thinking_block(item) for item in block)
    if not isinstance(block, dict):
        return ""
    if str(block.get("type", "")).lower() == "thinking":
        thinking = block.get("thinking")
        return thinking if isinstance(thinking, str) else ""
    fragments: list[str] = []
    for key in ("content", "content_block", "message", "event", "delta"):
        value = block.get(key)
        if value is not None:
            fragments.append(_extract_thinking_text(value))
    return "".join(fragments)


def _extract_thinking_delta(delta: Any) -> str:
    if isinstance(delta, list):
        return "".join(_extract_thinking_delta(item) for item in delta)
    if not isinstance(delta, dict):
        return ""
    if str(delta.get("type", "")).lower() == "thinking_delta":
        thinking = delta.get("thinking")
        return thinking if isinstance(thinking, str) else ""
    return ""


def _extract_tool_use_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    event_type = str(payload.get("type", "")).lower()
    if event_type == "content_block_start":
        content_block = payload.get("content_block")
        if isinstance(content_block, dict) and str(content_block.get("type", "")).lower() == "tool_use":
            name = content_block.get("name")
            return name if isinstance(name, str) else "tool_use"
    if event_type == "assistant":
        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and str(item.get("type", "")).lower() == "tool_use":
                        name = item.get("name")
                        return name if isinstance(name, str) else "tool_use"
    event = payload.get("event")
    if isinstance(event, dict):
        stop_reason = str(event.get("stop_reason", "")).lower()
        if stop_reason == "tool_use":
            return "tool_use"
        return _extract_tool_use_name(event)
    delta = payload.get("delta")
    if isinstance(delta, dict):
        stop_reason = str(delta.get("stop_reason", "")).lower()
        if stop_reason == "tool_use":
            return "tool_use"
    return ""


def _payload_has_thinking(payload: Any) -> bool:
    if isinstance(payload, dict):
        event_type = str(payload.get("type", "")).lower()
        if event_type == "content_block_start":
            content_block = payload.get("content_block")
            if isinstance(content_block, dict) and str(content_block.get("type", "")).lower() == "thinking":
                return True
        if event_type == "content_block_delta":
            delta = payload.get("delta")
            if isinstance(delta, dict) and str(delta.get("type", "")).lower() == "thinking_delta":
                return True
        for key in ("event", "message", "content", "content_block", "delta"):
            value = payload.get(key)
            if _payload_has_thinking(value):
                return True
        return False
    if isinstance(payload, list):
        return any(_payload_has_thinking(item) for item in payload)
    return False
