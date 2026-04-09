"""
Helpers for parsing Qwen stream-json worker output.
"""

from __future__ import annotations

import json
from typing import Any


def render_qwen_stream_output(transcript: str) -> str:
    """Render the final assistant-visible payload from Qwen stream-json output."""
    final_result = ""
    assistant_text_chunks: list[str] = []

    for payload in iter_qwen_stream_payloads(transcript):
        if not isinstance(payload, dict):
            continue
        event_type = str(payload.get("type", "") or "").lower()
        if event_type == "result":
            result_text = payload.get("result")
            if isinstance(result_text, str) and result_text.strip():
                final_result = result_text.strip()
        elif event_type == "assistant":
            text = _extract_assistant_message_text(payload.get("message"))
            if text:
                assistant_text_chunks.append(text)

    if final_result:
        return final_result
    return "\n".join(chunk for chunk in assistant_text_chunks if chunk).strip()


def iter_qwen_stream_payloads(transcript: str) -> list[Any]:
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


def _extract_assistant_message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "") or "").lower() != "text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "\n".join(chunks).strip()
