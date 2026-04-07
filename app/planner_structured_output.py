"""
Structured planner-output extraction for Claude stream transcripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass
class PlannerStructuredOutput:
    """Canonical planner transport extraction result."""

    structured_payload: dict[str, Any] | None = None
    structured_payload_source: str = "none"
    rendered_text_clean: str = ""
    raw_stream_transcript: str = ""
    transport_errors: list[str] = field(default_factory=list)
    structured_delta_bytes: int = 0
    structured_payload_bytes: int = 0
    saw_structured_output_activity: bool = False
    saw_tool_result_success: bool = False
    transcript_complete: bool = True
    truncation_detected: bool = False
    failure_detail: str = ""


def extract_planner_structured_output(
    raw_stream_transcript: str,
    *,
    rendered_text: str = "",
) -> PlannerStructuredOutput:
    """Extract the structured planner payload from Claude stream-json output."""
    result = PlannerStructuredOutput(
        rendered_text_clean=rendered_text,
        raw_stream_transcript=raw_stream_transcript,
    )
    if not raw_stream_transcript.strip():
        return _fallback_to_rendered_text(result)

    delta_fragments: list[str] = []
    for payload in _iter_stream_payloads(raw_stream_transcript, result):
        _consume_payload(payload, result, delta_fragments)

    if result.structured_payload is not None:
        return _finalize_payload(result, result.structured_payload_source)

    if delta_fragments:
        delta_json = "".join(delta_fragments)
        try:
            decoded = json.loads(delta_json)
        except json.JSONDecodeError as exc:
            result.transport_errors.append(f"input_json_delta decode failed: {exc}")
        else:
            if isinstance(decoded, dict):
                result.structured_payload = decoded
                return _finalize_payload(result, "input_json_delta")
            result.transport_errors.append("input_json_delta decoded to non-object payload")

    if result.saw_structured_output_activity:
        detail = (
            result.failure_detail
            or "StructuredOutput activity detected but no recoverable structured payload was found"
        )
        result.transport_errors.append(detail)

    return _fallback_to_rendered_text(result)


def _iter_stream_payloads(raw_stream_transcript: str, result: PlannerStructuredOutput) -> list[Any]:
    decoder = json.JSONDecoder()
    idx = 0
    payloads: list[Any] = []
    length = len(raw_stream_transcript)

    while idx < length:
        while idx < length and raw_stream_transcript[idx].isspace():
            idx += 1
        if idx >= length:
            break
        if idx == 0 and raw_stream_transcript[idx] not in "{[":
            result.failure_detail = ""
            break
        try:
            payload, end = decoder.raw_decode(raw_stream_transcript, idx)
        except json.JSONDecodeError as exc:
            # If this is the very first event, assume truncated output
            if not payloads:
                result.transcript_complete = False
                result.truncation_detected = True
                tail = raw_stream_transcript[idx:idx + 240]
                result.failure_detail = f"stdout_truncated_mid_event: {exc}"
                result.transport_errors.append(result.failure_detail)
                result.transport_errors.append(f"stdout_tail={tail!r}")
                break
            # Non-first event: the stream is likely truncated mid-event.
            # Record the error and stop — attempting to skip forward and
            # re-parse is unsafe because a { inside a JSON string value can
            # form a syntactically valid but semantically wrong event.
            result.transcript_complete = False
            result.truncation_detected = True
            tail = raw_stream_transcript[idx:idx + 240]
            result.failure_detail = f"stdout_truncated_mid_event: {exc}"
            result.transport_errors.append(result.failure_detail)
            result.transport_errors.append(f"stdout_tail={tail!r}")
            break
        payloads.append(payload)
        idx = end

    return payloads


def _consume_payload(
    payload: Any,
    result: PlannerStructuredOutput,
    delta_fragments: list[str],
) -> None:
    if not isinstance(payload, dict):
        return

    payload_type = str(payload.get("type", "")).lower()
    if payload_type == "assistant":
        _consume_assistant_message(payload.get("message"), result)
        return

    if payload_type == "stream_event":
        _consume_stream_event(payload.get("event"), result, delta_fragments)
        return

    if payload_type == "user":
        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if str(block.get("type", "")).lower() == "tool_result":
                        tool_text = str(block.get("content", ""))
                        if tool_text.strip() == "Structured output provided successfully":
                            result.saw_tool_result_success = True


def _consume_assistant_message(message: Any, result: PlannerStructuredOutput) -> None:
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if str(block.get("type", "")).lower() != "tool_use":
            continue
        if str(block.get("name", "")).strip() != "StructuredOutput":
            continue
        result.saw_structured_output_activity = True
        tool_input = block.get("input")
        if isinstance(tool_input, dict) and tool_input:
            result.structured_payload = tool_input
            result.structured_payload_source = "tool_use_input"
            return


def _consume_stream_event(
    event: Any,
    result: PlannerStructuredOutput,
    delta_fragments: list[str],
) -> None:
    if not isinstance(event, dict):
        return

    event_type = str(event.get("type", "")).lower()
    if event_type == "content_block_start":
        content_block = event.get("content_block")
        if not isinstance(content_block, dict):
            return
        if str(content_block.get("type", "")).lower() != "tool_use":
            return
        if str(content_block.get("name", "")).strip() != "StructuredOutput":
            return
        result.saw_structured_output_activity = True
        tool_input = content_block.get("input")
        if isinstance(tool_input, dict) and tool_input:
            result.structured_payload = tool_input
            result.structured_payload_source = "tool_use_input"
        return

    if event_type != "content_block_delta":
        return
    delta = event.get("delta")
    if not isinstance(delta, dict):
        return
    if str(delta.get("type", "")).lower() != "input_json_delta":
        return
    partial_json = delta.get("partial_json")
    if not isinstance(partial_json, str):
        return
    result.saw_structured_output_activity = True
    delta_fragments.append(partial_json)
    result.structured_delta_bytes += len(partial_json.encode("utf-8", errors="replace"))


def _fallback_to_rendered_text(result: PlannerStructuredOutput) -> PlannerStructuredOutput:
    rendered = result.rendered_text_clean.strip()
    if not rendered:
        return result
    try:
        from app.result_parser import parse_plan_output

        parsed = parse_plan_output(rendered)
    except Exception as exc:
        result.transport_errors.append(f"rendered_text parse failed: {exc}")
        return result

    if parsed.get("_parse_failed"):
        return result

    result.structured_payload = parsed
    result.structured_payload_source = "rendered_text"
    return _finalize_payload(result, "rendered_text")


def _finalize_payload(result: PlannerStructuredOutput, source: str) -> PlannerStructuredOutput:
    result.structured_payload_source = source
    if result.structured_payload is not None:
        result.structured_payload_bytes = len(
            json.dumps(result.structured_payload, ensure_ascii=False).encode(
                "utf-8",
                errors="replace",
            )
        )
    return result
