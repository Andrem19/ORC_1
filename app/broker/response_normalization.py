"""
Normalize MCP call results against the public FOUNDATION ToolResponse envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_RUNNING_STATUSES = {
    "queued",
    "pending",
    "running",
    "started",
    "in_progress",
    "running_compute",
    "publishing",
    "publishing_result",
    "persisting",
}


@dataclass
class NormalizedToolPayload:
    raw_payload: dict[str, Any]
    protocol_payload: dict[str, Any]
    tool_response: dict[str, Any]
    data: dict[str, Any] | list[Any] | None
    tool_status: str
    operation_status: str
    summary: str
    warnings: list[str] = field(default_factory=list)
    error_class: str = ""
    error_message: str = ""
    operation_ref: str = ""
    resume_tool: str = ""
    resume_token: str = ""
    resume_arguments: dict[str, Any] = field(default_factory=dict)
    key_facts: dict[str, Any] = field(default_factory=dict)
    artifact_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        if self.tool_status:
            return self.tool_status.lower() == "ok" and not self.error_class
        return not self.error_class

    @property
    def should_resume(self) -> bool:
        return bool(self.resume_tool and self.resume_arguments and self.operation_status.lower() in _RUNNING_STATUSES)


def normalize_tool_payload(raw_payload: dict[str, Any]) -> NormalizedToolPayload:
    protocol_payload = _primary_protocol_payload(raw_payload)
    tool_response = _extract_tool_response(protocol_payload)
    data = _extract_data(tool_response)
    tool_status = str(tool_response.get("status", "") or "").strip().lower()
    summary = _extract_summary(tool_response=tool_response, data=data)
    warnings = _extract_warnings(tool_response=tool_response, data=data)
    error_class, error_message = _extract_error(tool_response=tool_response, protocol_payload=protocol_payload)
    operation_status = _extract_operation_status(data)
    resume_tool, resume_arguments, resume_token = _extract_resume_descriptor(data)
    operation_ref = resume_token or _extract_operation_ref(data)
    return NormalizedToolPayload(
        raw_payload=raw_payload,
        protocol_payload=protocol_payload,
        tool_response=tool_response,
        data=data,
        tool_status=tool_status,
        operation_status=operation_status,
        summary=summary,
        warnings=warnings,
        error_class=error_class,
        error_message=error_message,
        operation_ref=operation_ref,
        resume_tool=resume_tool,
        resume_token=resume_token,
        resume_arguments=resume_arguments,
        key_facts=_extract_key_facts(data),
        artifact_ids=_extract_artifact_ids(data),
    )


def _primary_protocol_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        return {}
    autopoll = raw_payload.get("autopoll", []) or []
    if isinstance(autopoll, list):
        for item in reversed(autopoll):
            if isinstance(item, dict):
                return item
    initial = raw_payload.get("initial")
    if isinstance(initial, dict):
        return initial
    return raw_payload


def _extract_tool_response(protocol_payload: dict[str, Any]) -> dict[str, Any]:
    structured = protocol_payload.get("structuredContent") or protocol_payload.get("structured_content")
    if isinstance(structured, dict):
        return structured
    if _looks_like_tool_response(protocol_payload):
        return protocol_payload
    return {"status": "ok", "message": "", "data": protocol_payload}


def _looks_like_tool_response(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("status", "message", "data", "warnings", "error"))


def _extract_data(tool_response: dict[str, Any]) -> dict[str, Any] | list[Any] | None:
    data = tool_response.get("data")
    if isinstance(data, (dict, list)):
        return data
    return None


def _extract_summary(*, tool_response: dict[str, Any], data: dict[str, Any] | list[Any] | None) -> str:
    tool_name = str(tool_response.get("tool_name", "") or "").strip()
    specialized = _extract_specialized_summary(tool_name=tool_name, data=data)
    if specialized:
        return specialized[:600]
    if isinstance(data, dict):
        for key in ("summary", "message", "detail"):
            value = str(data.get(key, "") or "").strip()
            if value:
                return value[:600]
    message = str(tool_response.get("message", "") or "").strip()
    if message:
        return message[:600]
    if isinstance(data, dict):
        for key in ("status", "final_status", "wait_satisfied_by_status"):
            value = str(data.get(key, "") or "").strip()
            if value:
                return value[:600]
    return ""


def _extract_specialized_summary(*, tool_name: str, data: dict[str, Any] | list[Any] | None) -> str:
    if not isinstance(data, dict):
        return ""
    view = str(data.get("view", "") or "").strip().lower()
    if tool_name == "features_custom" and view == "contract":
        contract = data.get("contract")
        if isinstance(contract, dict):
            validate_fields = contract.get("required_fields_by_action", {}).get("validate", [])
            publish_fields = contract.get("required_fields_by_action", {}).get("publish", [])
            validate_text = ", ".join(str(item) for item in validate_fields[:6] if str(item).strip())
            publish_text = ", ".join(str(item) for item in publish_fields[:6] if str(item).strip())
            if validate_text or publish_text:
                return (
                    "Custom feature contract loaded. "
                    f"validate requires [{validate_text}] "
                    f"publish requires [{publish_text}]."
                ).strip()
    return ""


def _extract_warnings(*, tool_response: dict[str, Any], data: dict[str, Any] | list[Any] | None) -> list[str]:
    warnings: list[str] = []
    raw_tool_warnings = tool_response.get("warnings", []) or []
    if isinstance(raw_tool_warnings, list):
        warnings.extend(str(item).strip() for item in raw_tool_warnings if str(item).strip())
    warnings.extend(_collect_nested_warnings(data))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in warnings:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped[:20]


def _collect_nested_warnings(value: Any, *, depth: int = 0) -> list[str]:
    if depth > 4:
        return []
    if isinstance(value, dict):
        warnings: list[str] = []
        for key, nested in value.items():
            if key in {"warnings", "important_warnings"} and isinstance(nested, list):
                warnings.extend(str(item).strip() for item in nested if str(item).strip())
            else:
                warnings.extend(_collect_nested_warnings(nested, depth=depth + 1))
        return warnings
    if isinstance(value, list):
        warnings: list[str] = []
        for item in value[:50]:
            warnings.extend(_collect_nested_warnings(item, depth=depth + 1))
        return warnings
    return []


def _extract_error(*, tool_response: dict[str, Any], protocol_payload: dict[str, Any]) -> tuple[str, str]:
    if protocol_payload.get("isError"):
        return "tool_error", str(protocol_payload.get("error", "") or "tool_error")
    error = tool_response.get("error")
    tool_status = str(tool_response.get("status", "") or "").strip().lower()
    if isinstance(error, dict):
        error_class = str(error.get("class", "") or error.get("code", "") or "")
        if error_class:
            return error_class, str(error.get("message", "") or error_class)
        return "server_error", str(error.get("message", "") or "server_error")
    if error:
        error_text = str(error).strip()
        return _classify_error_text(error_text), error_text
    if tool_status == "error":
        message = str(tool_response.get("message", "") or "tool_error")
        return _classify_error_text(message), message
    return "", ""


def _classify_error_text(error_text: str) -> str:
    lower = error_text.lower()
    if "timeout" in lower:
        return "timeout"
    if "transport" in lower or "connection" in lower or "session" in lower:
        return "transport_error"
    return "server_error"


def _extract_operation_status(data: dict[str, Any] | list[Any] | None) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("final_status", "wait_satisfied_by_status", "status", "state"):
        value = str(data.get(key, "") or "").strip()
        if value:
            return value
    for nested_key in ("operation", "job", "run"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            for key in ("status", "state"):
                value = str(nested.get(key, "") or "").strip()
                if value:
                    return value
    return ""


def _extract_resume_descriptor(data: dict[str, Any] | list[Any] | None) -> tuple[str, dict[str, Any], str]:
    if not isinstance(data, dict):
        return "", {}, ""
    resume_tool = str(data.get("resume_tool", "") or "").strip()
    resume_arguments = data.get("resume_arguments", {}) or {}
    resume_token = str(data.get("resume_token", "") or "").strip()
    if resume_tool and isinstance(resume_arguments, dict):
        return resume_tool, dict(resume_arguments), resume_token
    return "", {}, resume_token


def _extract_operation_ref(data: dict[str, Any] | list[Any] | None) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("operation_id", "job_id", "run_id"):
        value = str(data.get(key, "") or "").strip()
        if value:
            return value
    for nested_key in ("operation", "job", "run"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            for key in ("operation_id", "job_id", "run_id"):
                value = str(nested.get(key, "") or "").strip()
                if value:
                    return value
    return ""


def _extract_key_facts(data: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    facts: dict[str, Any] = {}
    for key, value in data.items():
        if key.endswith("_id") or key in {"summary", "status", "final_status", "wait_timeout", "wait_satisfied", "count", "rows", "tool_count"}:
            facts[key] = value
    for nested_key in ("operation", "job", "run", "dataset", "result", "operation_result"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            for key, value in nested.items():
                if key.endswith("_id") or key in {"status", "scheduler_state", "queue_id", "row_count"}:
                    facts[f"{nested_key}.{key}"] = value
    return facts


def _extract_artifact_ids(data: dict[str, Any] | list[Any] | None) -> list[str]:
    if not isinstance(data, dict):
        return []
    artifacts: list[str] = []
    for key, value in data.items():
        if key.endswith("_path") or key in {"resume_token", "operation_id", "job_id", "run_id", "artifact_path"}:
            text = str(value).strip()
            if text:
                artifacts.append(text)
    for nested_key in ("operation", "job", "run"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            for key, value in nested.items():
                if key.endswith("_path") or key in {"operation_id", "job_id", "run_id", "queue_id"}:
                    text = str(value).strip()
                    if text:
                        artifacts.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in artifacts:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


__all__ = ["NormalizedToolPayload", "normalize_tool_payload"]
