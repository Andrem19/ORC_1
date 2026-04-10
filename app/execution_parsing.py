"""
Strict parsing and validation for planner and worker JSON outputs.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict
from typing import Any

from app.execution_models import (
    BaselineRef,
    ExecutionPlan,
    PlanSlice,
    WorkerAction,
    WorkerReportableIssue,
    make_id,
)


class StructuredOutputError(ValueError):
    """Raised when planner/worker output does not match the contract."""


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise StructuredOutputError("empty_output")
    if raw.startswith("```"):
        raw = _strip_code_fence(raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        repaired = _repair_common_json_malformed_output(raw, exc)
        if repaired is not None:
            try:
                payload = json.loads(repaired)
            except json.JSONDecodeError as repaired_exc:
                raise StructuredOutputError(_json_error_code(repaired_exc)) from repaired_exc
        else:
            candidate = _find_balanced_json(raw)
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError as balanced_exc:
                raise StructuredOutputError(_json_error_code(balanced_exc)) from balanced_exc
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        raise StructuredOutputError("top_level_json_must_be_object")
    return payload


def parse_execution_plan_output(text: str) -> ExecutionPlan:
    payload = extract_json_object(text)
    required = {"plan_id", "goal", "baseline_ref", "global_constraints", "slices"}
    missing = sorted(required - set(payload))
    if missing:
        raise StructuredOutputError(f"execution_plan_missing_fields:{','.join(missing)}")
    baseline_raw = payload["baseline_ref"]
    if not isinstance(baseline_raw, dict):
        raise StructuredOutputError("baseline_ref_must_be_object")
    slices_raw = payload["slices"]
    if not isinstance(slices_raw, list) or not slices_raw:
        raise StructuredOutputError("execution_plan_requires_non_empty_slices")
    if len(slices_raw) > 3:
        raise StructuredOutputError("execution_plan_supports_max_3_slices")
    global_constraints = _string_list(payload["global_constraints"], field_name="global_constraints")
    slices = [parse_plan_slice(item) for item in slices_raw]
    parallel_slots = {item.parallel_slot for item in slices}
    if any(slot < 1 or slot > 3 for slot in parallel_slots):
        raise StructuredOutputError("parallel_slot_must_be_between_1_and_3")
    if len(parallel_slots) != len(slices):
        raise StructuredOutputError("parallel_slot_must_be_unique_per_slice")
    plan = ExecutionPlan(
        plan_id=str(payload["plan_id"]).strip() or make_id("plan"),
        goal=str(payload["goal"]).strip(),
        baseline_ref=BaselineRef(**_filter_baseline_fields(baseline_raw)),
        global_constraints=global_constraints,
        slices=slices,
    )
    if not plan.goal:
        raise StructuredOutputError("execution_plan_goal_must_be_non_empty")
    return plan


def parse_plan_slice(payload: Any) -> PlanSlice:
    if not isinstance(payload, dict):
        raise StructuredOutputError("slice_must_be_object")
    required = {
        "slice_id",
        "title",
        "hypothesis",
        "objective",
        "success_criteria",
        "allowed_tools",
        "evidence_requirements",
        "policy_tags",
        "max_turns",
        "max_tool_calls",
        "max_expensive_calls",
        "parallel_slot",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise StructuredOutputError(f"slice_missing_fields:{','.join(missing)}")
    allowed_tools = _string_list(payload["allowed_tools"], field_name="allowed_tools")
    if not allowed_tools:
        raise StructuredOutputError("slice_allowed_tools_must_be_non_empty")
    slice_obj = PlanSlice(
        slice_id=str(payload["slice_id"]).strip() or make_id("slice"),
        title=str(payload["title"]).strip(),
        hypothesis=str(payload["hypothesis"]).strip(),
        objective=str(payload["objective"]).strip(),
        success_criteria=_string_list(payload["success_criteria"], field_name="success_criteria"),
        allowed_tools=allowed_tools,
        evidence_requirements=_string_list(payload["evidence_requirements"], field_name="evidence_requirements"),
        policy_tags=_string_list(payload["policy_tags"], field_name="policy_tags"),
        max_turns=_positive_int(payload["max_turns"], field_name="max_turns"),
        max_tool_calls=_positive_int(payload["max_tool_calls"], field_name="max_tool_calls"),
        max_expensive_calls=_non_negative_int(payload["max_expensive_calls"], field_name="max_expensive_calls"),
        parallel_slot=_positive_int(payload["parallel_slot"], field_name="parallel_slot"),
    )
    if not slice_obj.title or not slice_obj.objective or not slice_obj.hypothesis:
        raise StructuredOutputError("slice_title_hypothesis_objective_must_be_non_empty")
    return slice_obj


def parse_worker_action_output(text: str, *, allowlist: set[str]) -> WorkerAction:
    try:
        payload = extract_json_object(text)
    except StructuredOutputError:
        recovered = _recover_worker_action_payload(text)
        if recovered is not None:
            payload = recovered
        else:
            recovered = _recover_inline_tool_call(text, allowlist=allowlist)
            if recovered is None:
                raise
            payload = recovered
    raw_type = str(payload.get("type", "") or payload.get("action_type", "")).strip()
    if raw_type not in {"tool_call", "checkpoint", "final_report", "abort"}:
        recovered = _recover_worker_action_payload(text)
        if recovered is None:
            raise StructuredOutputError("worker_action_type_invalid")
        payload = recovered
        raw_type = str(payload.get("type", "") or payload.get("action_type", "")).strip()
    tool_name = str(payload.get("tool", "") or "").strip()
    arguments = _dict_or_empty(payload.get("args", payload.get("arguments", {})), field_name="arguments")
    if raw_type == "tool_call":
        arguments = _normalize_tool_arguments(tool_name=tool_name, arguments=arguments)
    summary = str(payload.get("summary", "") or "").strip()
    if raw_type == "abort" and not summary:
        summary = str(payload.get("reason", "") or "").strip()
    if raw_type in {"checkpoint", "final_report"} and not summary:
        summary = str(payload.get("reason", "") or "").strip()
    issues = [
        WorkerReportableIssue(
            summary=str(item.get("summary", "") or "").strip(),
            severity=str(item.get("severity", "medium") or "medium"),
            details=str(item.get("details", "") or "").strip(),
            affected_tool=str(item.get("affected_tool", "") or "").strip(),
            category=str(item.get("category", "runtime") or "runtime"),
        )
        for item in payload.get("reportable_issues", []) or []
        if isinstance(item, dict) and str(item.get("summary", "") or "").strip()
    ]
    action = WorkerAction(
        action_id=str(payload.get("action_id", "") or make_id("action")),
        action_type=raw_type,
        tool=tool_name,
        arguments=arguments,
        reason=str(payload.get("reason", "") or "").strip(),
        expected_evidence=_string_list(payload.get("expected_evidence", []), field_name="expected_evidence"),
        status=str(payload.get("status", "") or "").strip(),
        summary=summary,
        facts=_dict_or_empty(payload.get("facts", {}), field_name="facts"),
        artifacts=_string_list(payload.get("artifacts", []), field_name="artifacts"),
        pending_questions=_string_list(payload.get("pending_questions", []), field_name="pending_questions"),
        reportable_issues=issues,
        verdict=str(payload.get("verdict", "") or "").strip(),
        key_metrics=_dict_or_empty(payload.get("key_metrics", {}), field_name="key_metrics"),
        findings=_string_list(payload.get("findings", []), field_name="findings"),
        rejected_findings=_string_list(payload.get("rejected_findings", []), field_name="rejected_findings"),
        next_actions=_string_list(payload.get("next_actions", []), field_name="next_actions"),
        risks=_string_list(payload.get("risks", []), field_name="risks"),
        evidence_refs=_string_list(payload.get("evidence_refs", []), field_name="evidence_refs"),
        confidence=_float_or_zero(payload.get("confidence", 0.0)),
        reason_code=str(payload.get("reason_code", "") or "").strip(),
        retryable=bool(payload.get("retryable", False)),
    )
    _validate_worker_action(action, allowlist=allowlist)
    return action


def worker_action_to_dict(action: WorkerAction) -> dict[str, Any]:
    return asdict(action)


def _validate_worker_action(action: WorkerAction, *, allowlist: set[str]) -> None:
    if action.action_type == "tool_call":
        if not action.tool:
            raise StructuredOutputError("tool_call_requires_tool")
        if action.tool.startswith("mcp__dev_space1__"):
            raise StructuredOutputError(f"tool_prefixed_namespace_forbidden:{action.tool}")
        if action.tool not in allowlist:
            raise StructuredOutputError(f"tool_not_in_allowlist:{action.tool}")
        if not isinstance(action.arguments, dict):
            raise StructuredOutputError("tool_call_arguments_must_be_object")
        if not action.reason:
            raise StructuredOutputError("tool_call_requires_reason")
        return
    if action.action_type == "checkpoint":
        if action.status not in {"partial", "complete", "blocked"}:
            raise StructuredOutputError("checkpoint_status_invalid")
        if not action.summary:
            raise StructuredOutputError("checkpoint_requires_summary")
        return
    if action.action_type == "final_report":
        if not action.summary or not action.verdict:
            raise StructuredOutputError("final_report_requires_summary_and_verdict")
        return
    if action.action_type == "abort":
        if not action.reason_code or not action.summary:
            raise StructuredOutputError("abort_requires_reason_code_and_summary")


def _dict_or_empty(value: Any, *, field_name: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise StructuredOutputError(f"{field_name}_must_be_object")
    return dict(value)


def _normalize_tool_arguments(*, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    for _ in range(3):
        updated = _normalize_tool_arguments_once(tool_name=tool_name, arguments=normalized)
        if updated == normalized:
            return updated
        normalized = updated
    return normalized


def _normalize_tool_arguments_once(*, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    nested_tool_value = arguments.get("tool")
    nested_tool = str(nested_tool_value or "").strip() if not isinstance(nested_tool_value, dict) else ""
    nested_arguments = arguments.get("arguments")
    if isinstance(nested_tool_value, dict):
        nested_payload = dict(nested_tool_value)
        nested_payload_tool = str(nested_payload.get("tool", "") or "").strip()
        nested_payload_arguments = nested_payload.get("arguments")
        if isinstance(nested_payload_arguments, dict):
            if nested_payload_tool and nested_payload_tool != tool_name:
                raise StructuredOutputError("tool_argument_wrapper_conflicts_with_tool_name")
            return dict(nested_payload_arguments)
        if set(arguments) == {"tool"}:
            return nested_payload
        return nested_payload
    if not isinstance(nested_arguments, dict):
        if len(arguments) == 1 and tool_name in arguments and isinstance(arguments.get(tool_name), dict):
            return dict(arguments.get(tool_name) or {})
        return arguments
    if nested_tool and nested_tool != tool_name:
        raise StructuredOutputError("tool_argument_wrapper_conflicts_with_tool_name")
    if set(arguments) <= {"tool", "arguments"}:
        return dict(nested_arguments)
    return arguments


def _string_list(value: Any, *, field_name: str) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise StructuredOutputError(f"{field_name}_must_be_list")
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _positive_int(value: Any, *, field_name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise StructuredOutputError(f"{field_name}_must_be_int") from exc
    if resolved <= 0:
        raise StructuredOutputError(f"{field_name}_must_be_positive")
    return resolved


def _non_negative_int(value: Any, *, field_name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise StructuredOutputError(f"{field_name}_must_be_int") from exc
    if resolved < 0:
        raise StructuredOutputError(f"{field_name}_must_be_non_negative")
    return resolved


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _strip_code_fence(raw: str) -> str:
    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


_INLINE_TOOL_CALL_PATTERN = re.compile(r"`([a-zA-Z_][a-zA-Z0-9_]*\([^`]*\))`")


def _recover_inline_tool_call(text: str, *, allowlist: set[str]) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    matches = _INLINE_TOOL_CALL_PATTERN.findall(raw)
    if not matches:
        return None
    candidate = matches[-1].strip()
    try:
        parsed = ast.parse(candidate, mode="eval")
    except SyntaxError:
        return None
    call = parsed.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
        return None
    tool_name = str(call.func.id or "").strip()
    if tool_name not in allowlist:
        return None
    arguments: dict[str, Any] = {}
    for item in call.keywords:
        if item.arg is None:
            return None
        try:
            arguments[item.arg] = ast.literal_eval(item.value)
        except Exception:
            return None
    return {
        "type": "tool_call",
        "tool": tool_name,
        "arguments": arguments,
        "reason": "Recovered from free-form worker output after the model failed to return strict JSON.",
        "expected_evidence": [],
    }


def _recover_worker_action_payload(text: str) -> dict[str, Any] | None:
    candidates = _extract_json_object_candidates(text)
    for candidate in reversed(candidates):
        payload = _normalize_recovered_payload(candidate)
        if payload is None:
            continue
        raw_type = str(payload.get("type", "") or payload.get("action_type", "")).strip()
        if raw_type in {"tool_call", "checkpoint", "final_report", "abort"}:
            return payload
    return None


def _find_balanced_json(raw: str) -> str:
    object_start = raw.find("{")
    array_start = raw.find("[")
    starts = [index for index in (object_start, array_start) if index >= 0]
    if not starts:
        raise StructuredOutputError("json_object_not_found")
    start = min(starts)
    opener = raw[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                in_string = False
            continue
        if char == "\"":
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return raw[start:index + 1]
    raise StructuredOutputError("balanced_json_object_not_found")


def _extract_json_object_candidates(text: str) -> list[Any]:
    raw = (text or "").strip()
    if not raw:
        return []
    candidates: list[Any] = []
    seen: set[str] = set()
    for start in range(len(raw)):
        if raw[start] not in "{[":
            continue
        try:
            candidate_text = _find_balanced_json(raw[start:])
            payload = json.loads(candidate_text)
        except (StructuredOutputError, json.JSONDecodeError):
            continue
        key = json.dumps(payload, sort_keys=True, ensure_ascii=False) if isinstance(payload, (dict, list)) else repr(payload)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(payload)
    return candidates


def _normalize_recovered_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    return payload


def _repair_common_json_malformed_output(raw: str, exc: json.JSONDecodeError) -> str | None:
    """
    Repair near-valid model JSON without paying a heavy parsing cost on the hot path.

    The common failure we see in worker outputs is an extra unmatched closer just
    before the model resumes emitting additional top-level fields:

        {"type": "...", "arguments": {...}},"reason":"..."}

    In that case json.loads() fails with Extra data. We remove the redundant
    closer nearest to the extra-data boundary and retry a few times.
    """

    if exc.msg != "Extra data":
        return None
    candidate = raw
    current_exc = exc
    for _ in range(3):
        redundant_index = _find_redundant_closer_before_extra(candidate, current_exc.pos)
        if redundant_index is None:
            return None
        candidate = candidate[:redundant_index] + candidate[redundant_index + 1 :]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError as retry_exc:
            current_exc = retry_exc
            continue
    return None


def _json_error_code(exc: json.JSONDecodeError) -> str:
    message = str(exc).strip()
    if not message:
        return "json_decode_error"
    return f"json_decode_error:{message}"


def _find_redundant_closer_before_extra(raw: str, extra_pos: int) -> int | None:
    index = min(max(extra_pos - 1, 0), len(raw) - 1)
    while index >= 0 and raw[index].isspace():
        index -= 1
    if index < 0:
        return None
    if raw[index] == ",":
        index -= 1
        while index >= 0 and raw[index].isspace():
            index -= 1
    if index < 0 or raw[index] not in "}]":
        return None
    return index


def _filter_baseline_fields(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {"snapshot_id", "version", "symbol", "anchor_timeframe", "execution_timeframe"}
    filtered = {key: value[key] for key in allowed if key in value}
    if "snapshot_id" not in filtered or "version" not in filtered:
        raise StructuredOutputError("baseline_ref_requires_snapshot_id_and_version")
    return filtered
