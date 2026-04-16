"""
Semantic progress heuristics for direct slices.
"""

from __future__ import annotations

import json
from typing import Any

from app.services.direct_execution.handle_hygiene import durable_created_id_values

_MUTATING_ACTIONS = {
    "add",
    "apply",
    "archive",
    "build",
    "capture",
    "clone",
    "create",
    "delete",
    "materialize",
    "publish",
    "record_attempt",
    "refresh",
    "remove",
    "rename",
    "run",
    "save_version",
    "start",
    "sync",
    "train",
    "update",
    "validate",
}
_HANDLE_FIELDS = ("project_id", "job_id", "run_id", "snapshot_id", "operation_id", "branch_id")
_STRICT_AUTO_FINALIZE_PROFILES = frozenset({"research_setup", "research_shortlist"})


def tool_call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    action = str(arguments.get("action") or "").strip().lower()
    if not action and not any(key in arguments for key in ("record", "payload", "project", "coordinates")):
        return ""
    is_readonly = action and action not in _MUTATING_ACTIONS
    payload = {"tool": str(tool_name or "").strip(), "action": action}
    for field_name in _HANDLE_FIELDS:
        value = str(arguments.get(field_name) or "").strip()
        if value:
            payload[field_name] = value
    for field_name in ("kind", "name", "view", "scope", "wait"):
        value = str(arguments.get(field_name) or "").strip()
        if value:
            payload[field_name] = value
    record = arguments.get("record")
    if isinstance(record, dict):
        for field_name in ("title", "summary"):
            value = str(record.get(field_name) or "").strip()
            if value:
                payload[f"record.{field_name}"] = value
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return ("ro:" + encoded) if is_readonly else encoded


def compact_tool_result_message(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    success_criteria: list[str],
    runtime_profile: str = "",
    allowed_tools: set[str] | list[str] | None = None,
) -> str:
    normalized = {
        "tool": tool_name,
        "ok": bool(result_payload.get("ok", False)) and "error" not in result_payload,
        "status": _extract_status(result_payload),
        "summary": _extract_summary(result_payload),
        "ids": _extract_ids(result_payload),
        "warnings": _extract_warnings(result_payload)[:4],
        "errors": _extract_errors(result_payload)[:4],
    }
    guidance = _runtime_guidance(
        tool_name=tool_name,
        arguments=arguments,
        result_payload=result_payload,
        success_criteria=success_criteria,
        runtime_profile=runtime_profile,
        allowed_tools=allowed_tools,
    )
    if guidance:
        normalized["runtime_guidance"] = guidance
    return json.dumps(normalized, ensure_ascii=False)


def should_auto_finalize_research_slice(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    success_criteria: list[str],
    allowed_tools: set[str],
    required_output_facts: list[str],
    prior_contract_issue: bool,
    runtime_profile: str = "",
    finalization_mode: str = "",
) -> bool:
    del prior_contract_issue
    effective_finalization_mode = str(finalization_mode or "").strip() or ("fact_based" if required_output_facts else "none")
    if effective_finalization_mode != "fact_based":
        return False
    if not _supports_fact_based_auto_finalize(runtime_profile=runtime_profile, allowed_tools=allowed_tools):
        return False
    if _extract_status(result_payload) in {"error", "failed"}:
        return False
    derived_facts = derive_research_write_facts(
        arguments=arguments,
        result_payload=result_payload,
        runtime_profile=runtime_profile,
        tool_name=tool_name,
    )
    if any(_is_missing_fact(derived_facts.get(key)) for key in required_output_facts):
        return False
    if not _is_mutating_success(arguments=arguments, result_payload=result_payload):
        return False
    if derived_facts:
        return True
    summary_text = " ".join(
        [
            _extract_summary(result_payload),
            json.dumps(_record_metadata(arguments), ensure_ascii=False),
        ]
    ).lower()
    normalized_criteria = [str(item).strip().lower() for item in success_criteria if str(item).strip()]
    return bool(normalized_criteria and any(item in summary_text for item in normalized_criteria))


def build_auto_final_report(
    *,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    success_criteria: list[str],
    runtime_profile: str = "",
    tool_name: str = "",
) -> str:
    summary = _record_summary(arguments) or _extract_summary(result_payload) or "Direct slice completed after successful mutating tool call."
    metadata = _record_metadata(arguments)
    facts = derive_research_write_facts(
        arguments=arguments,
        result_payload=result_payload,
        runtime_profile=runtime_profile,
        tool_name=tool_name,
    )
    facts["execution.kind"] = "direct"
    facts["direct.auto_finalized_after_fact_based_write"] = True
    if tool_name:
        facts["direct.write_tool"] = tool_name
    ids = _extract_ids(result_payload)
    if ids:
        facts["direct.write_ids"] = ids
    findings = [summary]
    if success_criteria:
        findings.extend(f"Success criterion satisfied: {item}" for item in success_criteria[:3])
    payload = {
        "type": "final_report",
        "summary": summary,
        "verdict": "COMPLETE",
        "findings": findings,
        "facts": facts,
        "evidence_refs": list(metadata.get("evidence_refs") or ids),
        "confidence": 0.88,
    }
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def build_semantic_loop_abort(*, summary: str) -> str:
    payload = {
        "type": "abort",
        "summary": summary,
        "reason_code": "direct_semantic_loop_detected",
        "retryable": False,
    }
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def build_watchdog_checkpoint(*, summary: str, reason_code: str, facts: dict[str, Any] | None = None) -> str:
    payload = {
        "type": "checkpoint",
        "status": "blocked",
        "summary": summary,
        "reason_code": reason_code,
        "facts": {"execution.kind": "direct"} | dict(facts or {}),
    }
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def derive_research_write_facts(
    *,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    runtime_profile: str = "",
    tool_name: str = "",
) -> dict[str, Any]:
    del tool_name
    facts: dict[str, Any] = {}
    structured = _extract_structured(result_payload.get("payload"))
    data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
    record = data.get("record") if isinstance(data.get("record"), dict) else {}
    record_refs = data.get("record_refs") if isinstance(data.get("record_refs"), dict) else {}
    for field_name in _HANDLE_FIELDS:
        value = _first_string(arguments, field_name) or _first_string(data, field_name) or _first_string(structured, field_name)
        if not value:
            continue
        facts[field_name] = value
        if field_name == "project_id":
            facts["research.project_id"] = value
    refs = _coerce_refs(_record_metadata(arguments).get("evidence_refs"))
    ids = _extract_ids(result_payload)
    if refs:
        facts["direct.evidence_refs"] = refs
    if ids:
        facts["direct.created_ids"] = ids
        hypothesis_ids = [item for item in ids if "node" in str(item).lower()]
        if hypothesis_ids:
            facts["research.hypothesis_refs"] = hypothesis_ids
    elif refs:
        hypothesis_refs = [item for item in refs if "node" in str(item).lower()]
        if hypothesis_refs:
            facts["research.hypothesis_refs"] = hypothesis_refs
    memory_node_id = (
        _first_string(record_refs, "memory_node_id")
        or _first_string(data, "memory_node_id")
        or _first_string(record, "node_id")
    )
    if memory_node_id:
        facts["research.memory_node_id"] = memory_node_id
    shortlist = _coerce_shortlist(arguments, record)
    if shortlist:
        facts["research.shortlist_families"] = shortlist
    novelty_justification_present = _coerce_novelty_justification(arguments, record)
    if novelty_justification_present is not None:
        facts["research.novelty_justification_present"] = novelty_justification_present
    return facts


def _supports_fact_based_auto_finalize(*, runtime_profile: str, allowed_tools: set[str]) -> bool:
    normalized_profile = str(runtime_profile or "").strip()
    if normalized_profile in _STRICT_AUTO_FINALIZE_PROFILES:
        return True
    if normalized_profile in {"research_memory", "write_result"}:
        normalized_tools = {str(item or "").strip() for item in allowed_tools if str(item or "").strip()}
        return bool(normalized_tools) and all(_is_research_family_tool(item) for item in normalized_tools)
    return False


def _is_research_family_tool(name: str) -> bool:
    return str(name or "").strip().startswith("research_")


def _runtime_guidance(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    success_criteria: list[str],
    runtime_profile: str = "",
    allowed_tools: set[str] | list[str] | None = None,
) -> str:
    if _is_mutating_success(arguments=arguments, result_payload=result_payload):
        return "If downstream facts are complete, return terminal JSON now instead of repeating the same mutating call."
    if _is_mixed_domain_domain_probe(
        tool_name=tool_name,
        arguments=arguments,
        result_payload=result_payload,
        runtime_profile=runtime_profile,
        allowed_tools=allowed_tools,
    ):
        return (
            "Non-research evidence captured. If this already covers the active criterion, return final_report now "
            "instead of going back to research_memory for wording or context restatement."
        )
    if _is_mixed_domain_contract_context_read(
        tool_name=tool_name,
        arguments=arguments,
        runtime_profile=runtime_profile,
        allowed_tools=allowed_tools,
    ):
        return (
            "Context recovered. Stop repeating research_memory reads and switch to a non-research "
            "contract tool such as features_custom, features_dataset, features_analytics, or models_dataset."
        )
    summary = _extract_summary(result_payload).lower()
    if str(tool_name or "").strip() == "features_analytics" and "requires feature_name" in summary:
        return (
            "Choose one explicit feature_name from prior live evidence before feature analytics. "
            "Do not guess a feature identifier."
        )
    if str(tool_name or "").strip() == "features_custom" and "requires name" in summary:
        return (
            "Choose one explicit custom-feature name from the inspected list before detail/source views. "
            "Do not guess a feature name."
        )
    if _extract_status(result_payload) == "error":
        return "Rewrite the same tool call according to the remediation in the error payload."
    if success_criteria:
        return f"Keep moving toward terminal evidence: {success_criteria[0]}"
    return f"Use the latest result from {tool_name} to move toward a terminal report."


def _is_mutating_success(*, arguments: dict[str, Any], result_payload: dict[str, Any]) -> bool:
    action = str(arguments.get("action") or "").strip().lower()
    if action and action not in _MUTATING_ACTIONS:
        return False
    return _extract_status(result_payload) in {"ok", "completed", "created", "started"}


def _extract_status(result_payload: dict[str, Any]) -> str:
    payload = result_payload.get("payload")
    structured = _extract_structured(payload)
    return str(structured.get("status") or ("error" if result_payload.get("error") else "ok")).strip().lower()


def _extract_summary(result_payload: dict[str, Any]) -> str:
    payload = result_payload.get("payload")
    structured = _extract_structured(payload)
    for key in ("summary", "message"):
        value = str(structured.get(key) or "").strip()
        if value:
            return value
    if result_payload.get("error"):
        return str(result_payload.get("error"))
    return ""


def _extract_warnings(result_payload: dict[str, Any]) -> list[str]:
    structured = _extract_structured(result_payload.get("payload"))
    warnings = structured.get("warnings") or structured.get("important_warnings") or []
    return [str(item) for item in warnings if str(item).strip()]


def _extract_errors(result_payload: dict[str, Any]) -> list[str]:
    if result_payload.get("error"):
        return [str(result_payload.get("error"))]
    structured = _extract_structured(result_payload.get("payload"))
    error = structured.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        return [message] if message else []
    text = str(error or "").strip()
    return [text] if text else []


def _extract_ids(result_payload: dict[str, Any]) -> list[str]:
    structured = _extract_structured(result_payload.get("payload"))
    data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
    sources = [data, structured]
    for sub_key in ("project", "record", "state_summary", "job", "operation", "result_ref"):
        sub = data.get(sub_key)
        if isinstance(sub, dict):
            sources.append(sub)
    return durable_created_id_values(sources=sources)[:8]


def _extract_structured(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    structured = payload.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    content = payload.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text") or "").strip()
                if text.startswith("{"):
                    try:
                        decoded = json.loads(text)
                    except Exception:
                        continue
                    if isinstance(decoded, dict):
                        return decoded
    return {}


def _is_mixed_domain_contract_context_read(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    runtime_profile: str,
    allowed_tools: set[str] | list[str] | None,
) -> bool:
    normalized_tool = str(tool_name or "").strip()
    if normalized_tool != "research_memory":
        return False
    action = str(arguments.get("action") or "").strip().lower()
    if action in _MUTATING_ACTIONS:
        return False
    return _is_mixed_domain_context_slice(runtime_profile=runtime_profile, allowed_tools=allowed_tools)


def _is_mixed_domain_domain_probe(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    runtime_profile: str,
    allowed_tools: set[str] | list[str] | None,
) -> bool:
    del arguments
    if not _is_mixed_domain_context_slice(runtime_profile=runtime_profile, allowed_tools=allowed_tools):
        return False
    if str(tool_name or "").strip() not in {
        "features_catalog",
        "events",
        "datasets",
        "features_custom",
        "features_dataset",
        "features_analytics",
        "models_dataset",
    }:
        return False
    if result_payload.get("error"):
        return False
    if result_payload.get("ok") is False:
        return False
    return _extract_status(result_payload) not in {"error", "failed"}


def _is_mixed_domain_context_slice(
    *,
    runtime_profile: str,
    allowed_tools: set[str] | list[str] | None,
) -> bool:
    normalized_profile = str(runtime_profile or "").strip()
    normalized_tools = {str(item or "").strip() for item in list(allowed_tools or []) if str(item or "").strip()}
    has_domain_contract_tool = bool(
        normalized_tools
        & {
            "features_catalog",
            "events",
            "datasets",
            "features_custom",
            "features_dataset",
            "features_analytics",
            "models_dataset",
        }
    )
    has_research = "research_memory" in normalized_tools
    return normalized_profile == "generic_mutation" and has_research and has_domain_contract_tool


def _record_summary(arguments: dict[str, Any]) -> str:
    record = arguments.get("record")
    return str(record.get("summary") or "").strip() if isinstance(record, dict) else ""


def _record_metadata(arguments: dict[str, Any]) -> dict[str, Any]:
    record = arguments.get("record")
    if isinstance(record, dict) and isinstance(record.get("metadata"), dict):
        return record["metadata"]
    return {}


def _coerce_refs(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_shortlist(arguments: dict[str, Any], record_payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for metadata in (_record_metadata(arguments), _record_metadata_from_record(record_payload)):
        families = metadata.get("shortlist_families") or metadata.get("signal_families")
        if isinstance(families, list):
            for item in families:
                candidate = str(item).strip()
                if candidate and candidate not in values:
                    values.append(candidate)
    for record in (_record_from_arguments(arguments), record_payload):
        for candidate in _candidate_payloads(record):
            family = str(candidate.get("family") or "").strip()
            if family and family not in values:
                values.append(family)
    if values:
        return values[:20]
    record = _record_from_arguments(arguments)
    if isinstance(record, dict):
        summary = str(record.get("summary") or "").strip()
        if ":" in summary:
            _, tail = summary.split(":", 1)
            values = [item.strip() for item in tail.split(",") if item.strip()]
            if values:
                return values[:20]
    return []


def _coerce_novelty_justification(arguments: dict[str, Any], record_payload: dict[str, Any]) -> bool | None:
    for metadata in (_record_metadata(arguments), _record_metadata_from_record(record_payload)):
        value = metadata.get("novelty_justification_present")
        if isinstance(value, bool):
            return value
    candidates = _candidate_payloads(_record_from_arguments(arguments)) + _candidate_payloads(record_payload)
    if candidates and all(str(item.get("why_new") or "").strip() for item in candidates):
        return True
    return None


def _record_from_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    record = arguments.get("record")
    return dict(record) if isinstance(record, dict) else {}


def _record_metadata_from_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _candidate_payloads(record: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        return []
    content = record.get("content")
    if not isinstance(content, dict):
        return []
    candidates = content.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [dict(item) for item in candidates if isinstance(item, dict)]


def _first_string(source: dict[str, Any], field_name: str) -> str:
    value = str(source.get(field_name) or "").strip()
    return value


def _is_missing_fact(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not bool(value)
    return False


__all__ = [
    "build_auto_final_report",
    "build_semantic_loop_abort",
    "build_watchdog_checkpoint",
    "compact_tool_result_message",
    "derive_research_write_facts",
    "should_auto_finalize_research_slice",
    "tool_call_signature",
]
