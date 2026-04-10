"""
Semantic progress heuristics for direct research slices.
"""

from __future__ import annotations

import json
from typing import Any


def tool_call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    if str(tool_name or "").strip() != "research_record":
        return ""
    if str(arguments.get("action") or "").strip() != "create":
        return ""
    record = arguments.get("record")
    if not isinstance(record, dict):
        return ""
    signature_payload = {
        "tool": "research_record",
        "action": "create",
        "kind": str(arguments.get("kind") or "").strip(),
        "project_id": str(arguments.get("project_id") or "").strip(),
        "title": str(record.get("title") or "").strip(),
        "summary": str(record.get("summary") or "").strip(),
    }
    return json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)


def compact_tool_result_message(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    success_criteria: list[str],
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
    guidance = _runtime_guidance(tool_name=tool_name, arguments=arguments, result_payload=result_payload, success_criteria=success_criteria)
    if guidance:
        normalized["runtime_guidance"] = guidance
    return json.dumps(normalized, ensure_ascii=False)


def is_successful_terminal_research_write(tool_name: str, arguments: dict[str, Any], result_payload: dict[str, Any]) -> bool:
    if str(tool_name or "").strip() != "research_record":
        return False
    if str(arguments.get("action") or "").strip() != "create":
        return False
    if str(arguments.get("kind") or "").strip() not in {"result", "milestone", "note"}:
        return False
    return _extract_status(result_payload) in {"ok", "completed", "created"}


def should_auto_finalize_research_slice(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    success_criteria: list[str],
    allowed_tools: set[str],
    required_output_facts: list[str],
    prior_contract_issue: bool,
) -> bool:
    if not is_successful_terminal_research_write(tool_name, arguments, result_payload):
        return False
    if not allowed_tools or not all(item.startswith("research_") for item in allowed_tools):
        return False
    derived_facts = derive_research_write_facts(arguments=arguments, result_payload=result_payload)
    if not str(derived_facts.get("research.project_id") or "").strip():
        return False
    if not (derived_facts.get("research.hypothesis_refs") or derived_facts.get("research.evidence_refs")):
        return False
    if any(_is_missing_fact(derived_facts.get(key)) for key in required_output_facts):
        return False
    summary_text = " ".join(
        [
            str(_record_title(arguments)),
            str(_record_summary(arguments)),
            str(_extract_summary(result_payload)),
            json.dumps(_record_metadata(arguments), ensure_ascii=False),
        ]
    ).lower()
    if "success criteria met" in summary_text or "all success criteria met" in summary_text:
        return True
    metadata = _record_metadata(arguments)
    if bool(metadata.get("success_criteria_met")):
        return True
    normalized_criteria = [str(item).strip().lower() for item in success_criteria if str(item).strip()]
    if normalized_criteria and any(item in summary_text for item in normalized_criteria):
        return True
    return _has_terminal_research_marker(summary_text)


def build_auto_final_report(
    *,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    success_criteria: list[str],
) -> str:
    summary = _record_summary(arguments) or _extract_summary(result_payload) or "Research slice completed after successful result recording."
    metadata = _record_metadata(arguments)
    facts = derive_research_write_facts(arguments=arguments, result_payload=result_payload)
    facts["execution.kind"] = "direct"
    facts["direct.auto_finalized_after_research_record"] = True
    facts["research_record.kind"] = str(arguments.get("kind") or "").strip()
    facts["research_record.project_id"] = str(arguments.get("project_id") or "").strip()
    ids = _extract_ids(result_payload)
    if ids:
        facts["research_record.ids"] = ids
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
        "confidence": 0.9,
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


def derive_research_write_facts(*, arguments: dict[str, Any], result_payload: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    project_id = str(arguments.get("project_id") or "").strip()
    if not project_id:
        structured = _extract_structured(result_payload.get("payload"))
        data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
        project_id = str(data.get("project_id") or structured.get("project_id") or "").strip()
    if project_id:
        facts["research.project_id"] = project_id
    refs = _coerce_refs(_record_metadata(arguments).get("evidence_refs"))
    ids = _extract_ids(result_payload)
    if refs:
        facts["research.evidence_refs"] = refs
    if ids:
        facts["research.created_ids"] = ids
        node_ids = [item for item in ids if str(item).startswith("node")]
        if node_ids:
            facts["research.hypothesis_refs"] = node_ids
    shortlist = _coerce_shortlist(arguments)
    if shortlist:
        facts["research.shortlist_families"] = shortlist
    return facts


def _runtime_guidance(*, tool_name: str, arguments: dict[str, Any], result_payload: dict[str, Any], success_criteria: list[str]) -> str:
    if is_successful_terminal_research_write(tool_name, arguments, result_payload):
        return "Do not call research_record(create) again for the same final write. If success criteria are satisfied, return terminal JSON now."
    if str(tool_name or "").strip() == "research_record" and _extract_status(result_payload) == "error":
        return "Rewrite the same research_record call according to the error remediation. Do not alternate blindly between milestone and result."
    if str(tool_name or "").strip() == "research_search":
        return "Use research_search only with a concrete query string."
    if success_criteria:
        return f"Keep moving toward terminal evidence: {success_criteria[0]}"
    return ""


def _extract_status(result_payload: dict[str, Any]) -> str:
    payload = result_payload.get("payload")
    structured = _extract_structured(payload)
    return str(structured.get("status") or ("error" if result_payload.get("error") else "ok")).strip()


def _has_terminal_research_marker(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    markers = (
        "terminal evidence",
        "final verification",
        "success criteria",
        "short-list complete",
        "shortlist complete",
        "wave-1 shortlist",
        "wave-1 short-list",
        "completion",
        "completed",
    )
    return any(marker in normalized for marker in markers)


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
    ids: list[str] = []
    seen: set[str] = set()
    sources = [data]
    for sub_key in ("project", "record", "state_summary"):
        sub = data.get(sub_key)
        if isinstance(sub, dict):
            sources.append(sub)
    sources.append(structured)
    for source in sources:
        for key in ("node_id", "project_id", "branch_id", "root_node_id", "default_branch_id", "operation_id"):
            value = str(source.get(key) or "").strip()
            if value and value not in seen:
                seen.add(value)
                ids.append(value)
    return ids[:8]


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


def _record_title(arguments: dict[str, Any]) -> str:
    record = arguments.get("record")
    return str(record.get("title") or "").strip() if isinstance(record, dict) else ""


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


def _coerce_shortlist(arguments: dict[str, Any]) -> list[str]:
    metadata = _record_metadata(arguments)
    families = metadata.get("shortlist_families") or metadata.get("signal_families")
    if isinstance(families, list):
        values = [str(item).strip() for item in families if str(item).strip()]
        if values:
            return values[:20]
    title = _record_title(arguments)
    summary = _record_summary(arguments)
    for text in (summary, title):
        parsed = _parse_family_list_from_text(text)
        if parsed:
            return parsed
    return []


def _parse_family_list_from_text(text: str) -> list[str]:
    marker = "signal families:"
    source = str(text or "")
    lower = source.lower()
    idx = lower.find(marker)
    if idx >= 0:
        tail = source[idx + len(marker):].strip()
        return [item.strip(" .") for item in tail.split(",") if item.strip(" .")][:20]
    return []


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
    "is_successful_terminal_research_write",
    "should_auto_finalize_research_slice",
    "tool_call_signature",
]
