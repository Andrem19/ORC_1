"""
Derive reusable downstream facts from direct tool transcripts.
"""

from __future__ import annotations

from typing import Any

from app.services.direct_execution.backtests_facts import normalize_backtests_facts
from app.services.direct_execution.handle_hygiene import (
    DURABLE_HANDLE_FIELDS,
    durable_created_id_values,
    is_suspicious_handle_value,
)

_HANDLE_FIELDS = DURABLE_HANDLE_FIELDS
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
    "set_baseline",
    "start",
    "sync",
    "train",
    "update",
    "validate",
}


def derive_facts_from_transcript(
    transcript: list[dict[str, Any]],
    *,
    runtime_profile: str = "",
) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    created_ids: list[str] = []
    evidence_refs: list[str] = []
    supported_evidence_refs: list[str] = []
    hypothesis_refs: list[str] = []
    warnings: list[str] = []
    statuses: list[str] = []
    successful_tool_names: list[str] = []
    successful_tool_count = 0
    successful_mutating_tool_count = 0
    for idx, item in enumerate(transcript, start=1):
        if item.get("kind") != "tool_result":
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        structured = _extract_structured(payload.get("payload"))
        data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
        record = data.get("record") if isinstance(data.get("record"), dict) else {}
        if _is_successful_tool_payload(payload=payload, structured=structured):
            successful_tool_count += 1
            tool_name = str(item.get("tool") or "").strip()
            if tool_name:
                _append_unique(successful_tool_names, [tool_name])
                _append_unique(supported_evidence_refs, [f"transcript:{idx}:{tool_name}"])
            if _is_mutating_arguments(arguments):
                successful_mutating_tool_count += 1
            _merge_handle_facts(facts=facts, sources=[arguments, structured, data, record], runtime_profile=runtime_profile)
            extracted_ids = _extract_ids(sources=[structured, data, record])
            _append_unique(created_ids, extracted_ids)
            _append_unique(supported_evidence_refs, extracted_ids)
            extracted_refs = _extract_evidence_refs(arguments=arguments, sources=[structured, data])
            _append_unique(evidence_refs, extracted_refs)
            _append_unique(supported_evidence_refs, extracted_refs)
            _merge_dimension_facts(facts=facts, structured=structured, data=data)
            _merge_shortlist_facts(facts=facts, arguments=arguments, data=data, record=record)
            _merge_backtests_fact_aliases(
                facts=facts,
                sources=[arguments, structured, data, record],
            )
            _merge_research_setup_facts(
                facts=facts,
                tool_name=str(item.get("tool") or "").strip(),
                arguments=arguments,
                structured=structured,
                data=data,
                record=record,
                hypothesis_refs=hypothesis_refs,
            )
        _append_unique(warnings, _extract_warnings(payload=payload, structured=structured))
        status = str(structured.get("status") or "").strip()
        if status:
            statuses.append(status)
    if hypothesis_refs:
        facts.setdefault("research.hypothesis_refs", hypothesis_refs[:20])
    elif created_ids:
        fallback_hypothesis_refs = [item for item in created_ids if "node" in str(item).lower()]
        if fallback_hypothesis_refs:
            facts.setdefault("research.hypothesis_refs", fallback_hypothesis_refs[:20])
    if created_ids:
        facts["direct.created_ids"] = created_ids[:20]
    if evidence_refs:
        facts["direct.evidence_refs"] = evidence_refs[:20]
    if supported_evidence_refs:
        facts["direct.supported_evidence_refs"] = supported_evidence_refs[:40]
    if warnings:
        facts["direct.warnings"] = warnings[:20]
    if statuses:
        facts["direct.statuses"] = statuses[-10:]
    if successful_tool_names:
        facts["direct.successful_tool_names"] = successful_tool_names[:20]
    if successful_tool_count:
        facts["direct.successful_tool_count"] = successful_tool_count
    if successful_mutating_tool_count:
        facts["direct.successful_mutating_tool_count"] = successful_mutating_tool_count
    return normalize_backtests_facts(facts)


def _extract_structured(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    structured = payload.get("structuredContent")
    return structured if isinstance(structured, dict) else {}


def _is_successful_tool_payload(*, payload: dict[str, Any], structured: dict[str, Any]) -> bool:
    if payload.get("error") or payload.get("ok") is False:
        return False
    status = str(structured.get("status") or "").strip().lower()
    if status in {"error", "failed"}:
        return False
    return True


def _merge_handle_facts(*, facts: dict[str, Any], sources: list[dict[str, Any]], runtime_profile: str) -> None:
    for field_name in _HANDLE_FIELDS:
        value = _first_string_field(sources, field_name)
        if not value:
            continue
        if is_suspicious_handle_value(value, field_name=field_name):
            continue
        facts.setdefault(field_name, value)
        if field_name == "project_id" or runtime_profile == "research_memory" and field_name == "project_id":
            facts.setdefault("research.project_id", value)


def _first_string_field(sources: list[dict[str, Any]], field_name: str) -> str:
    candidates = _field_candidates(field_name)
    for source in sources:
        if not isinstance(source, dict):
            continue
        for candidate in candidates:
            value = str(source.get(candidate) or "").strip()
            if value:
                return value
        for nested_key in ("project", "record", "job", "operation", "state_summary", "result_ref"):
            nested = source.get(nested_key)
            if isinstance(nested, dict):
                for candidate in candidates:
                    nested_value = str(nested.get(candidate) or "").strip()
                    if nested_value:
                        return nested_value
    return ""


def _field_candidates(field_name: str) -> tuple[str, ...]:
    if field_name == "branch_id":
        return ("branch_id", "default_branch_id")
    return (field_name,)


def _extract_ids(*, sources: list[dict[str, Any]]) -> list[str]:
    return durable_created_id_values(sources=sources)


def _extract_evidence_refs(*, arguments: dict[str, Any], sources: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    record = arguments.get("record") if isinstance(arguments.get("record"), dict) else {}
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    if isinstance(metadata.get("evidence_refs"), list):
        _append_unique(refs, [str(item).strip() for item in metadata["evidence_refs"] if str(item).strip()])
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("evidence_refs", "refs"):
            raw = source.get(key)
            if isinstance(raw, list):
                _append_unique(refs, [str(item).strip() for item in raw if str(item).strip()])
    return refs


def _extract_warnings(*, payload: dict[str, Any], structured: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for source in (payload, structured):
        raw = source.get("warnings") if isinstance(source, dict) else None
        if isinstance(raw, list):
            _append_unique(result, [str(item).strip() for item in raw if str(item).strip()])
    return result


def _append_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in target:
            target.append(normalized)


def _is_mutating_arguments(arguments: dict[str, Any]) -> bool:
    action = str(arguments.get("action") or "").strip().lower()
    if action:
        return action in _MUTATING_ACTIONS
    return any(key in arguments for key in ("record", "payload", "project", "coordinates"))


def _merge_dimension_facts(*, facts: dict[str, Any], structured: dict[str, Any], data: dict[str, Any]) -> None:
    for source in (data, structured):
        raw_dimensions = source.get("dimensions") if isinstance(source, dict) else None
        if not isinstance(raw_dimensions, list):
            continue
        names = [str(item.get("name") or "").strip() for item in raw_dimensions if isinstance(item, dict) and str(item.get("name") or "").strip()]
        if names:
            facts.setdefault("atlas_dimensions", names)
            return


def _merge_backtests_fact_aliases(*, facts: dict[str, Any], sources: list[dict[str, Any]]) -> None:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            key_text = str(key or "").strip()
            if key_text not in {"feature_long_job", "feature_short_job", "diagnostics_run"}:
                continue
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            facts.setdefault(key_text, value)


def _merge_shortlist_facts(*, facts: dict[str, Any], arguments: dict[str, Any], data: dict[str, Any], record: dict[str, Any]) -> None:
    shortlist = _extract_shortlist_families(arguments=arguments, data=data, record=record)
    if shortlist:
        facts.setdefault("research.shortlist_families", shortlist[:20])
    novelty = _extract_novelty_justification(arguments=arguments, record=record)
    if novelty is not None:
        facts.setdefault("research.novelty_justification_present", novelty)


def _merge_research_setup_facts(
    *,
    facts: dict[str, Any],
    tool_name: str,
    arguments: dict[str, Any],
    structured: dict[str, Any],
    data: dict[str, Any],
    record: dict[str, Any],
    hypothesis_refs: list[str],
) -> None:
    if tool_name == "research_project":
        _merge_research_project_setup_facts(
            facts=facts,
            arguments=arguments,
            structured=structured,
            data=data,
        )
        return
    if tool_name == "research_map":
        _merge_research_map_setup_facts(facts=facts, structured=structured, data=data)
        return
    if tool_name in {"research_memory", "research_record"}:
        _merge_research_memory_setup_facts(
            facts=facts,
            arguments=arguments,
            structured=structured,
            data=data,
            record=record,
            hypothesis_refs=hypothesis_refs,
        )


def _merge_research_project_setup_facts(
    *,
    facts: dict[str, Any],
    arguments: dict[str, Any],
    structured: dict[str, Any],
    data: dict[str, Any],
) -> None:
    sources = [arguments, structured, data]
    project = data.get("project") if isinstance(data.get("project"), dict) else {}
    state_summary = data.get("state_summary") if isinstance(data.get("state_summary"), dict) else {}
    if project:
        sources.append(project)
    if state_summary:
        sources.append(state_summary)
    branch_id = _first_string_field(sources, "branch_id")
    if branch_id:
        facts.setdefault("research.branch_id", branch_id)
    action = str(arguments.get("action") or data.get("action") or "").strip().lower()
    if action != "set_baseline":
        return
    snapshot_id = _first_string_field(sources, "snapshot_id")
    version = _first_int_field(sources, "version")
    if snapshot_id:
        facts["research.baseline_snapshot_id"] = snapshot_id
    if version is not None:
        facts["research.baseline_version"] = version
    if snapshot_id and version is not None:
        facts["research.baseline_configured"] = True


def _merge_research_map_setup_facts(*, facts: dict[str, Any], structured: dict[str, Any], data: dict[str, Any]) -> None:
    atlas_summary = data.get("atlas_summary") if isinstance(data.get("atlas_summary"), dict) else {}
    state_summary = data.get("state_summary") if isinstance(data.get("state_summary"), dict) else {}
    dimensions = atlas_summary.get("dimensions") if isinstance(atlas_summary.get("dimensions"), list) else data.get("dimensions")
    dimension_count = None
    if isinstance(dimensions, list):
        dimension_count = len([item for item in dimensions if isinstance(item, dict)])
    elif isinstance(state_summary.get("dimension_count"), int):
        dimension_count = int(state_summary["dimension_count"])
    atlas_defined = bool(
        state_summary.get("atlas_defined")
        or atlas_summary
        or (dimension_count is not None and dimension_count > 0)
    )
    if atlas_defined:
        facts["research.atlas_defined"] = True
    if dimension_count is not None:
        facts["research.atlas_dimension_count"] = dimension_count


def _merge_research_memory_setup_facts(
    *,
    facts: dict[str, Any],
    arguments: dict[str, Any],
    structured: dict[str, Any],
    data: dict[str, Any],
    record: dict[str, Any],
    hypothesis_refs: list[str],
) -> None:
    record_refs = data.get("record_refs") if isinstance(data.get("record_refs"), dict) else {}
    node_id = _first_string_field([record_refs, record, data, structured], "memory_node_id") or _first_string_field(
        [record, data, structured],
        "node_id",
    )
    if node_id:
        facts["research.memory_node_id"] = node_id
        if str(arguments.get("kind") or "").strip().lower() == "hypothesis":
            _append_unique(hypothesis_refs, [node_id])
    metadata_sources: list[dict[str, Any]] = []
    argument_record = arguments.get("record") if isinstance(arguments.get("record"), dict) else {}
    if argument_record:
        metadata_sources.append(argument_record)
    if record:
        metadata_sources.append(record)
    for source in metadata_sources:
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        if not isinstance(metadata, dict):
            continue
        if metadata.get("invariants"):
            facts["research.invariants_recorded"] = True
        if metadata.get("naming_convention"):
            facts["research.naming_recorded"] = True


def _extract_shortlist_families(*, arguments: dict[str, Any], data: dict[str, Any], record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    metadata = arguments.get("record", {}).get("metadata") if isinstance(arguments.get("record"), dict) else {}
    for key in ("shortlist_families", "signal_families"):
        raw = metadata.get(key) if isinstance(metadata, dict) else None
        if isinstance(raw, list):
            _append_unique(values, [str(item).strip() for item in raw if str(item).strip()])
    record_metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for key in ("shortlist_families", "signal_families"):
        raw = record_metadata.get(key) if isinstance(record_metadata, dict) else None
        if isinstance(raw, list):
            _append_unique(values, [str(item).strip() for item in raw if str(item).strip()])
    for source_record in (_record_from_arguments(arguments), record):
        for candidate in _candidate_payloads(source_record):
            family = str(candidate.get("family") or "").strip()
            if family:
                _append_unique(values, [family])
    return values


def _extract_novelty_justification(*, arguments: dict[str, Any], record: dict[str, Any]) -> bool | None:
    for source_record in (_record_from_arguments(arguments), record):
        metadata = source_record.get("metadata") if isinstance(source_record.get("metadata"), dict) else {}
        value = metadata.get("novelty_justification_present") if isinstance(metadata, dict) else None
        if isinstance(value, bool):
            return value
    candidates = _candidate_payloads(_record_from_arguments(arguments)) + _candidate_payloads(record)
    if candidates and all(str(item.get("why_new") or "").strip() for item in candidates):
        return True
    return None


def _record_from_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    record = arguments.get("record")
    return dict(record) if isinstance(record, dict) else {}


def _candidate_payloads(record: dict[str, Any]) -> list[dict[str, Any]]:
    content = record.get("content") if isinstance(record.get("content"), dict) else {}
    raw = content.get("candidates")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _first_int_field(sources: list[dict[str, Any]], field_name: str) -> int | None:
    candidates = _field_candidates(field_name)
    for source in sources:
        if not isinstance(source, dict):
            continue
        for candidate in candidates:
            value = _coerce_int(source.get(candidate))
            if value is not None:
                return value
        for nested_key in ("project", "record", "job", "operation", "state_summary", "result_ref"):
            nested = source.get(nested_key)
            if not isinstance(nested, dict):
                continue
            for candidate in candidates:
                value = _coerce_int(nested.get(candidate))
                if value is not None:
                    return value
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


__all__ = ["derive_facts_from_transcript"]
