"""
Generic handle tracking and repair for direct MCP tool calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.services.mcp_catalog.models import McpCatalogSnapshot
from app.services.direct_execution.handle_hygiene import (
    DURABLE_HANDLE_FIELDS,
    is_placeholder_handle_value,
    is_suspicious_handle_value,
)

_HANDLE_FIELDS = DURABLE_HANDLE_FIELDS[:-1]


@dataclass
class ResearchHandleState:
    confirmed_project_id: str = ""
    confirmed_job_id: str = ""
    confirmed_experiment_job_id: str = ""
    confirmed_run_id: str = ""
    confirmed_snapshot_id: str = ""
    confirmed_operation_id: str = ""
    project_aliases: set[str] = field(default_factory=set)
    experiment_job_ids: set[str] = field(default_factory=set)
    atlas_dimensions: dict[str, list[Any]] = field(default_factory=dict)
    known_handles: dict[str, set[str]] = field(default_factory=dict)


def seed_handle_state_from_facts(state: ResearchHandleState, facts: dict[str, Any] | None) -> None:
    """Prime confirmed durable handles from accepted prerequisite facts."""

    if not isinstance(facts, dict):
        return
    for field_name in _HANDLE_FIELDS:
        current = _confirmed_handle_value(state, field_name)
        candidates = _handle_candidates_from_facts(facts, field_name)
        for candidate in candidates:
            state.known_handles.setdefault(field_name, set()).add(candidate)
        if current or not candidates:
            continue
        preferred = _select_seed_candidate(candidates)
        if preferred:
            _set_confirmed_handle(state, field_name, preferred)
            if field_name == "job_id":
                state.experiment_job_ids.add(preferred)


def repair_handle_arguments(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    state: ResearchHandleState,
    catalog_snapshot: McpCatalogSnapshot | None,
    runtime_profile: str = "",
) -> tuple[dict[str, Any], list[str]]:
    del runtime_profile
    tool = catalog_snapshot.get_tool(tool_name) if catalog_snapshot is not None else None
    accepted = set(tool.accepted_handle_fields) if tool is not None else {
        field_name for field_name in _HANDLE_FIELDS
        if field_name in arguments or _confirmed_handle_value(state, field_name)
    }
    if not accepted:
        return arguments, []
    rewritten = json.loads(json.dumps(arguments))
    notes: list[str] = []
    for field_name in _HANDLE_FIELDS:
        if field_name not in accepted:
            continue
        confirmed = _confirmed_handle_value(state, field_name)
        if not confirmed:
            continue
        current = str(rewritten.get(field_name) or "").strip()
        known_values = set(state.known_handles.get(field_name, set()))
        if field_name == "job_id":
            known_values |= set(state.experiment_job_ids)
        if field_name == "project_id" and current and current != confirmed:
            if (
                current in state.project_aliases
                or confirmed.startswith(f"{current}-")
                or _should_rewrite_invalid_project_id(current, confirmed=confirmed, known_values=known_values)
            ):
                rewritten[field_name] = confirmed
                notes.append(f"rewrote stale {field_name} '{current}' to confirmed '{confirmed}'")
            continue
        if current and current != confirmed and len(known_values) == 1 and confirmed in known_values:
            rewritten[field_name] = confirmed
            notes.append(f"rewrote stale {field_name} '{current}' to confirmed '{confirmed}'")
            continue
        if current and not is_placeholder_handle_value(current):
            continue
        rewritten[field_name] = confirmed
        if current:
            notes.append(f"rewrote placeholder {field_name} '{current}' to confirmed '{confirmed}'")
        else:
            notes.append(f"filled missing {field_name} with confirmed '{confirmed}'")
    return rewritten, notes


def update_handle_state(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    state: ResearchHandleState,
    runtime_profile: str = "",
) -> None:
    del tool_name, runtime_profile
    structured = _extract_structured(result_payload.get("payload"))
    data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
    _remember_project_candidates(state=state, data=data)
    sources = [arguments, structured, data]
    for sub_key in ("project", "record", "state_summary", "job", "operation", "result_ref"):
        sub = data.get(sub_key)
        if isinstance(sub, dict):
            sources.append(sub)
    for field_name in _HANDLE_FIELDS:
        value = _first_string_field(sources, field_name)
        if value:
            if is_suspicious_handle_value(value, field_name=field_name):
                continue
            _set_confirmed_handle(state, field_name, value)
            state.known_handles.setdefault(field_name, set()).add(value)
            if field_name == "job_id":
                state.experiment_job_ids.add(value)
    for candidate in (
        str(arguments.get("project_id") or "").strip(),
        _first_string_field(sources, "name"),
        _first_string_field(sources, "project_id"),
    ):
        if candidate:
            state.project_aliases.add(candidate)


def repair_project_handle(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    state: ResearchHandleState,
    catalog_snapshot: McpCatalogSnapshot | None = None,
    runtime_profile: str = "",
) -> tuple[dict[str, Any], list[str]]:
    return repair_handle_arguments(
        tool_name=tool_name,
        arguments=arguments,
        state=state,
        catalog_snapshot=catalog_snapshot,
        runtime_profile=runtime_profile,
    )


def repair_experiment_handle(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    state: ResearchHandleState,
    catalog_snapshot: McpCatalogSnapshot | None = None,
    runtime_profile: str = "",
) -> tuple[dict[str, Any], list[str]]:
    return repair_handle_arguments(
        tool_name=tool_name,
        arguments=arguments,
        state=state,
        catalog_snapshot=catalog_snapshot,
        runtime_profile=runtime_profile,
    )


def repair_atlas_coordinates(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    state: ResearchHandleState,
) -> tuple[dict[str, Any], list[str]]:
    del tool_name
    atlas = arguments.get("atlas") if isinstance(arguments.get("atlas"), dict) else None
    if atlas is None:
        return arguments, []
    coordinates = atlas.get("coordinates") if isinstance(atlas.get("coordinates"), dict) else None
    if coordinates is None or not state.atlas_dimensions:
        return arguments, []
    rewritten = json.loads(json.dumps(arguments))
    atlas_copy = rewritten.get("atlas") if isinstance(rewritten.get("atlas"), dict) else {}
    coords_copy = atlas_copy.get("coordinates") if isinstance(atlas_copy.get("coordinates"), dict) else {}
    normalized, notes = _normalize_coordinates(coords_copy, state.atlas_dimensions)
    if not notes:
        return arguments, []
    atlas_copy["coordinates"] = normalized
    rewritten["atlas"] = atlas_copy
    return rewritten, notes


def _handle_candidates_from_facts(facts: dict[str, Any], field_name: str) -> list[str]:
    candidates: list[str] = []
    accepted_keys = {field_name}
    if field_name == "project_id":
        accepted_keys.add("research.project_id")
    for raw_key, raw_value in facts.items():
        key = str(raw_key or "").strip()
        if key not in accepted_keys and not any(key.endswith(f".{item}") for item in accepted_keys):
            continue
        value = str(raw_value or "").strip()
        if not value:
            continue
        if is_placeholder_handle_value(value) or is_suspicious_handle_value(value, field_name=field_name):
            continue
        if value not in candidates:
            candidates.append(value)
    return candidates


def _select_seed_candidate(candidates: list[str]) -> str:
    if len(candidates) == 1:
        return candidates[0]
    return ""


def _extract_structured(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    structured = payload.get("structuredContent")
    return structured if isinstance(structured, dict) else {}


def _first_string_field(sources: list[dict[str, Any]], field_name: str) -> str:
    for source in sources:
        if not isinstance(source, dict):
            continue
        value = str(source.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _confirmed_handle_value(state: ResearchHandleState, field_name: str) -> str:
    return {
        "project_id": state.confirmed_project_id,
        "job_id": state.confirmed_job_id or state.confirmed_experiment_job_id,
        "run_id": state.confirmed_run_id,
        "snapshot_id": state.confirmed_snapshot_id,
        "operation_id": state.confirmed_operation_id,
    }.get(field_name, "")


def _set_confirmed_handle(state: ResearchHandleState, field_name: str, value: str) -> None:
    if field_name == "project_id":
        state.confirmed_project_id = value
    elif field_name == "job_id":
        state.confirmed_job_id = value
        state.confirmed_experiment_job_id = value
    elif field_name == "run_id":
        state.confirmed_run_id = value
    elif field_name == "snapshot_id":
        state.confirmed_snapshot_id = value
    elif field_name == "operation_id":
        state.confirmed_operation_id = value


def _remember_project_candidates(*, state: ResearchHandleState, data: dict[str, Any]) -> None:
    projects = data.get("projects") if isinstance(data.get("projects"), list) else []
    if not isinstance(projects, list):
        return
    candidates: list[tuple[str, dict[str, Any]]] = []
    for item in projects:
        if not isinstance(item, dict):
            continue
        project_id = str(item.get("project_id") or "").strip()
        if not project_id:
            continue
        state.known_handles.setdefault("project_id", set()).add(project_id)
        name = str(item.get("name") or "").strip()
        if name:
            state.project_aliases.add(name)
        candidates.append((project_id, item))
    if state.confirmed_project_id:
        return
    selected = _select_best_project_candidate(candidates)
    if selected:
        state.confirmed_project_id = selected


def _select_best_project_candidate(candidates: list[tuple[str, dict[str, Any]]]) -> str:
    preferred: list[str] = []
    fallback: list[str] = []
    for project_id, project in candidates:
        fallback.append(project_id)
        name = str(project.get("name") or "").strip().lower()
        tags = {
            str(item).strip().lower()
            for item in project.get("tags", [])
            if str(item).strip()
        }
        metadata = project.get("metadata") if isinstance(project.get("metadata"), dict) else {}
        metadata_kind = str(metadata.get("kind") or "").strip().lower()
        if (
            "incident" in name
            or "placeholder" in project_id.lower()
            or metadata_kind == "incident_backlog"
            or {"incident", "infrastructure", "mcp"} & tags
        ):
            continue
        preferred.append(project_id)
    if len(preferred) == 1:
        return preferred[0]
    if not preferred and len(fallback) == 1:
        return fallback[0]
    return ""


def _should_rewrite_invalid_project_id(
    current: str,
    *,
    confirmed: str,
    known_values: set[str],
) -> bool:
    if not current or current == confirmed or current in known_values:
        return False
    if current.isdigit():
        return True
    lowered = current.lower()
    if lowered in {"true", "false"}:
        return True
    if len(current) >= 16 and all(ch in "0123456789abcdef" for ch in lowered):
        return True
    return False


def _normalize_coordinates(
    coordinates: dict[str, Any],
    atlas_dimensions: dict[str, list[Any]],
) -> tuple[dict[str, Any], list[str]]:
    normalized: dict[str, Any] = {}
    used_dimensions: set[str] = set()
    notes: list[str] = []
    for raw_key, raw_value in coordinates.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        target_key = key if key in atlas_dimensions else _match_dimension_key(
            value=raw_value,
            atlas_dimensions=atlas_dimensions,
            used_dimensions=used_dimensions,
        )
        if not target_key:
            continue
        coerced = _coerce_to_allowed(raw_value, atlas_dimensions.get(target_key) or [])
        if coerced is None:
            continue
        normalized[target_key] = coerced
        used_dimensions.add(target_key)
        if target_key != key or coerced != raw_value:
            notes.append(f"normalized atlas coordinate '{key}' to '{target_key}'")
    for key, allowed_values in atlas_dimensions.items():
        if key in normalized or not allowed_values:
            continue
        if len(allowed_values) == 1:
            normalized[key] = allowed_values[0]
            notes.append(f"filled missing atlas coordinate '{key}' from the only allowed value")
    return normalized, notes


def _match_dimension_key(
    *,
    value: Any,
    atlas_dimensions: dict[str, list[Any]],
    used_dimensions: set[str],
) -> str:
    for key, allowed_values in atlas_dimensions.items():
        if key in used_dimensions:
            continue
        if _coerce_to_allowed(value, allowed_values) is not None:
            return key
    return ""


def _coerce_to_allowed(value: Any, allowed_values: list[Any]) -> Any | None:
    candidates = value if isinstance(value, list) else [value]
    for candidate in candidates:
        for allowed in allowed_values:
            if _values_match(candidate, allowed):
                return allowed
    return allowed_values[0] if allowed_values else None


def _values_match(candidate: Any, allowed: Any) -> bool:
    return bool(_coerce_numeric_token(candidate) == _coerce_numeric_token(allowed) or _normalize_value(candidate) == _normalize_value(allowed))


def _normalize_value(value: Any) -> str:
    return str(value or "").strip().lower()


def _coerce_numeric_token(value: Any) -> str:
    text = _normalize_value(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits or text


__all__ = [
    "ResearchHandleState",
    "repair_atlas_coordinates",
    "repair_experiment_handle",
    "repair_handle_arguments",
    "repair_project_handle",
    "seed_handle_state_from_facts",
    "update_handle_state",
]
