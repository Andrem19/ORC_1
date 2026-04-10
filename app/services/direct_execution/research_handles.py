"""
Confirmed research handle tracking and deterministic project-id repair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchHandleState:
    confirmed_project_id: str = ""
    project_aliases: set[str] = field(default_factory=set)
    atlas_dimensions: dict[str, list[Any]] = field(default_factory=dict)
    confirmed_experiment_job_id: str = ""
    experiment_job_ids: set[str] = field(default_factory=set)


def repair_project_handle(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    state: ResearchHandleState,
) -> tuple[dict[str, Any], list[str]]:
    if not tool_name.startswith("research_"):
        return arguments, []
    project_id = str(arguments.get("project_id") or "").strip()
    confirmed = str(state.confirmed_project_id or "").strip()
    if not project_id or not confirmed or project_id == confirmed:
        return arguments, []
    if project_id in state.project_aliases or confirmed.startswith(f"{project_id}-"):
        rewritten = json.loads(json.dumps(arguments))
        rewritten["project_id"] = confirmed
        return rewritten, [f"rewrote stale project_id '{project_id}' to confirmed '{confirmed}'"]
    return arguments, []


def repair_atlas_coordinates(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    state: ResearchHandleState,
) -> tuple[dict[str, Any], list[str]]:
    if tool_name != "research_record":
        return arguments, []
    if str(arguments.get("action") or "").strip() != "create" or str(arguments.get("kind") or "").strip() != "hypothesis":
        return arguments, []
    atlas = arguments.get("atlas")
    if not isinstance(atlas, dict) or not state.atlas_dimensions:
        return arguments, []
    coords = atlas.get("coordinates")
    if not isinstance(coords, dict):
        return arguments, []
    repaired = json.loads(json.dumps(arguments))
    new_coords: dict[str, Any] = {}
    notes: list[str] = []
    for dim_name, allowed in state.atlas_dimensions.items():
        candidate = coords.get(dim_name)
        if candidate is None:
            candidate = _alias_coordinate(dim_name, coords)
        coerced = _coerce_to_allowed(candidate, allowed)
        if coerced is None:
            return arguments, []
        new_coords[dim_name] = coerced
        if candidate != coerced or dim_name not in coords:
            notes.append(f"repaired atlas.coordinates.{dim_name} -> {coerced!r}")
    if new_coords != coords:
        repaired["atlas"]["coordinates"] = new_coords
        return repaired, notes
    return arguments, []


def repair_experiment_handle(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    state: ResearchHandleState,
) -> tuple[dict[str, Any], list[str]]:
    if tool_name not in {"experiments_read", "experiments_inspect"}:
        return arguments, []
    if tool_name == "experiments_inspect" and str(arguments.get("view") or "").strip() == "list":
        return arguments, []
    confirmed = str(state.confirmed_experiment_job_id or "").strip()
    if not confirmed:
        return arguments, []
    current = str(arguments.get("job_id") or "").strip()
    if current and not _is_placeholder_job_id(current) and current in state.experiment_job_ids:
        return arguments, []
    if current and not _is_placeholder_job_id(current) and current == confirmed:
        return arguments, []
    rewritten = json.loads(json.dumps(arguments))
    rewritten["job_id"] = confirmed
    if current:
        return rewritten, [f"rewrote experiment job_id '{current}' to confirmed '{confirmed}'"]
    return rewritten, [f"filled missing experiment job_id with confirmed '{confirmed}'"]


def update_handle_state(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    state: ResearchHandleState,
) -> None:
    structured = _extract_structured(result_payload.get("payload"))
    data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
    project = data.get("project") if isinstance(data.get("project"), dict) else {}
    confirmed = (
        str(project.get("project_id") or "").strip()
        or str(data.get("project_id") or "").strip()
        or str((data.get("state_summary") or {}).get("project_id") or "").strip()
    )
    if confirmed:
        state.confirmed_project_id = confirmed
    dims = data.get("dimensions")
    if isinstance(dims, list):
        parsed: dict[str, list[Any]] = {}
        for item in dims:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            values = item.get("values")
            if name and isinstance(values, list) and values:
                parsed[name] = list(values)
        if parsed:
            state.atlas_dimensions = parsed
    for candidate in (
        str(arguments.get("project_id") or "").strip(),
        str(project.get("name") or "").strip(),
        str(project.get("project_id") or "").strip(),
        str((data.get("state_summary") or {}).get("project_id") or "").strip(),
    ):
        if candidate:
            state.project_aliases.add(candidate)
    job_ids = _extract_experiment_job_ids(data)
    if job_ids:
        for job_id in job_ids:
            state.experiment_job_ids.add(job_id)
        if state.confirmed_experiment_job_id not in state.experiment_job_ids:
            state.confirmed_experiment_job_id = job_ids[0]
    arg_job_id = str(arguments.get("job_id") or "").strip()
    if arg_job_id and arg_job_id in state.experiment_job_ids:
        state.confirmed_experiment_job_id = arg_job_id


def _extract_structured(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    structured = payload.get("structuredContent")
    return structured if isinstance(structured, dict) else {}


def _alias_coordinate(dim_name: str, coords: dict[str, Any]) -> Any:
    aliases = {
        "timeframe": ("anchor",),
        "anchor": ("timeframe",),
    }
    for alias in aliases.get(dim_name, ()):
        if alias in coords:
            return coords.get(alias)
    return None


def _coerce_to_allowed(candidate: Any, allowed: list[Any]) -> Any | None:
    if isinstance(candidate, (list, tuple, set)):
        for item in candidate:
            coerced = _coerce_to_allowed(item, allowed)
            if coerced is not None:
                return coerced
        return allowed[0] if allowed else None
    if candidate in allowed:
        return candidate
    allowed_strings = {str(item): item for item in allowed}
    text = str(candidate or "").strip()
    if text in allowed_strings:
        return allowed_strings[text]
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        for item in allowed:
            if isinstance(item, int) and item == int(digits):
                return item
        for item in allowed:
            if isinstance(item, str) and item.startswith(digits):
                return item
    if isinstance(candidate, int):
        for item in allowed:
            if isinstance(item, str) and item.startswith(str(candidate)):
                return item
    return allowed[0] if allowed else None


def _extract_experiment_job_ids(data: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    job = data.get("job")
    if isinstance(job, dict):
        job_id = str(job.get("job_id") or "").strip()
        if job_id:
            ids.append(job_id)
    jobs = data.get("jobs")
    if isinstance(jobs, list):
        for item in jobs:
            if not isinstance(item, dict):
                continue
            job_id = str(item.get("job_id") or "").strip()
            if job_id:
                ids.append(job_id)
    operation = data.get("operation")
    if isinstance(operation, dict):
        result_ref = operation.get("result_ref")
        if isinstance(result_ref, dict) and str(result_ref.get("kind") or "").strip() == "experiment_job":
            job_id = str(result_ref.get("id") or "").strip()
            if job_id:
                ids.append(job_id)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _is_placeholder_job_id(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized in {"<job_id>", "job_id", "latest", "pending", "none", "null", "0", "unknown"}:
        return True
    if normalized.startswith("job_"):
        return True
    return False


__all__ = [
    "ResearchHandleState",
    "repair_atlas_coordinates",
    "repair_experiment_handle",
    "repair_project_handle",
    "update_handle_state",
]
