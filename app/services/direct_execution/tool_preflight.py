"""
Cheap local preflight and repair for direct dev_space1 tool calls.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolPreflightResult:
    arguments: dict[str, Any]
    local_payload: dict[str, Any] | None = None
    charge_budget: bool = True
    repair_notes: list[str] = field(default_factory=list)


def preflight_direct_tool_call(tool_name: str, arguments: dict[str, Any]) -> ToolPreflightResult:
    normalized_tool = str(tool_name or "").strip()
    normalized_arguments = _deepcopy_jsonable(arguments)
    if normalized_tool == "research_search":
        return _preflight_research_search(normalized_arguments)
    if normalized_tool == "research_record":
        return _preflight_research_record(normalized_arguments)
    if normalized_tool == "research_map":
        return _preflight_research_map(normalized_arguments)
    if normalized_tool == "experiments_read":
        return _preflight_experiments_read(normalized_arguments)
    if normalized_tool == "experiments_inspect":
        return _preflight_experiments_inspect(normalized_arguments)
    if normalized_tool == "features_catalog":
        return _preflight_features_catalog(normalized_arguments)
    if normalized_tool == "backtests_conditions":
        return _preflight_backtests_conditions(normalized_arguments)
    if normalized_tool == "backtests_analysis":
        return _preflight_backtests_analysis(normalized_arguments)
    return ToolPreflightResult(arguments=normalized_arguments)


def _preflight_research_search(arguments: dict[str, Any]) -> ToolPreflightResult:
    query = str(arguments.get("query") or "").strip()
    if query:
        return ToolPreflightResult(arguments=arguments)
    return ToolPreflightResult(
        arguments=arguments,
        local_payload={
            "ok": False,
            "error_class": "agent_contract_misuse",
            "summary": "research_search requires a non-empty query",
            "details": {
                "tool_name": "research_search",
                "required_field": "query",
                "suggested_fix": "Retry with a concrete query string, or use research_project/research_map to inspect state.",
            },
        },
        charge_budget=False,
    )


def _preflight_research_record(arguments: dict[str, Any]) -> ToolPreflightResult:
    action = str(arguments.get("action") or "").strip()
    kind = str(arguments.get("kind") or "").strip()

    if action == "status":
        if str(arguments.get("operation_id") or "").strip():
            return ToolPreflightResult(arguments=arguments)
        return ToolPreflightResult(
            arguments=arguments,
            local_payload={
                "ok": False,
                "error_class": "agent_contract_misuse",
                "summary": "research_record(action='status') requires operation_id",
                "details": {
                    "tool_name": "research_record",
                    "required_field": "operation_id",
                    "suggested_fix": (
                        "Use the operation_id returned by a previous research_record(action='create') response. "
                        "Check the ids field in the tool result."
                    ),
                },
            },
            charge_budget=False,
        )

    if action != "create" or kind != "hypothesis":
        return ToolPreflightResult(arguments=arguments)

    repair_notes: list[str] = []
    record = arguments.get("record")
    top_level_atlas = arguments.get("atlas")
    top_level_coordinates = arguments.pop("coordinates", None)

    if not isinstance(top_level_atlas, dict) and isinstance(record, dict) and isinstance(record.get("atlas"), dict):
        arguments["atlas"] = _deepcopy_jsonable(record.pop("atlas"))
        top_level_atlas = arguments["atlas"]
        repair_notes.append("moved record.atlas to top-level atlas")

    if isinstance(top_level_atlas, dict) and top_level_coordinates is not None and "coordinates" not in top_level_atlas:
        top_level_atlas["coordinates"] = _coerce_mapping(top_level_coordinates)
        repair_notes.append("moved top-level coordinates into atlas.coordinates")

    if isinstance(top_level_atlas, dict) and "coordinates" in top_level_atlas:
        coerced = _coerce_mapping(top_level_atlas.get("coordinates"))
        if coerced is not top_level_atlas.get("coordinates"):
            top_level_atlas["coordinates"] = coerced
            repair_notes.append("coerced atlas.coordinates into object form")

    atlas = arguments.get("atlas")
    missing_fields: list[str] = []
    if not isinstance(atlas, dict):
        missing_fields = ["atlas.statement", "atlas.expected_outcome", "atlas.falsification_criteria", "atlas.coordinates"]
    else:
        for field in ("statement", "expected_outcome", "falsification_criteria", "coordinates"):
            value = atlas.get(field)
            if field == "coordinates":
                if not isinstance(value, dict) or not value:
                    missing_fields.append(f"atlas.{field}")
            elif not str(value or "").strip():
                missing_fields.append(f"atlas.{field}")

    if missing_fields:
        return ToolPreflightResult(
            arguments=arguments,
            local_payload={
                "ok": False,
                "error_class": "agent_contract_misuse",
                "summary": "research_record hypothesis in atlas-enabled project requires top-level atlas block",
                "details": {
                    "tool_name": "research_record",
                    "required_fields": missing_fields,
                    "repair_hint": (
                        "Use research_record(action='create', kind='hypothesis', atlas={statement, "
                        "expected_outcome, falsification_criteria, coordinates}) with coordinates as an object."
                    ),
                    "do_not_do": [
                        "do not put atlas inside record",
                        "do not pass coordinates as a string",
                    ],
                },
            },
            charge_budget=False,
            repair_notes=repair_notes,
        )
    return ToolPreflightResult(arguments=arguments, repair_notes=repair_notes)


def _preflight_research_map(arguments: dict[str, Any]) -> ToolPreflightResult:
    if str(arguments.get("project_id") or "").strip():
        return ToolPreflightResult(arguments=arguments)
    return ToolPreflightResult(
        arguments=arguments,
        local_payload={
            "ok": False,
            "error_class": "agent_contract_misuse",
            "summary": "research_map always requires project_id",
            "details": {
                "tool_name": "research_map",
                "required_field": "project_id",
                "suggested_fix": (
                    "Call research_project(action='list') first to get available project_ids, "
                    "then retry research_map with a project_id from the list."
                ),
            },
        },
        charge_budget=False,
    )


def _preflight_experiments_read(arguments: dict[str, Any]) -> ToolPreflightResult:
    if str(arguments.get("job_id") or "").strip():
        return ToolPreflightResult(arguments=arguments)
    return ToolPreflightResult(
        arguments=arguments,
        local_payload={
            "ok": False,
            "error_class": "agent_contract_misuse",
            "summary": "experiments_read is not a listing endpoint; it always requires a concrete job_id",
            "details": {
                "tool_name": "experiments_read",
                "required_field": "job_id",
                "suggested_fix": (
                    "Call experiments_inspect(view='list') first to see available jobs, "
                    "then retry experiments_read with a specific job_id from the list."
                ),
            },
        },
        charge_budget=False,
    )


def _preflight_experiments_inspect(arguments: dict[str, Any]) -> ToolPreflightResult:
    view = str(arguments.get("view") or "").strip()
    if view == "list":
        return ToolPreflightResult(arguments=arguments)
    if str(arguments.get("job_id") or "").strip():
        return ToolPreflightResult(arguments=arguments)
    return ToolPreflightResult(
        arguments=arguments,
        local_payload={
            "ok": False,
            "error_class": "agent_contract_misuse",
            "summary": f"experiments_inspect(view='{view}') requires a concrete job_id",
            "details": {
                "tool_name": "experiments_inspect",
                "required_field": "job_id",
                "view": view,
                "suggested_fix": (
                    "Call experiments_inspect(view='list') first to see available jobs, "
                    "then retry with a specific job_id."
                ),
            },
        },
        charge_budget=False,
    )


def _preflight_features_catalog(arguments: dict[str, Any]) -> ToolPreflightResult:
    if str(arguments.get("scope") or "").strip() != "timeframe":
        return ToolPreflightResult(arguments=arguments)
    timeframe = arguments.get("timeframe")
    if isinstance(timeframe, str) and timeframe.strip():
        repaired = _normalize_timeframe_token(timeframe)
        if repaired == timeframe:
            return ToolPreflightResult(arguments=arguments)
        rewritten = _deepcopy_jsonable(arguments)
        rewritten["timeframe"] = repaired
        return ToolPreflightResult(arguments=rewritten, repair_notes=[f"normalized features_catalog.timeframe '{timeframe}' -> '{repaired}'"])
    if isinstance(timeframe, (int, float)):
        repaired = _normalize_timeframe_token(str(int(timeframe)))
        rewritten = _deepcopy_jsonable(arguments)
        rewritten["timeframe"] = repaired
        return ToolPreflightResult(arguments=rewritten, repair_notes=[f"coerced numeric features_catalog.timeframe {timeframe!r} -> '{repaired}'"])
    return ToolPreflightResult(arguments=arguments)


def _preflight_backtests_conditions(arguments: dict[str, Any]) -> ToolPreflightResult:
    action = str(arguments.get("action") or "run").strip().lower()
    repair_notes: list[str] = []
    if action == "run":
        snapshot_raw = str(arguments.get("snapshot_id") or "").strip()
        if "@" in snapshot_raw and not str(arguments.get("version") or "").strip():
            snapshot_id, _, version = snapshot_raw.partition("@")
            if snapshot_id.strip() and version.strip().isdigit():
                rewritten = _deepcopy_jsonable(arguments)
                rewritten["snapshot_id"] = snapshot_id.strip()
                rewritten["version"] = version.strip()
                arguments = rewritten
                repair_notes.append(f"split snapshot_id '{snapshot_raw}' -> snapshot_id='{snapshot_id.strip()}', version='{version.strip()}'")
    if action == "list":
        return ToolPreflightResult(arguments=arguments, repair_notes=repair_notes)
    if action in {"status", "result", "cancel"}:
        if str(arguments.get("job_id") or "").strip():
            return ToolPreflightResult(arguments=arguments, repair_notes=repair_notes)
        return ToolPreflightResult(
            arguments=arguments,
            local_payload={
                "ok": False,
                "error_class": "agent_contract_misuse",
                "summary": f"backtests_conditions(action='{action}') requires job_id",
                "details": {
                    "tool_name": "backtests_conditions",
                    "required_field": "job_id",
                    "suggested_fix": "Call backtests_conditions(action='list') first, then retry with a concrete job_id.",
                },
            },
            charge_budget=False,
            repair_notes=repair_notes,
        )
    if str(arguments.get("snapshot_id") or "").strip():
        return ToolPreflightResult(arguments=arguments, repair_notes=repair_notes)
    return ToolPreflightResult(
        arguments=arguments,
        local_payload={
            "ok": False,
            "error_class": "agent_contract_misuse",
            "summary": "backtests_conditions(action='run') requires snapshot_id",
            "details": {
                "tool_name": "backtests_conditions",
                "required_field": "snapshot_id",
                "suggested_fix": (
                    "Call backtests_strategy(action='inspect', view='detail', "
                    "snapshot_id='active-signal-v1', version='1') first to get snapshot_id, "
                    "then retry backtests_conditions with that snapshot_id."
                ),
            },
        },
        charge_budget=False,
        repair_notes=repair_notes,
    )


def _preflight_backtests_analysis(arguments: dict[str, Any]) -> ToolPreflightResult:
    action = str(arguments.get("action") or "inspect").strip().lower()
    if action in {"list", "inspect"}:
        return ToolPreflightResult(arguments=arguments)
    if action in {"status", "result", "cancel"}:
        if str(arguments.get("job_id") or "").strip():
            return ToolPreflightResult(arguments=arguments)
        rewritten = _deepcopy_jsonable(arguments)
        rewritten["action"] = "list"
        return ToolPreflightResult(
            arguments=rewritten,
            repair_notes=[f"rewrote backtests_analysis(action='{action}') without job_id to action='list'"],
        )
    if action != "start":
        return ToolPreflightResult(arguments=arguments)
    if str(arguments.get("run_id") or "").strip():
        run_id = str(arguments.get("run_id") or "").strip()
        if "@" in run_id:
            rewritten = _deepcopy_jsonable(arguments)
            rewritten["action"] = "list"
            rewritten.pop("run_id", None)
            return ToolPreflightResult(
                arguments=rewritten,
                repair_notes=[f"rewrote backtests_analysis run_id='{run_id}' (snapshot-like ref) to action='list'"],
            )
        return ToolPreflightResult(arguments=arguments)
    rewritten = _deepcopy_jsonable(arguments)
    rewritten["action"] = "list"
    return ToolPreflightResult(
        arguments=rewritten,
        repair_notes=["rewrote backtests_analysis(action='start') without run_id to action='list'"],
    )


def _coerce_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return _deepcopy_jsonable(value)
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return _deepcopy_jsonable(parsed)
    return value


def _deepcopy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _normalize_timeframe_token(token: str) -> str:
    text = str(token or "").strip().lower()
    if not text:
        return token
    if text.endswith(("m", "h", "d", "w")):
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text
    value = int(digits)
    if value < 60:
        return f"{value}m"
    if value % 60 == 0 and value < 1440:
        return f"{value // 60}h"
    if value % 1440 == 0:
        return f"{value // 1440}d"
    return f"{value}m"


__all__ = ["ToolPreflightResult", "preflight_direct_tool_call"]
