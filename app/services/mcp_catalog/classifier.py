"""
Deterministic, schema-driven heuristics for MCP tool operational semantics.
"""

from __future__ import annotations

from typing import Any

from app.services.mcp_catalog.models import McpCatalogSnapshot

_HANDLE_FIELDS = (
    "project_id",
    "job_id",
    "run_id",
    "snapshot_id",
    "operation_id",
    "branch_id",
)

_MUTATING_ACTIONS = {
    "add",
    "apply",
    "archive",
    "backfill",
    "build",
    "cancel",
    "capture",
    "cleanup_apply",
    "clone",
    "create",
    "delete",
    "hard_reset",
    "mark",
    "materialize",
    "organize",
    "plan",
    "promote_version",
    "publish",
    "record_attempt",
    "refresh",
    "remove",
    "rename",
    "repair_apply",
    "reindex",
    "reset_store",
    "resolve_violation",
    "run",
    "save_version",
    "scaffold",
    "set_baseline",
    "shutdown",
    "start",
    "stop",
    "sync",
    "tidy",
    "train",
    "unmark",
    "update",
    "update_card",
    "validate",
}

_DISCOVERY_ACTIONS = {
    "analytics",
    "backfill_preview",
    "catalog",
    "compare_summary",
    "contract",
    "describe",
    "detail",
    "heatmap",
    "inspect",
    "list",
    "meta",
    "portability",
    "preview",
    "raw",
    "render",
    "result",
    "rows",
    "search",
    "source",
    "status",
    "summary",
    "validation",
    "view",
}

_EXPENSIVE_ACTIONS = {
    "backfill",
    "build",
    "materialize",
    "publish",
    "refresh",
    "run",
    "start",
    "sync",
    "train",
}

_EXPENSIVE_DESC_TOKENS = (
    "background",
    "backfill",
    "build",
    "expensive",
    "materialize",
    "publish",
    "refresh",
    "study",
    "sync",
    "train",
    "walk-forward",
)

_PREFIX_FAMILY_MAP = {
    "research": "research_memory",
    "datasets": "data_readiness",
    "events": "events",
    "features": "feature_contract",
    "models": "modeling",
    "experiments": "experiments",
    "notify": "finalization",
    "signal": "finalization",
    "backtests": "backtesting",
    "system": "finalization",
    "incidents": "finalization",
}


def derive_tool_semantics(
    *,
    name: str,
    description: str,
    schema: dict[str, Any],
    manifest_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = dict(manifest_entry or {})
    action_enum = _enum_values(schema, "action")
    view_enum = _enum_values(schema, "view")
    accepted_handle_fields = _accepted_handle_fields(schema)
    side_effects = str(entry.get("side_effects") or "").strip() or _infer_side_effects(
        name=name,
        description=description,
        action_enum=action_enum,
    )
    async_like = _infer_async_like(action_enum=action_enum, view_enum=view_enum)
    supports_polling = bool(entry.get("supports_polling", False)) or _infer_supports_polling(
        action_enum=action_enum,
        view_enum=view_enum,
    )
    supports_discovery = bool(entry.get("supports_discovery", False)) or _infer_supports_discovery(
        action_enum=action_enum,
        view_enum=view_enum,
        description=description,
    )
    supports_terminal_write = bool(entry.get("supports_terminal_write", False)) or _infer_supports_terminal_write(
        action_enum=action_enum,
        side_effects=side_effects,
        schema=schema,
    )
    return {
        "cost_class": str(entry.get("cost_class") or "").strip() or _infer_cost_class(
            description=description,
            action_enum=action_enum,
            side_effects=side_effects,
            async_like=async_like,
        ),
        "side_effects": side_effects,
        "async_pattern": str(entry.get("async_pattern") or "").strip() or _infer_async_pattern(
            action_enum=action_enum,
            view_enum=view_enum,
        ),
        "id_fields": _normalize_string_list(entry.get("id_fields")) or _infer_id_fields(schema),
        "recommended_discovery_flow": _normalize_string_list(entry.get("recommended_discovery_flow")),
        "replaces_tools": _normalize_string_list(entry.get("replaces_tools")),
        "deprecated": bool(entry.get("deprecated", False)),
        "stability": str(entry.get("stability") or "stable").strip() or "stable",
        "family": (
            str(entry.get("family") or "").strip()
            or str(entry.get("domain") or "").strip()
            or infer_tool_family(name=name, description=description)
        ),
        "accepted_handle_fields": accepted_handle_fields,
        "produced_handle_fields": _normalize_string_list(entry.get("produced_handle_fields")) or _infer_produced_handle_fields(
            description=description,
            schema=schema,
            action_enum=action_enum,
            async_like=async_like,
        ),
        "supports_terminal_write": supports_terminal_write,
        "supports_discovery": supports_discovery,
        "supports_polling": supports_polling,
        "async_like": async_like,
    }


def infer_tool_family(*, name: str, description: str) -> str:
    lowered = f"{name} {description}".lower()
    if any(token in lowered for token in ("analysis", "diagnostic", "verdict", "layer compare", "ownership")):
        return "analysis"
    prefix = str(name or "").split("_", 1)[0].strip().lower()
    if prefix in _PREFIX_FAMILY_MAP:
        return _PREFIX_FAMILY_MAP[prefix]
    if any(token in lowered for token in ("project", "research", "hypothesis", "atlas", "memory")):
        return "research_memory"
    if any(token in lowered for token in ("dataset", "catalog", "timeframe", "instrument")):
        return "data_readiness"
    if any(token in lowered for token in ("event", "funding", "expiry")):
        return "events"
    if any(token in lowered for token in ("feature", "analytics", "column")):
        return "feature_contract"
    if any(token in lowered for token in ("model", "training", "classifier")):
        return "modeling"
    if any(token in lowered for token in ("experiment", "artifact", "run(ctx)")):
        return "experiments"
    if any(token in lowered for token in ("analysis", "diagnostic", "verdict", "layer compare", "ownership")):
        return "analysis"
    if any(token in lowered for token in ("backtest", "walk-forward", "oos", "diagnostic", "layer compare")):
        return "backtesting" if "analysis" not in lowered else "analysis"
    if any(token in lowered for token in ("notify", "binding", "incident", "health", "workspace", "logs", "queue")):
        return "finalization"
    return ""


def build_family_tool_map(snapshot: McpCatalogSnapshot) -> dict[str, list[str]]:
    family_map = {
        "research_memory": [],
        "data_readiness": [],
        "feature_contract": [],
        "modeling": [],
        "backtesting": [],
        "analysis": [],
        "events": [],
        "experiments": [],
        "finalization": [],
    }
    domain_overrides = _manifest_domain_overrides(snapshot)
    for tool in snapshot.tools:
        family = domain_overrides.get(tool.name) or tool.family or infer_tool_family(name=tool.name, description=tool.description)
        if family in family_map and tool.name not in family_map[family]:
            family_map[family].append(tool.name)
    for names in family_map.values():
        names.sort()
    return family_map


def is_expensive_tool(snapshot: McpCatalogSnapshot, tool_name: str) -> bool:
    tool = snapshot.get_tool(tool_name)
    return bool(tool and tool.cost_class == "expensive")


_READ_ONLY_ACTIONS = frozenset({
    "inspect",
    "list",
    "search",
    "status",
    "compare",
    "prove",
    "find",
    "read",
})


def is_expensive_tool_call(
    snapshot: McpCatalogSnapshot,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> bool:
    """Check whether a specific tool call is expensive, considering the action.

    Tools like ``features_custom`` and ``features_dataset`` are classified as
    expensive at the tool level because they support mutating operations
    (publish, build).  However, read-only calls (inspect, list, status) should
    NOT consume the expensive budget — only actual mutations should.
    """
    if not is_expensive_tool(snapshot, tool_name):
        return False
    if not arguments or not isinstance(arguments, dict):
        return True
    action = str(arguments.get("action") or "").strip().lower()
    if not action:
        # No action specified — default to expensive for safety.
        return True
    return action not in _READ_ONLY_ACTIONS


def is_mutating_tool(snapshot: McpCatalogSnapshot, tool_name: str) -> bool:
    tool = snapshot.get_tool(tool_name)
    return bool(tool and tool.side_effects == "mutating")


def _infer_side_effects(*, name: str, description: str, action_enum: list[str]) -> str:
    lowered = f"{name} {description}".lower()
    if action_enum:
        if any(item in _MUTATING_ACTIONS for item in action_enum):
            return "mutating"
        if action_enum and set(action_enum).issubset(_DISCOVERY_ACTIONS):
            return "read_only"
    if any(token in lowered for token in ("apply", "delete", "publish", "train", "sync", "start", "create", "update", "remove", "record", "build")):
        return "mutating"
    return "read_only"


def _infer_cost_class(*, description: str, action_enum: list[str], side_effects: str, async_like: bool) -> str:
    lowered = str(description or "").lower()
    if any(action in _EXPENSIVE_ACTIONS for action in action_enum):
        return "expensive"
    if async_like and side_effects == "mutating":
        return "expensive"
    if any(token in lowered for token in _EXPENSIVE_DESC_TOKENS):
        return "expensive"
    return "cheap"


def _infer_async_pattern(*, action_enum: list[str], view_enum: list[str]) -> str:
    lowered_actions = {item.lower() for item in action_enum}
    lowered_views = {item.lower() for item in view_enum}
    if {"start", "status", "result"}.issubset(lowered_actions):
        return "start/status/result"
    if {"run", "status", "result"}.issubset(lowered_actions):
        return "run/status/result"
    if {"start", "cancel"}.issubset(lowered_actions) and "status" in lowered_actions:
        return "start/status/cancel"
    if {"status", "detail"} <= lowered_views:
        return "inspect/status/detail"
    return ""


def _infer_async_like(*, action_enum: list[str], view_enum: list[str]) -> bool:
    lowered_actions = {item.lower() for item in action_enum}
    lowered_views = {item.lower() for item in view_enum}
    if {"start", "status"} <= lowered_actions:
        return True
    if {"run", "status"} <= lowered_actions:
        return True
    if "wait" in lowered_actions or "status" in lowered_actions:
        return True
    return "status" in lowered_views and "result" in lowered_views


def _infer_supports_polling(*, action_enum: list[str], view_enum: list[str]) -> bool:
    lowered_actions = {item.lower() for item in action_enum}
    lowered_views = {item.lower() for item in view_enum}
    return "status" in lowered_actions or "result" in lowered_actions or "status" in lowered_views


def _infer_supports_discovery(*, action_enum: list[str], view_enum: list[str], description: str) -> bool:
    lowered_actions = {item.lower() for item in action_enum}
    lowered_views = {item.lower() for item in view_enum}
    lowered_desc = str(description or "").lower()
    if lowered_actions & _DISCOVERY_ACTIONS:
        return True
    if lowered_views & _DISCOVERY_ACTIONS:
        return True
    return any(token in lowered_desc for token in ("inspect", "list", "summary", "preview", "catalog", "detail", "search"))


def _infer_supports_terminal_write(*, action_enum: list[str], side_effects: str, schema: dict[str, Any]) -> bool:
    if side_effects != "mutating":
        return False
    if any(action in {"create", "update", "add", "publish", "apply", "record_attempt", "materialize"} for action in action_enum):
        return True
    props = schema.get("properties")
    if not isinstance(props, dict):
        return False
    return any(name in props for name in ("record", "project", "payload"))


def _accepted_handle_fields(schema: dict[str, Any]) -> list[str]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []
    return sorted(field for field in _HANDLE_FIELDS if field in properties)


def _infer_produced_handle_fields(
    *,
    description: str,
    schema: dict[str, Any],
    action_enum: list[str],
    async_like: bool,
) -> list[str]:
    produced: list[str] = []
    lowered_desc = str(description or "").lower()
    accepted = set(_accepted_handle_fields(schema))
    for field_name in _HANDLE_FIELDS:
        if field_name in accepted and field_name != "branch_id":
            continue
        if field_name in lowered_desc:
            produced.append(field_name)
    if async_like:
        for field_name in ("job_id", "run_id", "operation_id"):
            if field_name not in produced and field_name not in accepted:
                produced.append(field_name)
                break
    if any(action in {"create", "clone", "save_version", "publish"} for action in action_enum):
        for field_name in ("snapshot_id", "project_id", "operation_id"):
            if field_name not in produced and field_name not in accepted:
                produced.append(field_name)
                break
    return produced


def _infer_id_fields(schema: dict[str, Any]) -> list[str]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []
    result: list[str] = []
    for key in properties:
        lowered = str(key or "").strip().lower()
        if lowered.endswith("_id") or lowered in _HANDLE_FIELDS:
            result.append(str(key))
    return sorted(result)


def _enum_values(schema: dict[str, Any], field_name: str) -> list[str]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []
    field_schema = properties.get(field_name)
    if not isinstance(field_schema, dict):
        return []
    enum_values = field_schema.get("enum")
    if isinstance(enum_values, list):
        return [str(item).strip().lower() for item in enum_values if str(item).strip()]
    return []


def _normalize_string_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if value else []
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _manifest_domain_overrides(snapshot: McpCatalogSnapshot) -> dict[str, str]:
    manifest = snapshot.capability_manifest or snapshot.bootstrap_manifest
    data = manifest.get("data") if isinstance(manifest, dict) and isinstance(manifest.get("data"), dict) else manifest
    if not isinstance(data, dict):
        return {}
    overrides: dict[str, str] = {}
    for item in data.get("contract_index", []) or []:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool") or "").strip()
        domain = str(item.get("domain") or "").strip()
        if tool_name and domain:
            overrides[tool_name] = infer_tool_family(name=domain, description=domain)
    return overrides


__all__ = [
    "build_family_tool_map",
    "derive_tool_semantics",
    "infer_tool_family",
    "is_expensive_tool",
    "is_expensive_tool_call",
    "is_mutating_tool",
]
