"""Subject extraction helpers for acceptance verification."""

from __future__ import annotations

import json
import re
from typing import Any

from app.execution_models import WorkerAction

_RUN_ID_RE = re.compile(r"\b\d{8}-\d{6}-[0-9a-f]{8,}\b")
_NODE_ID_RE = re.compile(r"\b(?:node|note|incident)_[A-Za-z0-9_-]+\b|(?<!-)\bnode-[A-Za-z0-9_-]+\b")

# JSON field names that match _NODE_ID_RE but are NOT real node IDs.
_NODE_ID_FALSE_POSITIVES = frozenset({
    "node_id", "node_type", "node_refs", "node-id", "node_ids", "node_count",
    "note_id", "note_ids", "note_text",
    "incident_id", "incident_ids",
})


def run_ids_from_action_and_transcript(action: WorkerAction, transcript: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    facts = getattr(action, "facts", {}) or {}
    if isinstance(facts, dict):
        _append_strings(ids, _values_for_keys(facts, {"run_id", "base_run_id", "candidate_run_id", "diagnostics_run"}))
        for key in ("run_ids", "candidate_run_ids", "backtests.run_ids"):
            raw = facts.get(key)
            if isinstance(raw, list):
                _append_strings(ids, raw)
        for raw in facts.values():
            if isinstance(raw, str):
                _append_strings(ids, _RUN_ID_RE.findall(raw))
    _append_strings(ids, getattr(action, "evidence_refs", []) or [])
    for item in transcript:
        if not isinstance(item, dict):
            continue
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        _append_strings(ids, _values_for_keys(args, {"run_id", "base_run_id", "candidate_run_id"}))
    return [item for item in ids if _RUN_ID_RE.fullmatch(item)]


def node_ids_from_action_and_transcript(action: WorkerAction, transcript: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    facts = getattr(action, "facts", {}) or {}
    if isinstance(facts, dict):
        _append_strings(ids, _values_for_keys(facts, {"node_id", "research.memory_node_id"}))
        for key in ("research.hypothesis_refs", "direct.created_ids"):
            raw = facts.get(key)
            if isinstance(raw, list):
                _append_strings(ids, raw)
        _append_strings(ids, _NODE_ID_RE.findall(str(facts)))
    _append_strings(ids, getattr(action, "evidence_refs", []) or [])
    for item in transcript:
        if not isinstance(item, dict):
            continue
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        _append_strings(ids, _values_for_keys(args, {"node_id"}))
        _append_strings(ids, _NODE_ID_RE.findall(str(payload)))
    return [item for item in ids if item.startswith(("node", "note", "incident")) and item not in _NODE_ID_FALSE_POSITIVES]


def feature_names_from_action_and_transcript(action: WorkerAction, transcript: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    facts = getattr(action, "facts", {}) or {}
    if isinstance(facts, dict):
        for key in ("feature_name", "custom_feature_name", "published_feature_name"):
            value = facts.get(key)
            if isinstance(value, str):
                names.append(value)
        for key in ("features", "feature_names", "required_columns", "columns"):
            raw = facts.get(key)
            if isinstance(raw, list):
                _append_strings(names, [item for item in raw if str(item).startswith("cf_")])
    for item in transcript:
        if item.get("tool") != "features_custom":
            continue
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        name = str(args.get("name") or "").strip()
        if name:
            names.append(name)
        # Fallback: extract feature names from response payloads when the model
        # called features_custom(action='inspect', view='list') but did not pass
        # a specific name argument or report names in its facts.
        if not name:
            _append_strings(names, _extract_cf_names_from_payload(item.get("payload")))
    return _unique([item for item in names if item])


def model_refs_from_action_and_transcript(action: WorkerAction, transcript: list[dict[str, Any]]) -> tuple[list[str], list[tuple[str, str]]]:
    dataset_ids: list[str] = []
    versions: list[tuple[str, str]] = []
    facts = getattr(action, "facts", {}) or {}
    if isinstance(facts, dict):
        for key in ("dataset_id", "model_dataset_id"):
            value = str(facts.get(key) or "").strip()
            if value:
                dataset_ids.append(value)
        model_id = str(facts.get("model_id") or "").strip()
        version = str(facts.get("version") or facts.get("model_version") or "").strip()
        if model_id and version:
            versions.append((model_id, version))
    for item in transcript:
        if not isinstance(item, dict):
            continue
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        data = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        for source in (args, data):
            dataset_id = str(source.get("dataset_id") or "").strip()
            if dataset_id:
                dataset_ids.append(dataset_id)
            model_id = str(source.get("model_id") or "").strip()
            version = str(source.get("version") or "").strip()
            if model_id and version:
                versions.append((model_id, version))
    return _unique(dataset_ids), list(dict.fromkeys(versions))


def successful_mutating_tool_count(action: WorkerAction) -> int:
    facts = getattr(action, "facts", {}) or {}
    if not isinstance(facts, dict):
        return 0
    try:
        return int(facts.get("direct.successful_mutating_tool_count") or 0)
    except Exception:
        return 0


def successful_tool_names(action: WorkerAction) -> set[str]:
    facts = getattr(action, "facts", {}) or {}
    raw = facts.get("direct.successful_tool_names") if isinstance(facts, dict) else None
    if not isinstance(raw, list):
        return set()
    return {str(item).strip() for item in raw if str(item).strip()}


def _values_for_keys(value: Any, keys: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in keys and isinstance(item, str):
                found.append(item)
            found.extend(_values_for_keys(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.extend(_values_for_keys(item, keys))
    return found


def _extract_cf_names_from_payload(payload: Any) -> list[str]:
    """Extract cf_ feature names from a features_custom MCP response payload."""
    if not isinstance(payload, dict):
        return []
    inner = payload.get("payload")
    if not isinstance(inner, dict):
        return []
    content = inner.get("content")
    if not isinstance(content, list):
        return []
    names: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = str(block.get("text") or "").strip()
        if not text.startswith("{"):
            continue
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        features = (parsed.get("data") or {}).get("features")
        if not isinstance(features, list):
            continue
        for feature in features:
            if not isinstance(feature, dict):
                continue
            name = str(feature.get("name") or "").strip()
            if name.startswith("cf_"):
                names.append(name)
    return names


def _append_strings(target: list[str], values: Any) -> None:
    for item in values or []:
        normalized = str(item or "").strip()
        if normalized and normalized not in target:
            target.append(normalized)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


__all__ = [
    "feature_names_from_action_and_transcript",
    "model_refs_from_action_and_transcript",
    "node_ids_from_action_and_transcript",
    "run_ids_from_action_and_transcript",
    "successful_mutating_tool_count",
    "successful_tool_names",
]
