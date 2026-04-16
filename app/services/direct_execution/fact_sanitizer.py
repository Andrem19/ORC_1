"""
Sanitize model-supplied facts and evidence refs before quality gates.
"""

from __future__ import annotations

from typing import Any

from app.services.direct_execution.handle_hygiene import (
    DURABLE_HANDLE_FIELDS,
    is_suspicious_handle_value,
    is_transport_id_field,
    looks_like_numeric_identifier,
    looks_like_session_identifier,
)

_REF_LIST_KEYS = frozenset(
    {
        "direct.created_ids",
        "direct.write_ids",
        "direct.evidence_refs",
        "direct.supported_evidence_refs",
        "analysis_refs",
        "integration_refs",
        "backtests.analysis_refs",
        "backtests.integration_refs",
        "evidence_refs",
        "raw_refs",
    }
)
_REF_KEY_SUFFIXES = (
    "evidence_refs",
    "analysis_refs",
    "integration_refs",
    "created_ids",
    "write_ids",
    "supported_evidence_refs",
    "raw_refs",
)


def sanitize_evidence_refs(refs: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    """Remove transport/session ids while preserving durable refs and transcript refs."""

    cleaned: list[str] = []
    for item in refs or []:
        text = str(item or "").strip()
        if not text or _is_transport_like_ref(text):
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def sanitize_fact_payload(facts: dict[str, Any] | None) -> dict[str, Any]:
    """Drop transport ids from facts without touching normal metrics."""

    sanitized: dict[str, Any] = {}
    for raw_key, raw_value in dict(facts or {}).items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        normalized_key = key.lower()
        if is_transport_id_field(normalized_key):
            continue
        if normalized_key in DURABLE_HANDLE_FIELDS and is_suspicious_handle_value(raw_value, field_name=normalized_key):
            continue
        if _is_ref_list_key(normalized_key):
            refs = sanitize_evidence_refs(_as_list(raw_value))
            if refs:
                sanitized[key] = refs
            continue
        value = _sanitize_value(raw_value, key_hint=normalized_key)
        if _is_empty(value):
            continue
        sanitized[key] = value
    return sanitized


def _sanitize_value(value: Any, *, key_hint: str) -> Any:
    if isinstance(value, dict):
        nested: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            normalized_key = key.lower()
            if is_transport_id_field(normalized_key):
                continue
            if normalized_key in DURABLE_HANDLE_FIELDS and is_suspicious_handle_value(raw_value, field_name=normalized_key):
                continue
            if _is_ref_list_key(normalized_key):
                refs = sanitize_evidence_refs(_as_list(raw_value))
                if refs:
                    nested[key] = refs
                continue
            nested_value = _sanitize_value(raw_value, key_hint=normalized_key)
            if not _is_empty(nested_value):
                nested[key] = nested_value
        return nested
    if isinstance(value, (list, tuple, set)):
        if _is_ref_list_key(key_hint):
            return sanitize_evidence_refs(list(value))
        cleaned: list[Any] = []
        for item in value:
            nested = _sanitize_value(item, key_hint=key_hint)
            if not _is_empty(nested):
                cleaned.append(nested)
        return cleaned
    return value


def _is_ref_list_key(key: str) -> bool:
    normalized = str(key or "").strip().lower()
    return normalized in _REF_LIST_KEYS or any(normalized.endswith(suffix) for suffix in _REF_KEY_SUFFIXES)


def _is_transport_like_ref(value: str) -> bool:
    text = str(value or "").strip()
    if text.startswith("transcript:"):
        return False
    return looks_like_numeric_identifier(text) or looks_like_session_identifier(text)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if value is None:
        return []
    return [value]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not bool(value)
    return False


__all__ = ["sanitize_evidence_refs", "sanitize_fact_payload"]
