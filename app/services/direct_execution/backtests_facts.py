"""
Canonical backtests fact normalization for cross-slice handoff.
"""

from __future__ import annotations

from typing import Any

_CANDIDATE_HANDLE_SUFFIXES = (
    "feature_long_job",
    "feature_short_job",
    "candidate_job_id",
    "candidate_run_id",
    "long_job_id",
    "short_job_id",
)
_ANALYSIS_REF_SUFFIXES = (
    "diagnostics_run",
    "analysis_job_id",
    "analysis_run_id",
    "ownership_analysis_id",
)
_INTEGRATION_HANDLE_SUFFIXES = (
    "run_id",
    "run_ids",
    "base_run_id",
    "baseline_run_id",
    "candidate_run_id",
    "comparison_run_id",
)
_INTEGRATION_REF_SUFFIXES = (
    "analysis_id",
    "analysis_job_id",
    "analysis_run_id",
    "compare_summary_id",
    "ownership_analysis_id",
)


def normalize_backtests_facts(
    facts: dict[str, Any] | None,
    *,
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    normalized = dict(facts or {})
    _copy_alias(normalized, alias="candidate_handles", canonical="backtests.candidate_handles")
    _copy_alias(normalized, alias="analysis_refs", canonical="backtests.analysis_refs")
    _copy_alias(normalized, alias="integration_handles", canonical="backtests.integration_handles")
    _copy_alias(normalized, alias="integration_refs", canonical="backtests.integration_refs")
    candidate_handles = normalized.get("backtests.candidate_handles")
    if _is_missing(candidate_handles):
        derived_candidate_handles = _derive_mapping(normalized, suffixes=_CANDIDATE_HANDLE_SUFFIXES)
        if derived_candidate_handles:
            normalized["backtests.candidate_handles"] = derived_candidate_handles
    analysis_refs = normalized.get("backtests.analysis_refs")
    if _is_missing(analysis_refs):
        derived_analysis_refs = _derive_refs(
            normalized,
            suffixes=_ANALYSIS_REF_SUFFIXES,
            evidence_refs=evidence_refs,
        )
        if derived_analysis_refs:
            normalized["backtests.analysis_refs"] = derived_analysis_refs
    integration_handles = normalized.get("backtests.integration_handles")
    if _is_missing(integration_handles):
        derived_integration_handles = _derive_mapping(normalized, suffixes=_INTEGRATION_HANDLE_SUFFIXES)
        if derived_integration_handles:
            normalized["backtests.integration_handles"] = derived_integration_handles
    integration_refs = normalized.get("backtests.integration_refs")
    if _is_missing(integration_refs):
        derived_integration_refs = _derive_refs(
            normalized,
            suffixes=_INTEGRATION_REF_SUFFIXES,
            evidence_refs=evidence_refs,
        )
        if derived_integration_refs:
            normalized["backtests.integration_refs"] = derived_integration_refs
    return normalized


def _copy_alias(normalized: dict[str, Any], *, alias: str, canonical: str) -> None:
    if not _is_missing(normalized.get(canonical)):
        return
    value = normalized.get(alias)
    if not _is_missing(value):
        normalized[canonical] = value


def _derive_mapping(facts: dict[str, Any], *, suffixes: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in facts.items():
        if _is_missing(value):
            continue
        key_text = str(key or "").strip()
        suffix = _matching_suffix(key_text, suffixes=suffixes)
        if not suffix:
            continue
        result.setdefault(suffix, value)
    return result


def _derive_refs(
    facts: dict[str, Any],
    *,
    suffixes: tuple[str, ...],
    evidence_refs: list[str] | None,
) -> list[str]:
    refs: list[str] = []
    for key, value in facts.items():
        if _is_missing(value):
            continue
        key_text = str(key or "").strip()
        if not _matching_suffix(key_text, suffixes=suffixes):
            continue
        _append_refs(refs, value)
    for item in evidence_refs or []:
        text = str(item or "").strip()
        if text and text not in refs:
            refs.append(text)
    return refs


def _append_refs(target: list[str], value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _append_refs(target, item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _append_refs(target, item)
        return
    text = str(value or "").strip()
    if text and text not in target:
        target.append(text)


def _matching_suffix(key_text: str, *, suffixes: tuple[str, ...]) -> str:
    normalized = str(key_text or "").strip()
    for suffix in suffixes:
        if normalized == suffix or normalized.endswith(f".{suffix}"):
            return suffix
    return ""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not bool(value)
    return False


__all__ = ["normalize_backtests_facts"]
