"""
Hydrate and validate downstream-ready facts from direct final reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.backtests_facts import normalize_backtests_facts
from app.services.direct_execution.fact_sanitizer import sanitize_evidence_refs, sanitize_fact_payload


@dataclass(frozen=True)
class FinalReportReadiness:
    facts: dict[str, Any]
    missing_required_facts: list[str]
    evidence_refs: list[str]


def hydrate_final_report_facts(
    *,
    slice_obj: PlanSlice,
    action: WorkerAction,
    required_output_facts: list[str],
    inherited_facts: dict[str, Any] | None = None,
) -> FinalReportReadiness:
    strict_shortlist = _is_research_shortlist_slice(slice_obj)
    sanitized_evidence_refs = sanitize_evidence_refs(list(action.evidence_refs or []))
    facts = normalize_backtests_facts(
        sanitize_fact_payload(dict(action.facts or {})),
        evidence_refs=sanitized_evidence_refs,
    )
    for key, value in (inherited_facts or {}).items():
        facts.setdefault(str(key), value)
    project_id = _extract_valid_project_id(facts)
    # Purge invalid project_id values (e.g. transcript:2:research_project)
    # so _canonical_project_id does not reinstate them.
    for purge_key in ("research.project_id", "project_id"):
        val = facts.get(purge_key)
        if val is not None and _is_invalid_project_id(str(val)):
            del facts[purge_key]
    if project_id:
        facts["research.project_id"] = project_id
        facts.setdefault("project_id", project_id)
    canonical_project_id = _canonical_project_id(facts)
    if canonical_project_id:
        facts["research.project_id"] = canonical_project_id
        facts.setdefault("project_id", canonical_project_id)
    shortlist = facts.get("research.shortlist_families")
    if not shortlist and not strict_shortlist:
        shortlist = facts.get("valid_signal_types")
    if isinstance(shortlist, list):
        facts["research.shortlist_families"] = shortlist
    if strict_shortlist:
        novelty = facts.get("research.novelty_justification_present")
        if isinstance(novelty, bool):
            facts["research.novelty_justification_present"] = novelty
    refs = facts.get("research.hypothesis_refs")
    if not refs and not strict_shortlist:
        refs = _derive_refs(action.evidence_refs)
        if refs:
            facts["research.hypothesis_refs"] = refs
    missing = [key for key in required_output_facts if _is_missing(facts.get(key))]
    return FinalReportReadiness(facts=facts, missing_required_facts=missing, evidence_refs=sanitized_evidence_refs)

def _derive_refs(evidence_refs: list[str]) -> list[str]:
    return [str(item).strip() for item in evidence_refs or [] if str(item).strip().startswith("node")]


def _is_research_shortlist_slice(slice_obj: PlanSlice) -> bool:
    return str(slice_obj.runtime_profile or "").strip() == "research_shortlist"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not bool(value)
    return False


_INVALID_PROJECT_ID_PREFIXES = (
    "transcript:",
    "node-",
    "note-",
    "dim-",
    "analysis-",
    "branch-",
)


def _is_invalid_project_id(value: str) -> bool:
    """Return True when a project_id value is clearly not a real project id."""
    text = str(value or "").strip()
    if not text:
        return True
    return text.startswith(_INVALID_PROJECT_ID_PREFIXES)


def _extract_valid_project_id(facts: dict[str, Any]) -> str:
    """Extract a valid project_id from facts, rejecting transcript refs and other invalid patterns."""
    # 1. Check explicit project_id fields
    for key in ("research.project_id", "project_id"):
        value = str(facts.get(key) or "").strip()
        if value and not _is_invalid_project_id(value):
            return value
    # 2. Scan all fact keys for project_id-like fields
    for key, value in facts.items():
        key_text = str(key or "").strip()
        if not key_text.endswith(".project_id") and not key_text.endswith(".research.project_id"):
            continue
        candidate = str(value or "").strip()
        if candidate and not _is_invalid_project_id(candidate):
            return candidate
    # 3. Rescue from created_ids / evidence_refs when research_project was called
    tool_names = facts.get("direct.successful_tool_names") or []
    if "research_project" in tool_names:
        for source_key in ("direct.created_ids", "direct.supported_evidence_refs"):
            for candidate in (facts.get(source_key) or []):
                candidate_text = str(candidate).strip()
                if candidate_text and not _is_invalid_project_id(candidate_text):
                    return candidate_text
    return ""


def _canonical_project_id(facts: dict[str, Any]) -> str:
    primary = str(facts.get("research.project_id") or "").strip()
    fallback = str(facts.get("project_id") or "").strip()
    if primary and fallback and primary != fallback:
        if _looks_like_transient_project_handle(primary) and not _looks_like_transient_project_handle(fallback):
            return fallback
    return primary or fallback


def _looks_like_transient_project_handle(value: str) -> bool:
    text = str(value or "").strip().lower()
    if len(text) < 16:
        return False
    return all(ch in "0123456789abcdef" for ch in text)


__all__ = ["FinalReportReadiness", "hydrate_final_report_facts"]
