"""
Hydrate and validate downstream-ready facts from direct final reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.execution_models import PlanSlice, WorkerAction


@dataclass(frozen=True)
class FinalReportReadiness:
    facts: dict[str, Any]
    missing_required_facts: list[str]


def hydrate_final_report_facts(
    *,
    slice_obj: PlanSlice,
    action: WorkerAction,
    required_output_facts: list[str],
    inherited_facts: dict[str, Any] | None = None,
) -> FinalReportReadiness:
    del slice_obj
    facts = dict(action.facts or {})
    for key, value in (inherited_facts or {}).items():
        facts.setdefault(str(key), value)
    project_id = str(facts.get("research.project_id") or facts.get("project_id") or "").strip()
    if not project_id:
        for key, value in facts.items():
            key_text = str(key or "").strip()
            if not key_text.endswith(".project_id") and not key_text.endswith(".research.project_id"):
                continue
            candidate = str(value or "").strip()
            if candidate:
                project_id = candidate
                break
    if project_id:
        facts.setdefault("research.project_id", project_id)
    shortlist = facts.get("research.shortlist_families")
    if not shortlist:
        shortlist = facts.get("valid_signal_types")
    if not shortlist:
        shortlist = _derive_shortlist_from_findings(action.findings)
        if shortlist:
            facts["research.shortlist_families"] = shortlist
    elif isinstance(shortlist, list):
        facts["research.shortlist_families"] = shortlist
    refs = facts.get("research.hypothesis_refs")
    if not refs:
        refs = _derive_refs(action.evidence_refs)
        if refs:
            facts["research.hypothesis_refs"] = refs
    missing = [key for key in required_output_facts if _is_missing(facts.get(key))]
    return FinalReportReadiness(facts=facts, missing_required_facts=missing)


def _derive_shortlist_from_findings(findings: list[str]) -> list[str]:
    values: list[str] = []
    for item in findings or []:
        text = str(item or "").strip()
        if not text:
            continue
        head = text.split(" - ", 1)[0].split(":", 1)[0].strip()
        if head and head not in values:
            values.append(head)
    return values[:20]


def _derive_refs(evidence_refs: list[str]) -> list[str]:
    return [str(item).strip() for item in evidence_refs or [] if str(item).strip().startswith("node")]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not bool(value)
    return False


__all__ = ["FinalReportReadiness", "hydrate_final_report_facts"]
