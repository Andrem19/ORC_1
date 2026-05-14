"""Deterministic contradiction checks for terminal slice reports."""

from __future__ import annotations

from typing import Any

from app.execution_models import WorkerAction
from app.services.direct_execution.acceptance.contracts import FAIL, PASS, PredicateResult


def contradiction_predicates(action: WorkerAction, forbidden: list[str]) -> list[PredicateResult]:
    if not forbidden:
        return []
    text = _action_text(action)
    predicates: list[PredicateResult] = []
    for phrase in forbidden:
        normalized = str(phrase or "").strip().lower()
        if not normalized:
            continue
        if normalized in text:
            predicates.append(
                PredicateResult(
                    id=f"forbidden_contradiction:{normalized}",
                    status=FAIL,
                    details={"matched": normalized},
                )
            )
    if not predicates:
        predicates.append(PredicateResult(id="forbidden_contradictions_absent", status=PASS))
    return predicates


def _action_text(action: WorkerAction) -> str:
    parts: list[str] = [
        str(getattr(action, "summary", "") or ""),
        str(getattr(action, "reason", "") or ""),
        str(getattr(action, "reason_code", "") or ""),
    ]
    for field_name in ("findings", "rejected_findings", "risks", "next_actions"):
        value = getattr(action, field_name, None)
        if isinstance(value, list):
            parts.extend(str(item or "") for item in value)
    facts = getattr(action, "facts", {}) or {}
    if isinstance(facts, dict):
        parts.append(_flatten(facts))
    return " ".join(parts).lower()


def _flatten(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_flatten(item) for item in value)
    return str(value or "")


__all__ = ["contradiction_predicates"]
