"""
Cheap extraction of typed ids and async handles from accumulated known facts.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


_PLAN_OR_SLICE_ID_RE = re.compile(r"^(?:compiled_plan|plan|slice)_[A-Za-z0-9_:-]+$")


@dataclass(frozen=True)
class ConfirmedHandles:
    project_ids: tuple[str, ...] = ()
    branch_ids: tuple[str, ...] = ()
    default_branch_ids: tuple[str, ...] = ()
    research_record_operation_ids: tuple[str, ...] = ()
    experiment_job_ids: tuple[str, ...] = ()
    experiment_operation_ids: tuple[str, ...] = ()

    @property
    def preferred_project_id(self) -> str:
        return self.project_ids[0] if self.project_ids else ""

    @property
    def preferred_branch_id(self) -> str:
        if self.default_branch_ids:
            return self.default_branch_ids[0]
        return self.branch_ids[0] if self.branch_ids else ""

    @property
    def non_experiment_async_ids(self) -> set[str]:
        return set(self.research_record_operation_ids)


def is_plan_or_slice_id(value: str) -> bool:
    text = str(value or "").strip()
    return bool(text and _PLAN_OR_SLICE_ID_RE.match(text))


def extract_confirmed_handles(known_facts: Mapping[str, Any] | None) -> ConfirmedHandles:
    project_ids: list[str] = []
    branch_ids: list[str] = []
    default_branch_ids: list[str] = []
    research_record_operation_ids: list[str] = []
    experiment_job_ids: list[str] = []
    experiment_operation_ids: list[str] = []

    for key, value in dict(known_facts or {}).items():
        text = str(value or "").strip()
        if not text:
            continue
        normalized_key = str(key or "").strip().lower()

        if normalized_key in {
            "research_project.project_id",
            "project.project_id",
            "state_summary.project_id",
            "project_id",
        }:
            _append_unique(project_ids, text)
            continue
        if normalized_key.endswith(".project_id") and normalized_key.startswith("research_project."):
            _append_unique(project_ids, text)
            continue

        if normalized_key in {"research_project.default_branch_id", "project.default_branch_id"}:
            _append_unique(default_branch_ids, text)
            _append_unique(branch_ids, text)
            continue
        if normalized_key in {"research_project.branch_id", "project.branch_id", "branch_id"}:
            if text.startswith("branch-"):
                _append_unique(branch_ids, text)
            continue

        if normalized_key in {"research_record.operation_id", "research_record.resume_token"}:
            _append_unique(research_record_operation_ids, text)
            continue
        if normalized_key.endswith(".operation_id") and normalized_key.startswith("research_record."):
            _append_unique(research_record_operation_ids, text)
            continue

        if normalized_key in {"experiments_run.job_id", "experiments.job_id", "experiments_inspect.job_id"}:
            _append_unique(experiment_job_ids, text)
            continue
        if normalized_key in {"experiments_run.operation_id", "experiments.operation_id"}:
            _append_unique(experiment_operation_ids, text)
            continue

    return ConfirmedHandles(
        project_ids=tuple(project_ids),
        branch_ids=tuple(branch_ids),
        default_branch_ids=tuple(default_branch_ids),
        research_record_operation_ids=tuple(research_record_operation_ids),
        experiment_job_ids=tuple(experiment_job_ids),
        experiment_operation_ids=tuple(experiment_operation_ids),
    )


def format_confirmed_handles(known_facts: Mapping[str, Any] | None) -> list[str]:
    handles = extract_confirmed_handles(known_facts)
    lines: list[str] = []
    if handles.project_ids:
        lines.append(f"- research_project.project_id={handles.project_ids[0]}")
    if handles.default_branch_ids:
        lines.append(f"- research_project.default_branch_id={handles.default_branch_ids[0]}")
    elif handles.branch_ids:
        lines.append(f"- research_project.branch_id={handles.branch_ids[0]}")
    if handles.research_record_operation_ids:
        lines.append(f"- research_record.operation_id={handles.research_record_operation_ids[0]}")
    if handles.experiment_job_ids:
        lines.append(f"- experiments_run.job_id={handles.experiment_job_ids[0]}")
    if handles.experiment_operation_ids:
        lines.append(f"- experiments_run.operation_id={handles.experiment_operation_ids[0]}")
    return lines[:6]


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


__all__ = [
    "ConfirmedHandles",
    "extract_confirmed_handles",
    "format_confirmed_handles",
    "is_plan_or_slice_id",
]
