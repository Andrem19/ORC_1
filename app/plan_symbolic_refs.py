"""
Symbolic references for plan-mode task instructions.

Allows planner-authored tasks to reference outputs from earlier stages
without inventing concrete future IDs at plan creation time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from app.plan_models import TaskReport

SYMBOLIC_REF_PATTERN = re.compile(r"\{\{stage:(\d+)\.([A-Za-z0-9_\[\]\.-]+)\}\}")
STEP_REF_PATTERN = re.compile(r"\{\{step:([A-Za-z_][A-Za-z0-9_-]*)\.([A-Za-z0-9_\[\]\.-]+)\}\}")
RESULTS_TABLE_FIELD_PATTERN = re.compile(r"results_table\[0\]\.([A-Za-z_][\w-]*)$")

SUPPORTED_DIRECT_FIELDS = {
    "run_id",
    "snapshot_id",
    "version",
    "snapshot_ref",
}


@dataclass(frozen=True)
class SymbolicReference:
    """One symbolic reference embedded in a task instruction."""

    raw: str
    stage_number: int
    field: str


@dataclass(frozen=True)
class StepReference:
    """One intra-stage symbolic reference embedded in a task step."""

    raw: str
    step_id: str
    field: str


@dataclass
class SymbolicResolutionError:
    """One failed symbolic-reference resolution."""

    raw: str
    stage_number: int
    field: str
    message: str


@dataclass
class SymbolicResolutionResult:
    """Resolved instruction text plus any resolution errors."""

    resolved_text: str
    unresolved: list[SymbolicResolutionError] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return len(self.unresolved) == 0


def extract_symbolic_references(text: str) -> list[SymbolicReference]:
    """Extract all symbolic references from an instruction string."""
    refs: list[SymbolicReference] = []
    for match in SYMBOLIC_REF_PATTERN.finditer(text):
        refs.append(
            SymbolicReference(
                raw=match.group(0),
                stage_number=int(match.group(1)),
                field=match.group(2),
            )
        )
    return refs


def extract_step_references(text: str) -> list[StepReference]:
    refs: list[StepReference] = []
    for match in STEP_REF_PATTERN.finditer(text):
        refs.append(
            StepReference(
                raw=match.group(0),
                step_id=match.group(1),
                field=match.group(2),
            )
        )
    return refs


def is_supported_symbolic_field(field: str) -> bool:
    if field in SUPPORTED_DIRECT_FIELDS:
        return True
    return RESULTS_TABLE_FIELD_PATTERN.match(field) is not None


def resolve_symbolic_references(
    text: str,
    reports_by_stage: Mapping[int, TaskReport],
) -> SymbolicResolutionResult:
    """Resolve all symbolic references in one instruction string."""
    refs = extract_symbolic_references(text)
    if not refs:
        return SymbolicResolutionResult(resolved_text=text)

    resolved_text = text
    errors: list[SymbolicResolutionError] = []

    for ref in refs:
        report = reports_by_stage.get(ref.stage_number)
        if report is None:
            errors.append(
                SymbolicResolutionError(
                    raw=ref.raw,
                    stage_number=ref.stage_number,
                    field=ref.field,
                    message=f"Stage {ref.stage_number} has no report yet",
                )
            )
            continue

        value = _value_from_report(report, ref.field)
        if value is None:
            errors.append(
                SymbolicResolutionError(
                    raw=ref.raw,
                    stage_number=ref.stage_number,
                    field=ref.field,
                    message=(
                        f"Stage {ref.stage_number} report does not expose "
                        f"field '{ref.field}'"
                    ),
                )
            )
            continue

        resolved_text = resolved_text.replace(ref.raw, str(value))

    return SymbolicResolutionResult(
        resolved_text=resolved_text,
        unresolved=errors,
    )


def resolve_stage_references_in_value(
    value: Any,
    reports_by_stage: Mapping[int, TaskReport],
) -> tuple[Any, list[SymbolicResolutionError]]:
    """Resolve only cross-stage refs within strings, lists, and dicts."""
    if isinstance(value, str):
        result = resolve_symbolic_references(value, reports_by_stage)
        return result.resolved_text, result.unresolved
    if isinstance(value, list):
        resolved_items: list[Any] = []
        errors: list[SymbolicResolutionError] = []
        for item in value:
            resolved_item, item_errors = resolve_stage_references_in_value(item, reports_by_stage)
            resolved_items.append(resolved_item)
            errors.extend(item_errors)
        return resolved_items, errors
    if isinstance(value, dict):
        resolved_dict: dict[str, Any] = {}
        errors: list[SymbolicResolutionError] = []
        for key, item in value.items():
            resolved_item, item_errors = resolve_stage_references_in_value(item, reports_by_stage)
            resolved_dict[key] = resolved_item
            errors.extend(item_errors)
        return resolved_dict, errors
    return value, []


def _value_from_report(report: TaskReport, field: str) -> Any | None:
    row0 = report.results_table[0] if report.results_table else {}
    if not isinstance(row0, dict):
        row0 = {}

    if field == "run_id":
        return row0.get("run_id") or _find_artifact_value(report, "run_id:")
    if field == "snapshot_id":
        return row0.get("snapshot_id") or _snapshot_components(report)[0]
    if field == "version":
        if row0.get("version") is not None:
            return row0.get("version")
        snapshot_ref = _find_artifact_value(report, "snapshot:")
        if snapshot_ref and "@" in snapshot_ref:
            return snapshot_ref.rsplit("@", 1)[1]
        return None
    if field == "snapshot_ref":
        if row0.get("snapshot_ref"):
            return row0.get("snapshot_ref")
        snapshot_id = row0.get("snapshot_id")
        version = row0.get("version")
        if snapshot_id and version is not None:
            return f"{snapshot_id}@{version}"
        return _find_artifact_value(report, "snapshot:")

    match = RESULTS_TABLE_FIELD_PATTERN.match(field)
    if match:
        return row0.get(match.group(1))

    return None


def _find_artifact_value(report: TaskReport, prefix: str) -> str | None:
    for artifact in report.artifacts:
        if isinstance(artifact, str) and artifact.startswith(prefix):
            return artifact.split(":", 1)[1].strip()
    return None


def _snapshot_components(report: TaskReport) -> tuple[str | None, str | None]:
    snapshot_ref = _find_artifact_value(report, "snapshot:")
    if not snapshot_ref:
        return None, None
    if "@" not in snapshot_ref:
        return snapshot_ref, None
    snapshot_id, version = snapshot_ref.rsplit("@", 1)
    return snapshot_id, version
