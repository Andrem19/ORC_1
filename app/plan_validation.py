"""
Validation for structured research plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.plan_models import ResearchPlan
from app.plan_symbolic_refs import (
    extract_symbolic_references,
    has_legacy_placeholder,
    is_supported_symbolic_field,
)


@dataclass
class PlanValidationError:
    """One plan-validation failure."""

    stage_number: int
    code: str
    message: str
    offending_text: str = ""
    instruction_index: int | None = None


@dataclass
class PlanValidationResult:
    """Structured validation result for a research plan."""

    errors: list[PlanValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self, max_items: int = 3) -> str:
        if not self.errors:
            return "valid"
        shown = self.errors[:max_items]
        detail = "; ".join(
            f"stage {err.stage_number} {err.code}: {err.message}"
            for err in shown
        )
        extra = len(self.errors) - len(shown)
        if extra > 0:
            detail += f"; +{extra} more"
        return detail

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "stage_number": err.stage_number,
                "instruction_index": err.instruction_index,
                "code": err.code,
                "message": err.message,
                "offending_text": err.offending_text,
            }
            for err in self.errors
        ]


@dataclass
class PlanRepairRequest:
    """Payload needed to ask the planner to repair an invalid plan."""

    goal: str
    plan_version: int
    attempt_number: int
    invalid_plan_data: dict[str, Any]
    validation_errors: list[PlanValidationError]


def validate_plan(plan: ResearchPlan) -> PlanValidationResult:
    """Validate structure, dependencies, and executable instructions."""
    result = PlanValidationResult()
    known_stages = {task.stage_number for task in plan.tasks}

    for task in plan.tasks:
        dependencies = set(task.depends_on)

        if not task.agent_instructions or not any(step.strip() for step in task.agent_instructions):
            result.errors.append(
                PlanValidationError(
                    stage_number=task.stage_number,
                    code="empty_instructions",
                    message="Stage has no executable instructions",
                )
            )

        if task.stage_number in dependencies:
            result.errors.append(
                PlanValidationError(
                    stage_number=task.stage_number,
                    code="self_dependency",
                    message="Stage depends on itself",
                )
            )

        unknown_dependencies = sorted(dep for dep in dependencies if dep not in known_stages)
        if unknown_dependencies:
            result.errors.append(
                PlanValidationError(
                    stage_number=task.stage_number,
                    code="unknown_dependency",
                    message=f"Unknown dependencies: {unknown_dependencies}",
                )
            )

        for idx, instruction in enumerate(task.agent_instructions):
            stripped = instruction.strip()
            if not stripped:
                result.errors.append(
                    PlanValidationError(
                        stage_number=task.stage_number,
                        instruction_index=idx,
                        code="empty_instructions",
                        message="Instruction is empty",
                    )
                )
                continue

            if has_legacy_placeholder(stripped):
                result.errors.append(
                    PlanValidationError(
                        stage_number=task.stage_number,
                        instruction_index=idx,
                        code="legacy_placeholder",
                        message="Legacy <...> placeholder is not allowed",
                        offending_text=stripped,
                    )
                )

            if "..." in stripped:
                result.errors.append(
                    PlanValidationError(
                        stage_number=task.stage_number,
                        instruction_index=idx,
                        code="ellipsis_instruction",
                        message="Ellipsis is not allowed in executable instructions",
                        offending_text=stripped,
                    )
                )

            if _looks_like_broken_tool_call(stripped):
                result.errors.append(
                    PlanValidationError(
                        stage_number=task.stage_number,
                        instruction_index=idx,
                        code="non_executable_tool_call",
                        message="Instruction looks like an incomplete tool call",
                        offending_text=stripped,
                    )
                )

            for ref in extract_symbolic_references(stripped):
                if not is_supported_symbolic_field(ref.field):
                    result.errors.append(
                        PlanValidationError(
                            stage_number=task.stage_number,
                            instruction_index=idx,
                            code="unresolved_symbolic_reference",
                            message=f"Unsupported symbolic reference field '{ref.field}'",
                            offending_text=ref.raw,
                        )
                    )
                    continue

                if ref.stage_number not in known_stages:
                    result.errors.append(
                        PlanValidationError(
                            stage_number=task.stage_number,
                            instruction_index=idx,
                            code="unresolved_symbolic_reference",
                            message=f"Symbolic reference points to unknown stage {ref.stage_number}",
                            offending_text=ref.raw,
                        )
                    )
                    continue

                if ref.stage_number == task.stage_number:
                    result.errors.append(
                        PlanValidationError(
                            stage_number=task.stage_number,
                            instruction_index=idx,
                            code="unresolved_symbolic_reference",
                            message="Stage cannot reference its own outputs",
                            offending_text=ref.raw,
                        )
                    )
                    continue

                if ref.stage_number not in dependencies:
                    result.errors.append(
                        PlanValidationError(
                            stage_number=task.stage_number,
                            instruction_index=idx,
                            code="unresolved_symbolic_reference",
                            message=(
                                f"Symbolic reference to stage {ref.stage_number} requires "
                                f"`depends_on: [{ref.stage_number}]`"
                            ),
                            offending_text=ref.raw,
                        )
                    )

    return result


def _looks_like_broken_tool_call(text: str) -> bool:
    if "(" in text and ")" not in text:
        return True
    if text.count("(") != text.count(")"):
        return True
    if "action=" in text and "(" not in text:
        return True
    return False
