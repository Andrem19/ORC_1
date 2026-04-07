"""
Validation for structured research plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.plan_models import PlanStep, ResearchPlan
from app.planner_contract import inspect_legacy_instruction, validate_tool_step
from app.plan_symbolic_refs import has_legacy_placeholder


# ---------------------------------------------------------------------------
# Severity model: hard errors reject the plan, soft errors are warnings
# ---------------------------------------------------------------------------

HARD_ERROR_CODES = frozenset({
    "empty_instructions",
    "self_dependency",
    "unknown_dependency",
    "legacy_placeholder",
    "ellipsis_instruction",
})


@dataclass
class PlanValidationError:
    """One plan-validation failure."""

    stage_number: int
    code: str
    message: str
    offending_text: str = ""
    instruction_index: int | None = None
    severity: str = ""  # set by validate_plan based on HARD_ERROR_CODES

    def __post_init__(self) -> None:
        if not self.severity:
            self.severity = "hard" if self.code in HARD_ERROR_CODES else "soft"


@dataclass
class PlanValidationResult:
    """Structured validation result for a research plan."""

    errors: list[PlanValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def has_hard_errors(self) -> bool:
        return any(err.severity == "hard" for err in self.errors)

    @property
    def is_acceptable(self) -> bool:
        """True if no hard errors (soft errors are warnings only)."""
        return not self.has_hard_errors

    @property
    def soft_errors(self) -> list[PlanValidationError]:
        return [err for err in self.errors if err.severity == "soft"]

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
                "severity": err.severity,
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
        steps = task.normalized_steps()

        if not steps:
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

        step_ids_seen: list[str] = []
        for idx, step in enumerate(steps):
            _validate_step_contract(
                result=result,
                task=task,
                step=step,
                step_index=idx,
                step_ids_seen=step_ids_seen,
            )
            step_ids_seen.append(step.step_id)

        if task.steps:
            # Structured steps were validated above — skip raw instruction checks.
            # (normalized_steps() converts agent_instructions into PlanStep objects,
            # but when the planner already provided structured steps, the raw
            # agent_instructions list is empty so there's nothing to validate here.)
            continue

        # Legacy path: planner sent free-form agent_instructions instead of steps.
        # These were converted by normalized_steps() and validated as PlanStep objects
        # in the loop above, so the checks below are supplementary (catching patterns
        # that the step-level validation may miss for legacy-format plans).
        if not task.agent_instructions:
            continue

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

            for violation in inspect_legacy_instruction(stripped):
                result.errors.append(
                    PlanValidationError(
                        stage_number=task.stage_number,
                        instruction_index=idx,
                        code=violation.code,
                        message=violation.message,
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

    return result


def _validate_step_contract(
    *,
    result: PlanValidationResult,
    task: Any,
    step: PlanStep,
    step_index: int,
    step_ids_seen: list[str],
) -> None:
    if not step.step_id.strip():
        result.errors.append(
            PlanValidationError(
                stage_number=task.stage_number,
                instruction_index=step_index,
                code="step_ref_invalid",
                message="Step must define step_id",
            )
        )
    elif step.step_id in step_ids_seen:
        result.errors.append(
            PlanValidationError(
                stage_number=task.stage_number,
                instruction_index=step_index,
                code="step_ref_invalid",
                message=f"Duplicate step_id '{step.step_id}'",
                offending_text=step.step_id,
            )
        )

    if not step.instruction.strip():
        result.errors.append(
            PlanValidationError(
                stage_number=task.stage_number,
                instruction_index=step_index,
                code="empty_instructions",
                message="Step instruction is empty",
                offending_text=step.step_id,
            )
        )

    if step.kind == "tool_call":
        for violation in validate_tool_step(tool_name=step.tool_name, args=step.args):
            result.errors.append(
                PlanValidationError(
                    stage_number=task.stage_number,
                    instruction_index=step_index,
                    code=violation.code,
                    message=violation.message,
                    offending_text=step.tool_name or step.instruction,
                )
            )

    if step.kind == "decision" and not step.decision_outputs:
        result.errors.append(
            PlanValidationError(
                stage_number=task.stage_number,
                instruction_index=step_index,
                code="arg_missing",
                message="decision step must declare decision_outputs",
                offending_text=step.step_id,
            )
        )

    if step.kind == "record" and not step.binds:
        result.errors.append(
            PlanValidationError(
                stage_number=task.stage_number,
                instruction_index=step_index,
                code="arg_missing",
                message="record step should declare binds it records",
                offending_text=step.step_id,
            )
        )

    for text in _step_text_surfaces(step):
        if has_legacy_placeholder(text):
            result.errors.append(
                PlanValidationError(
                    stage_number=task.stage_number,
                    instruction_index=step_index,
                    code="legacy_placeholder",
                    message="Legacy <...> placeholder is not allowed",
                    offending_text=text,
                )
            )
        if "..." in text:
            result.errors.append(
                PlanValidationError(
                    stage_number=task.stage_number,
                    instruction_index=step_index,
                    code="ellipsis_instruction",
                    message="Ellipsis is not allowed in executable instructions",
                    offending_text=text,
                )
            )
        for violation in inspect_legacy_instruction(text):
            result.errors.append(
                PlanValidationError(
                    stage_number=task.stage_number,
                    instruction_index=step_index,
                    code=violation.code,
                    message=violation.message,
                    offending_text=text,
                )
            )


def _step_text_surfaces(step: PlanStep) -> list[str]:
    texts = [step.instruction, step.notes]
    for value in step.args.values():
        texts.extend(_stringify_surface_values(value))
    return [text for text in texts if text]


def _stringify_surface_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_stringify_surface_values(item))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_stringify_surface_values(item))
        return out
    return []


def _looks_like_broken_tool_call(text: str) -> bool:
    """Heuristic: flag instructions that look like malformed tool calls.

    Only triggers when the text starts with a known MCP tool name prefix,
    avoiding false positives on natural language containing parentheses.
    """
    # Quick check: must contain tool-call markers
    if "action=" not in text and "(" not in text:
        return False

    # Only check texts that start with something resembling a tool name
    text_stripped = text.lstrip()
    from app.planner_contract import TOOL_ACTIONS
    tool_prefixes = set()
    for tool_name in TOOL_ACTIONS:
        # Take first segment before any underscore
        tool_prefixes.add(tool_name.split("_")[0])
    first_word = text_stripped.split("(")[0].split("_")[0].strip().lower()
    if first_word not in tool_prefixes:
        return False

    if "(" in text and ")" not in text:
        return True
    if text.count("(") != text.count(")"):
        return True
    if "action=" in text and "(" not in text:
        return True
    return False
