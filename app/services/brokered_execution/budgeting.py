"""
Cheap local budget normalization for brokered execution slices.
"""

from __future__ import annotations

from app.execution_models import ExecutionPlan

_SYNC_TOOLS = {"events_sync", "datasets_sync"}
_MUTATING_EXPENSIVE_TOOLS = {
    "features_custom",
    "features_dataset",
    "models_dataset",
    "models_train",
    "experiments_run",
    "research_record",
    "backtests_runs",
    "backtests_studies",
    "backtests_walkforward",
    "backtests_analysis",
    "backtests_conditions",
}
_DYNAMIC_EXPENSIVE_BUDGET_HARD_CAP = 5


def normalize_plan_budgets(plan: ExecutionPlan) -> ExecutionPlan:
    for slice_obj in plan.slices:
        floor = _expensive_budget_floor(slice_obj.allowed_tools)
        if floor > slice_obj.max_expensive_calls:
            slice_obj.max_expensive_calls = floor
    return plan


def _expensive_budget_floor(allowed_tools: list[str]) -> int:
    allowed = {str(item).strip() for item in allowed_tools if str(item).strip()}
    floor = 1
    if allowed & _SYNC_TOOLS:
        floor += 1
    if allowed & _MUTATING_EXPENSIVE_TOOLS:
        floor += 1
    if any(tool.startswith("backtests_") for tool in allowed):
        floor += 1
    return min(max(floor, 1), 3)


def maybe_extend_expensive_budget(*, allowed_tools: list[str], current_budget: int, requested_tool: str) -> int:
    requested = str(requested_tool).strip()
    if not requested or requested not in {str(item).strip() for item in allowed_tools if str(item).strip()}:
        return current_budget
    if current_budget >= _DYNAMIC_EXPENSIVE_BUDGET_HARD_CAP:
        return current_budget
    return current_budget + 1
