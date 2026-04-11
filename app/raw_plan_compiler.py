"""
Deterministic compilation from semantic raw plans into execution plan batches.
"""

from __future__ import annotations

from pathlib import Path

from app.execution_models import ExecutionPlan, PlanSlice
from app.raw_plan_models import CompileReport, CompiledPlanSequence, RawPlanDocument, SemanticRawPlan, SemanticStage
from app.services.direct_execution.budgeting import normalize_plan_budgets

_PUBLIC_TOOL_NAMES = {
    "research_project",
    "research_map",
    "research_record",
    "research_search",
    "datasets",
    "datasets_sync",
    "datasets_preview",
    "features_catalog",
    "features_dataset",
    "features_custom",
    "features_analytics",
    "features_cleanup",
    "models_dataset",
    "models_train",
    "models_registry",
    "models_compare",
    "models_to_feature",
    "backtests_strategy",
    "backtests_strategy_validate",
    "backtests_plan",
    "backtests_runs",
    "backtests_studies",
    "backtests_walkforward",
    "backtests_conditions",
    "backtests_analysis",
    "events",
    "events_sync",
    "experiments_run",
    "experiments_inspect",
    "experiments_read",
    "experiments_registry_inspect",
    "notify_send",
}

_TOOL_HINT_MAP = {
    "research_memory": ["research_project", "research_map", "research_record", "research_search"],
    "data_readiness": ["datasets", "datasets_sync", "datasets_preview", "events", "events_sync", "features_catalog", "features_dataset"],
    "feature_contract": ["features_catalog", "features_custom", "features_dataset", "features_analytics"],
    "modeling": ["models_dataset", "models_train", "models_registry", "models_compare", "models_to_feature"],
    "backtesting": ["backtests_plan", "backtests_runs", "backtests_studies", "backtests_walkforward", "backtests_conditions", "backtests_analysis"],
    "analysis": ["backtests_analysis", "backtests_conditions", "experiments_run", "experiments_inspect", "experiments_read"],
    "events": ["events", "events_sync"],
    "experiments": ["experiments_run", "experiments_inspect", "experiments_read", "experiments_registry_inspect"],
    "finalization": ["research_record", "backtests_analysis", "notify_send"],
}

_BUDGET_PRESETS = {
    "setup": (4, 4, 0),
    "data_readiness": (6, 5, 1),
    "feature_contract": (6, 5, 1),
    "modeling": (6, 5, 2),
    "backtesting": (8, 6, 2),
    "analysis": (8, 6, 2),
    "finalization": (4, 3, 0),
    "default": (6, 5, 1),
}

_TOOL_SPLIT_THRESHOLD = 5

_EXPLORATION_TOOLS: frozenset[str] = frozenset({
    "datasets", "datasets_preview", "features_catalog", "features_analytics",
    "events", "research_search", "research_project", "research_map",
    "models_registry", "models_compare", "backtests_plan", "backtests_analysis",
    "backtests_conditions", "experiments_inspect", "experiments_read",
    "experiments_registry_inspect", "gold_collection",
})

_CONSTRUCTION_TOOLS: frozenset[str] = frozenset({
    "datasets_sync", "features_dataset", "features_custom", "features_cleanup",
    "events_sync", "models_dataset", "models_train", "models_to_feature",
    "backtests_strategy", "backtests_strategy_validate", "backtests_runs",
    "backtests_studies", "backtests_walkforward", "notify_send",
    "experiments_run",
})

_BOTH_PHASE_TOOLS: frozenset[str] = frozenset({"research_record"})


def compile_semantic_raw_plan(
    document: RawPlanDocument,
    semantic_plan: SemanticRawPlan,
    *,
    semantic_method: str,
    plan_source_kind: str = "compiled_raw",
) -> CompiledPlanSequence:
    sequence_id = f"compiled_{Path(document.source_file).stem}"
    plans: list[ExecutionPlan] = []
    warnings = list(document.parser_warnings) + list(semantic_plan.warnings)
    stages = _expand_stages(list(semantic_plan.stages))
    for batch_index in range(0, len(stages), 3):
        batch = stages[batch_index: batch_index + 3]
        plan_id = f"{sequence_id}_batch_{len(plans) + 1}"
        slices: list[PlanSlice] = []
        for index, stage in enumerate(batch):
            slice_obj, slice_warnings = _compile_stage(sequence_id=sequence_id, stage=stage, slot_index=index + 1)
            slices.append(slice_obj)
            warnings.extend(slice_warnings)
        plans.append(
            normalize_plan_budgets(
                ExecutionPlan(
                plan_id=plan_id,
                goal=semantic_plan.goal,
                baseline_ref=semantic_plan.baseline_ref,
                global_constraints=list(semantic_plan.global_constraints),
                slices=slices,
                plan_source_kind=plan_source_kind,
                source_sequence_id=sequence_id,
                source_raw_plan=document.source_file,
                sequence_batch_index=len(plans) + 1,
                )
            )
        )
    report = CompileReport(
        source_file=document.source_file,
        source_hash=document.source_hash,
        sequence_id=sequence_id,
        compile_status="compiled",
        parser_confidence=document.parse_confidence,
        semantic_method=semantic_method,
        stage_count=len(stages),
        compiled_plan_count=len(plans),
        warnings=warnings,
    )
    return CompiledPlanSequence(
        source_file=document.source_file,
        source_hash=document.source_hash,
        sequence_id=sequence_id,
        semantic_plan=semantic_plan,
        plans=plans,
        report=report,
    )


def build_failed_sequence(
    document: RawPlanDocument,
    *,
    semantic_method: str,
    errors: list[str],
) -> CompiledPlanSequence:
    sequence_id = f"compiled_{Path(document.source_file).stem}"
    report = CompileReport(
        source_file=document.source_file,
        source_hash=document.source_hash,
        sequence_id=sequence_id,
        compile_status="failed",
        parser_confidence=document.parse_confidence,
        semantic_method=semantic_method,
        stage_count=len(document.candidate_stages),
        compiled_plan_count=0,
        warnings=list(document.parser_warnings),
        errors=list(errors),
    )
    return CompiledPlanSequence(
        source_file=document.source_file,
        source_hash=document.source_hash,
        sequence_id=sequence_id,
        semantic_plan=None,
        plans=[],
        report=report,
    )


def _compile_stage(*, sequence_id: str, stage: SemanticStage, slot_index: int) -> tuple[PlanSlice, list[str]]:
    budget_class = _stage_budget_class(stage)
    max_turns, max_tool_calls, max_expensive_calls = _BUDGET_PRESETS[budget_class]
    allowed_tools = _infer_allowed_tools(stage)
    allowed_tools, warnings = _reconcile_stage_allowed_tools(stage=stage, allowed_tools=allowed_tools)
    policy_tags = list(stage.policy_tags)
    if not stage.required and "optional_candidate" not in policy_tags:
        policy_tags.append("optional_candidate")
    parallel_slot = slot_index if stage.parallelizable and slot_index <= 3 else 1
    return PlanSlice(
        slice_id=f"{sequence_id}_{stage.stage_id}",
        title=stage.title,
        hypothesis=stage.objective,
        objective=stage.objective,
        success_criteria=list(stage.success_criteria),
        allowed_tools=allowed_tools,
        evidence_requirements=list(stage.success_criteria),
        policy_tags=policy_tags or [budget_class],
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        max_expensive_calls=max_expensive_calls,
        parallel_slot=parallel_slot,
        depends_on=[f"{sequence_id}_{dep}" for dep in stage.depends_on if dep],
    ), warnings


def _infer_allowed_tools(stage: SemanticStage) -> list[str]:
    allowed: list[str] = []
    for hint in stage.tool_hints:
        if hint in _PUBLIC_TOOL_NAMES:
            if hint not in allowed:
                allowed.append(hint)
            continue
        for tool_name in _TOOL_HINT_MAP.get(hint, []):
            if tool_name not in allowed:
                allowed.append(tool_name)
    if not allowed:
        allowed = list(_TOOL_HINT_MAP["analysis"])
    return allowed


def _stage_budget_class(stage: SemanticStage) -> str:
    title = f"{stage.title} {stage.objective} {' '.join(stage.policy_tags)} {' '.join(stage.tool_hints)}".lower()
    if any(token in title for token in ("setup", "freeze", "project", "map", "record", "terminology")):
        return "setup"
    if any(token in title for token in ("dataset", "data", "readiness", "contract", "feature", "publish", "validate")):
        return "feature_contract" if "feature" in title or "contract" in title else "data_readiness"
    if any(token in title for token in ("model", "label", "train", "classification")):
        return "modeling"
    if any(token in title for token in ("backtest", "integration", "matched oos", "walk-forward", "walkforward")):
        return "backtesting"
    if any(token in title for token in ("analysis", "ownership", "cannibalization", "verdict", "final")):
        return "finalization" if any(token in title for token in ("verdict", "final")) else "analysis"
    return "default"


def _reconcile_stage_allowed_tools(*, stage: SemanticStage, allowed_tools: list[str]) -> tuple[list[str], list[str]]:
    normalized = list(allowed_tools)
    warnings: list[str] = []
    if "research_record" in normalized:
        return normalized, warnings
    if not normalized:
        return normalized, warnings
    normalized.append("research_record")
    warnings.append(
        f"Auto-added research_record for {stage.stage_id}: every non-trivial stage needs documentation capability."
    )
    return normalized, warnings


def _classify_split_tools(allowed_tools: list[str]) -> tuple[list[str], list[str]] | None:
    if len(allowed_tools) <= _TOOL_SPLIT_THRESHOLD:
        return None
    exploration = [t for t in allowed_tools if t in _EXPLORATION_TOOLS]
    construction = [t for t in allowed_tools if t in _CONSTRUCTION_TOOLS]
    both = [t for t in allowed_tools if t in _BOTH_PHASE_TOOLS]
    if not exploration or not construction:
        return None
    exploration = both + exploration
    construction = both + construction
    return (exploration, construction)


def _maybe_split_stage(stage: SemanticStage) -> list[SemanticStage]:
    allowed_tools = _infer_allowed_tools(stage)
    split = _classify_split_tools(allowed_tools)
    if split is None:
        return [stage]
    exploration_tools, construction_tools = split
    part1 = SemanticStage(
        stage_id=f"{stage.stage_id}_part1",
        title=f"{stage.title} (exploration)",
        objective=stage.objective,
        actions=list(stage.actions),
        success_criteria=list(stage.success_criteria) + ["Exploration phase complete"],
        tool_hints=exploration_tools,
        policy_tags=list(stage.policy_tags),
        depends_on=list(stage.depends_on),
        required=stage.required,
        parallelizable=False,
        gate_hint=stage.gate_hint,
        raw_stage_ref=stage.raw_stage_ref,
    )
    part2 = SemanticStage(
        stage_id=f"{stage.stage_id}_part2",
        title=f"{stage.title} (construction)",
        objective=stage.objective,
        actions=list(stage.actions),
        success_criteria=list(stage.success_criteria),
        tool_hints=construction_tools,
        policy_tags=list(stage.policy_tags),
        depends_on=[f"{stage.stage_id}_part1"],
        required=stage.required,
        parallelizable=False,
        gate_hint="",
        raw_stage_ref=stage.raw_stage_ref,
    )
    return [part1, part2]


def _expand_stages(stages: list[SemanticStage]) -> list[SemanticStage]:
    expanded: list[SemanticStage] = []
    split_ids: set[str] = set()
    for stage in stages:
        parts = _maybe_split_stage(stage)
        if len(parts) > 1:
            split_ids.add(stage.stage_id)
        expanded.extend(parts)
    if not split_ids:
        return expanded
    for stage in expanded:
        if not stage.depends_on:
            continue
        stage.depends_on = [
            f"{dep}_part2" if dep in split_ids else dep
            for dep in stage.depends_on
        ]
    return expanded


def _stage_requires_research_record(stage: SemanticStage) -> bool:
    haystack = " ".join(
        part.strip().lower()
        for part in (
            stage.title,
            stage.objective,
            *list(stage.actions or []),
            *list(stage.success_criteria or []),
        )
        if str(part).strip()
    )
    markers = (
        "record",
        "document",
        "postmortem",
        "rules",
        "methodology",
        "journal",
        "milestone",
        "freeze rationale",
    )
    return any(marker in haystack for marker in markers)
