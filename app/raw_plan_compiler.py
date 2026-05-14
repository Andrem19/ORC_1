"""
Deterministic compilation from semantic raw plans into execution plan batches.
"""

from __future__ import annotations

from pathlib import Path

from app.execution_models import ExecutionPlan, PlanSlice
from app.raw_plan_models import CompileReport, CompiledPlanSequence, RawPlanDocument, SemanticRawPlan, SemanticStage
from app.services.direct_execution.runtime_profiles import (
    derive_slice_acceptance_policy,
    derive_runtime_slice_metadata,
    resolve_runtime_slice_metadata_with_prerequisites,
)
from app.services.direct_execution.acceptance.builder import build_acceptance_contract
from app.services.mcp_catalog.classifier import build_family_tool_map
from app.services.mcp_catalog.models import McpCatalogSnapshot
from app.services.direct_execution.budgeting import normalize_plan_budgets

_BUDGET_PRESETS = {
    "setup": (4, 4, 0),
    "data_readiness": (6, 5, 1),
    "feature_contract": (6, 5, 2),
    "modeling": (6, 5, 2),
    "backtesting": (8, 6, 2),
    "analysis": (8, 6, 2),
    "finalization": (4, 3, 0),
    "default": (6, 5, 1),
}

_TOOL_SPLIT_THRESHOLD = 5
_FAMILY_ALIASES = {
    "analysis": "analysis",
    "backtesting": "backtesting",
    "data": "data_readiness",
    "data_readiness": "data_readiness",
    "events": "events",
    "experiments": "experiments",
    "feature": "feature_contract",
    "feature_contract": "feature_contract",
    "finalization": "finalization",
    "modeling": "modeling",
    "models": "modeling",
    "research": "research_memory",
    "research_memory": "research_memory",
}


def compile_semantic_raw_plan(
    document: RawPlanDocument,
    semantic_plan: SemanticRawPlan,
    *,
    semantic_method: str,
    plan_source_kind: str = "compiled_raw",
    catalog_snapshot: McpCatalogSnapshot,
) -> CompiledPlanSequence:
    sequence_id = f"compiled_{Path(document.source_file).stem}"
    plans: list[ExecutionPlan] = []
    warnings = list(document.parser_warnings) + list(semantic_plan.warnings)
    family_map = build_family_tool_map(catalog_snapshot)
    tool_name_set = catalog_snapshot.tool_name_set()
    stages = _expand_stages(
        list(semantic_plan.stages),
        catalog_snapshot=catalog_snapshot,
        tool_name_set=tool_name_set,
        family_map=family_map,
    )
    # Pre-compute output facts for each stage so downstream stages can
    # automatically inherit prerequisites from their depends_on ancestors.
    stage_output_facts: dict[str, list[str]] = {}
    for s in stages:
        s_allowed = _infer_allowed_tools(s, tool_name_set=tool_name_set, family_map=family_map)
        _, s_output, _ = derive_runtime_slice_metadata(
            allowed_tools=s_allowed,
            catalog_snapshot=catalog_snapshot,
            title=s.title,
            objective=s.objective,
            success_criteria=list(s.success_criteria),
            policy_tags=list(s.policy_tags),
        )
        stage_output_facts[s.stage_id] = list(s_output)

    errors: list[str] = []
    for batch_index in range(0, len(stages), 3):
        batch = stages[batch_index: batch_index + 3]
        plan_id = f"{sequence_id}_batch_{len(plans) + 1}"
        slices: list[PlanSlice] = []
        for index, stage in enumerate(batch):
            slice_obj, slice_warnings = _compile_stage(
                sequence_id=sequence_id,
                stage=stage,
                slot_index=index + 1,
                catalog_snapshot=catalog_snapshot,
                tool_name_set=tool_name_set,
                family_map=family_map,
                upstream_stage_facts=stage_output_facts,
            )
            slices.append(slice_obj)
            warnings.extend(slice_warnings)
        plan = normalize_plan_budgets(
            ExecutionPlan(
                plan_id=plan_id,
                goal=semantic_plan.goal,
                baseline_ref=semantic_plan.baseline_ref,
                global_constraints=list(semantic_plan.global_constraints),
                slices=slices,
                mcp_catalog_hash=catalog_snapshot.schema_hash,
                plan_source_kind=plan_source_kind,
                source_sequence_id=sequence_id,
                source_raw_plan=document.source_file,
                sequence_batch_index=len(plans) + 1,
            )
        )
        plan_errors = _validate_slice_fact_invariants(plan)
        errors.extend(plan_errors)
        plans.append(plan)
    compile_status = "failed" if errors else "compiled"
    report = CompileReport(
        source_file=document.source_file,
        source_hash=document.source_hash,
        sequence_id=sequence_id,
        mcp_catalog_hash=catalog_snapshot.schema_hash,
        compile_status=compile_status,
        parser_confidence=document.parse_confidence,
        semantic_method=semantic_method,
        stage_count=len(stages),
        compiled_plan_count=len(plans),
        warnings=warnings,
        errors=errors,
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
    mcp_catalog_hash: str,
) -> CompiledPlanSequence:
    sequence_id = f"compiled_{Path(document.source_file).stem}"
    report = CompileReport(
        source_file=document.source_file,
        source_hash=document.source_hash,
        sequence_id=sequence_id,
        mcp_catalog_hash=mcp_catalog_hash,
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


def _compile_stage(
    *,
    sequence_id: str,
    stage: SemanticStage,
    slot_index: int,
    catalog_snapshot: McpCatalogSnapshot,
    tool_name_set: set[str],
    family_map: dict[str, list[str]],
    upstream_stage_facts: dict[str, list[str]] | None = None,
) -> tuple[PlanSlice, list[str]]:
    budget_class = _stage_budget_class(stage)
    max_turns, max_tool_calls, max_expensive_calls = _BUDGET_PRESETS[budget_class]
    allowed_tools = _infer_allowed_tools(stage, tool_name_set=tool_name_set, family_map=family_map)
    allowed_tools, warnings = _reconcile_stage_allowed_tools(
        stage=stage,
        allowed_tools=allowed_tools,
        tool_name_set=tool_name_set,
        catalog_snapshot=catalog_snapshot,
    )
    runtime_profile, required_output_facts, finalization_mode = derive_runtime_slice_metadata(
        allowed_tools=allowed_tools,
        catalog_snapshot=catalog_snapshot,
        title=stage.title,
        objective=stage.objective,
        success_criteria=list(stage.success_criteria),
        policy_tags=list(stage.policy_tags),
    )
    _, required_output_facts, required_prerequisite_facts, finalization_mode = resolve_runtime_slice_metadata_with_prerequisites(
        runtime_profile=runtime_profile,
        required_output_facts=required_output_facts,
        required_prerequisite_facts=None,
        finalization_mode=finalization_mode,
        allowed_tools=allowed_tools,
        catalog_snapshot=catalog_snapshot,
        title=stage.title,
        objective=stage.objective,
        success_criteria=list(stage.success_criteria),
        policy_tags=list(stage.policy_tags),
    )
    # Derive prerequisites from upstream stage output facts when not already set.
    if not required_prerequisite_facts and stage.depends_on and upstream_stage_facts:
        computed: list[str] = []
        for dep_id in stage.depends_on:
            computed.extend(upstream_stage_facts.get(dep_id, []))
        required_prerequisite_facts = list(dict.fromkeys(computed)) or None
    policy_tags = list(stage.policy_tags)
    if not stage.required and "optional_candidate" not in policy_tags:
        policy_tags.append("optional_candidate")
    parallel_slot = slot_index if stage.parallelizable and slot_index <= 3 else 1
    acceptance_policy = derive_slice_acceptance_policy(
        runtime_profile=runtime_profile,
        allowed_tools=allowed_tools,
        policy_tags=policy_tags,
    )
    slice_obj = PlanSlice(
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
        runtime_profile=runtime_profile,
        required_prerequisite_facts=required_prerequisite_facts,
        required_output_facts=required_output_facts,
        finalization_mode=finalization_mode,
        dependency_unblock_mode=str(acceptance_policy["dependency_unblock_mode"]),
        watchlist_allows_unblock=bool(acceptance_policy["watchlist_allows_unblock"]),
        requires_mutating_evidence=bool(acceptance_policy["requires_mutating_evidence"]),
        requires_persisted_artifact=bool(acceptance_policy["requires_persisted_artifact"]),
        requires_live_handle_validation=bool(acceptance_policy["requires_live_handle_validation"]),
        depends_on=[f"{sequence_id}_{dep}" for dep in stage.depends_on if dep],
        gate_hint=stage.gate_hint or "",
    )
    slice_obj.acceptance_contract = build_acceptance_contract(slice_obj).to_dict()
    return slice_obj, warnings


def _infer_allowed_tools(
    stage: SemanticStage,
    *,
    tool_name_set: set[str],
    family_map: dict[str, list[str]],
) -> list[str]:
    allowed: list[str] = []
    for hint in stage.tool_hints:
        normalized_hint = str(hint or "").strip()
        if not normalized_hint:
            continue
        if normalized_hint in tool_name_set:
            if normalized_hint not in allowed:
                allowed.append(normalized_hint)
            continue
        family = _FAMILY_ALIASES.get(normalized_hint.lower(), normalized_hint.lower())
        family_names = list(family_map.get(family, []))
        if family == "backtesting":
            family_names.extend(tool_name for tool_name in family_map.get("analysis", []) if tool_name not in family_names)
        for tool_name in family_names:
            if tool_name not in allowed:
                allowed.append(tool_name)
    if not allowed:
        fallback_family = family_map.get("analysis") or family_map.get("backtesting") or []
        allowed = list(fallback_family[:8])
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


def _reconcile_stage_allowed_tools(
    *,
    stage: SemanticStage,
    allowed_tools: list[str],
    tool_name_set: set[str],
    catalog_snapshot: McpCatalogSnapshot,
) -> tuple[list[str], list[str]]:
    normalized = list(allowed_tools)
    warnings: list[str] = []
    doc_tool_name = _documentation_tool_name(catalog_snapshot)
    if not doc_tool_name:
        return normalized, warnings
    if doc_tool_name in normalized:
        return normalized, warnings
    if not normalized:
        return normalized, warnings
    if doc_tool_name in tool_name_set:
        normalized.append(doc_tool_name)
        warnings.append(
            f"Auto-added {doc_tool_name} for {stage.stage_id}: every non-trivial stage needs documentation capability."
        )
    else:
        warnings.append(
            f"No live documentation tool available in MCP catalog for {stage.stage_id}; skipping documentation auto-injection."
        )
    return normalized, warnings


def _classify_split_tools(
    allowed_tools: list[str],
    *,
    catalog_snapshot: McpCatalogSnapshot,
) -> tuple[list[str], list[str]] | None:
    if len(allowed_tools) <= _TOOL_SPLIT_THRESHOLD:
        return None
    exploration: list[str] = []
    construction: list[str] = []
    both: list[str] = []
    for tool_name in allowed_tools:
        tool = catalog_snapshot.get_tool(tool_name)
        if tool is not None and tool.supports_terminal_write and tool.family == "research_memory":
            both.append(tool_name)
            continue
        if tool is None:
            exploration.append(tool_name)
            continue
        if tool.side_effects == "mutating":
            construction.append(tool_name)
        else:
            exploration.append(tool_name)
    if not exploration or not construction:
        return None
    exploration = both + exploration
    construction = both + construction
    return (exploration, construction)


def _maybe_split_stage(
    stage: SemanticStage,
    *,
    catalog_snapshot: McpCatalogSnapshot,
    tool_name_set: set[str],
    family_map: dict[str, list[str]],
) -> list[SemanticStage]:
    allowed_tools = _infer_allowed_tools(stage, tool_name_set=tool_name_set, family_map=family_map)
    split = _classify_split_tools(allowed_tools, catalog_snapshot=catalog_snapshot)
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
        gate_hint=stage.gate_hint,  # propagate so optional-gate check works for construction too
        raw_stage_ref=stage.raw_stage_ref,
    )
    return [part1, part2]


def _expand_stages(
    stages: list[SemanticStage],
    *,
    catalog_snapshot: McpCatalogSnapshot,
    tool_name_set: set[str],
    family_map: dict[str, list[str]],
) -> list[SemanticStage]:
    expanded: list[SemanticStage] = []
    split_ids: set[str] = set()
    for stage in stages:
        parts = _maybe_split_stage(
            stage,
            catalog_snapshot=catalog_snapshot,
            tool_name_set=tool_name_set,
            family_map=family_map,
        )
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


def _validate_slice_fact_invariants(plan: ExecutionPlan) -> list[str]:
    errors: list[str] = []
    by_id = {item.slice_id: item for item in plan.slices}
    for slice_obj in plan.slices:
        prerequisites = [
            str(item).strip()
            for item in (slice_obj.required_prerequisite_facts or [])
            if str(item).strip()
        ]
        if not prerequisites:
            continue
        ancestors: set[str] = set()
        stack: list[str] = list(slice_obj.depends_on or [])
        while stack:
            dep_id = stack.pop()
            if dep_id in ancestors or dep_id not in by_id:
                continue
            ancestors.add(dep_id)
            stack.extend(by_id[dep_id].depends_on or [])
        produced: set[str] = set()
        for anc_id in ancestors:
            anc = by_id.get(anc_id)
            if anc is None:
                continue
            for fact in anc.required_output_facts or []:
                normalized = str(fact or "").strip()
                if normalized:
                    produced.add(normalized)
        missing = [fact for fact in prerequisites if fact not in produced]
        if missing:
            errors.append(
                f"Slice '{slice_obj.slice_id}' requires prerequisite facts "
                f"{missing} but no upstream slice (via depends_on) declares them "
                f"in required_output_facts."
            )
    return errors


def _documentation_tool_name(catalog_snapshot: McpCatalogSnapshot) -> str:
    for tool in catalog_snapshot.tools:
        if (
            tool.family == "research_memory"
            and tool.side_effects == "mutating"
            and ("record" in tool.fields or "kind" in tool.fields or "operation_id" in tool.fields)
        ):
            return tool.name
    return ""
