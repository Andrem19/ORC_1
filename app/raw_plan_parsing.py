"""
Strict parsing and validation for semantic raw-plan extraction output.
"""

from __future__ import annotations

from typing import Any

from app.execution_models import BaselineRef
from app.execution_parsing import StructuredOutputError, extract_json_object
from app.raw_plan_models import RawPlanDocument, SemanticRawPlan, SemanticStage

_SEMANTIC_TOOL_FAMILY_HINTS = {
    "research_memory",
    "data_readiness",
    "feature_contract",
    "modeling",
    "backtesting",
    "analysis",
    "events",
    "experiments",
    "finalization",
}


def parse_semantic_raw_plan_output(text: str, *, document: RawPlanDocument) -> SemanticRawPlan:
    payload = extract_json_object(text)
    required = {"source_title", "goal", "baseline_ref", "global_constraints", "stages"}
    missing = sorted(required - set(payload))
    if missing:
        raise StructuredOutputError(f"semantic_raw_plan_missing_fields:{','.join(missing)}")
    baseline = _resolve_baseline(payload.get("baseline_ref"), document=document)
    stages_raw = payload.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise StructuredOutputError("semantic_raw_plan_requires_non_empty_stages")
    stages = [parse_semantic_stage(item, known_stage_ids={f.stage_id for f in document.candidate_stages}) for item in stages_raw]
    stage_ids = [stage.stage_id for stage in stages]
    seen: set[str] = set()
    for stage in stages:
        if stage.stage_id in seen:
            raise StructuredOutputError(f"semantic_stage_duplicate_id:{stage.stage_id}")
        seen.add(stage.stage_id)
        for dep in stage.depends_on:
            if dep not in stage_ids:
                raise StructuredOutputError(f"semantic_stage_unknown_dependency:{dep}")
            if stage_ids.index(dep) >= stage_ids.index(stage.stage_id):
                raise StructuredOutputError(f"semantic_stage_dependency_must_point_backward:{stage.stage_id}:{dep}")
    plan = SemanticRawPlan(
        source_file=document.source_file,
        source_hash=document.source_hash,
        source_title=str(payload.get("source_title", document.title) or document.title).strip(),
        goal=str(payload.get("goal", "") or "").strip(),
        baseline_ref=baseline,
        global_constraints=_string_list(payload.get("global_constraints", []), field_name="global_constraints"),
        stages=stages,
        warnings=_string_list(payload.get("warnings", []), field_name="warnings"),
        parse_confidence=document.parse_confidence,
    )
    if not plan.goal:
        raise StructuredOutputError("semantic_raw_plan_goal_must_be_non_empty")
    return plan


def parse_semantic_stage(payload: Any, *, known_stage_ids: set[str]) -> SemanticStage:
    if not isinstance(payload, dict):
        raise StructuredOutputError("semantic_stage_must_be_object")
    required = {"stage_id", "title", "objective", "actions", "success_criteria"}
    missing = sorted(required - set(payload))
    if missing:
        raise StructuredOutputError(f"semantic_stage_missing_fields:{','.join(missing)}")
    tool_hints = _string_list(payload.get("tool_hints", []), field_name="tool_hints")
    if not tool_hints:
        tool_hints = ["analysis"]
    stage = SemanticStage(
        stage_id=str(payload.get("stage_id", "") or "").strip(),
        title=str(payload.get("title", "") or "").strip(),
        objective=str(payload.get("objective", "") or "").strip(),
        actions=_string_list(payload.get("actions", []), field_name="actions"),
        success_criteria=_string_list(payload.get("success_criteria", []), field_name="success_criteria"),
        tool_hints=[_normalize_tool_hint(item) for item in tool_hints],
        policy_tags=_string_list(payload.get("policy_tags", []), field_name="policy_tags"),
        depends_on=_string_list(payload.get("depends_on", []), field_name="depends_on"),
        required=bool(payload.get("required", True)),
        parallelizable=bool(payload.get("parallelizable", False)),
        gate_hint=str(payload.get("gate_hint", "") or "").strip(),
        raw_stage_ref=_resolve_raw_stage_ref(str(payload.get("raw_stage_ref", "") or "").strip(), known_stage_ids=known_stage_ids),
    )
    if not stage.stage_id or not stage.title or not stage.objective:
        raise StructuredOutputError("semantic_stage_requires_non_empty_id_title_objective")
    if not stage.actions:
        raise StructuredOutputError(f"semantic_stage_requires_actions:{stage.stage_id}")
    if not stage.success_criteria:
        raise StructuredOutputError(f"semantic_stage_requires_success_criteria:{stage.stage_id}")
    return stage


def _resolve_baseline(value: Any, *, document: RawPlanDocument) -> BaselineRef:
    payload = dict(value) if isinstance(value, dict) else {}
    if not payload:
        payload = dict(document.baseline_ref_hint)
    payload.setdefault("snapshot_id", document.baseline_ref_hint.get("snapshot_id", "active-signal-v1"))
    payload.setdefault("version", document.baseline_ref_hint.get("version", 1))
    payload.setdefault("symbol", document.baseline_ref_hint.get("symbol", "BTCUSDT"))
    payload.setdefault("anchor_timeframe", document.baseline_ref_hint.get("anchor_timeframe", "1h"))
    payload.setdefault("execution_timeframe", document.baseline_ref_hint.get("execution_timeframe", "5m"))
    return BaselineRef(
        snapshot_id=str(payload.get("snapshot_id", "") or "").strip(),
        version=payload.get("version", 1),
        symbol=str(payload.get("symbol", "BTCUSDT") or "BTCUSDT").strip(),
        anchor_timeframe=str(payload.get("anchor_timeframe", "1h") or "1h").strip(),
        execution_timeframe=str(payload.get("execution_timeframe", "5m") or "5m").strip(),
    )


def _normalize_tool_hint(value: str) -> str:
    item = str(value or "").strip()
    if not item:
        return "analysis"
    if item in _SEMANTIC_TOOL_FAMILY_HINTS:
        return item
    return item


def _resolve_raw_stage_ref(value: str, *, known_stage_ids: set[str]) -> str:
    if value in known_stage_ids:
        return value
    return value or ""


def _string_list(value: Any, *, field_name: str) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise StructuredOutputError(f"{field_name}_must_be_list")
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result
