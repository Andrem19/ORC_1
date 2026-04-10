"""
Parsing and validation for planner-driven semantic plan output.

Adapts the converter's parse_semantic_raw_plan_output for the planner
context where there is no RawPlanDocument from markdown.  A minimal
synthetic document is constructed so the existing validation logic
can be reused directly.
"""

from __future__ import annotations

from typing import Any

from app.execution_models import BaselineRef
from app.execution_parsing import StructuredOutputError, extract_json_object
from app.raw_plan_models import RawPlanDocument, SemanticRawPlan
from app.raw_plan_parsing import parse_semantic_stage, _string_list


def parse_planner_semantic_output(
    text: str,
    *,
    goal: str,
    baseline_bootstrap: dict[str, Any],
) -> tuple[RawPlanDocument, SemanticRawPlan]:
    """Parse LLM output into a SemanticRawPlan via the same validation as converter.

    Returns a tuple of (synthetic RawPlanDocument, SemanticRawPlan) ready
    for compile_semantic_raw_plan().
    """
    payload = extract_json_object(text)

    required_fields = {"goal", "baseline_ref", "global_constraints", "stages"}
    missing = sorted(required_fields - set(payload))
    if missing:
        raise StructuredOutputError(f"planner_semantic_missing_fields:{','.join(missing)}")

    stages_raw = payload.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise StructuredOutputError("planner_semantic_requires_non_empty_stages")

    # Build a synthetic RawPlanDocument for compiler compatibility.
    document = _make_synthetic_document(stages_raw)

    # Parse stages using the same per-stage validator as the converter.
    known_ids: set[str] = set()
    stages = []
    for item in stages_raw:
        stage = parse_semantic_stage(item, known_stage_ids=known_ids)
        stages.append(stage)
        known_ids.add(stage.stage_id)

    # Validate dependencies: unique IDs, backward-only references.
    stage_ids = [s.stage_id for s in stages]
    seen: set[str] = set()
    for stage in stages:
        if stage.stage_id in seen:
            raise StructuredOutputError(f"planner_semantic_duplicate_stage_id:{stage.stage_id}")
        seen.add(stage.stage_id)
        for dep in stage.depends_on:
            if dep not in stage_ids:
                raise StructuredOutputError(f"planner_semantic_unknown_dependency:{dep}")
            if stage_ids.index(dep) >= stage_ids.index(stage.stage_id):
                raise StructuredOutputError(
                    f"planner_semantic_dependency_must_point_backward:{stage.stage_id}:{dep}"
                )

    baseline = _resolve_baseline(payload.get("baseline_ref"), baseline_bootstrap=baseline_bootstrap)
    plan_goal = str(payload.get("goal", "") or goal).strip() or goal.strip()

    semantic_plan = SemanticRawPlan(
        source_file=document.source_file,
        source_hash=document.source_hash,
        source_title=str(payload.get("source_title", "planner") or "planner").strip(),
        goal=plan_goal,
        baseline_ref=baseline,
        global_constraints=_string_list(payload.get("global_constraints", []), field_name="global_constraints"),
        stages=stages,
        warnings=_string_list(payload.get("warnings", []), field_name="warnings"),
        parse_confidence=document.parse_confidence,
    )
    if not semantic_plan.goal:
        raise StructuredOutputError("planner_semantic_goal_must_be_non_empty")
    return document, semantic_plan


def _make_synthetic_document(stages_raw: list[dict[str, Any]]) -> RawPlanDocument:
    """Build a minimal RawPlanDocument for compiler compatibility."""
    from app.raw_plan_models import RawPlanStageFragment

    candidate_stages = []
    for idx, stage in enumerate(stages_raw):
        candidate_stages.append(
            RawPlanStageFragment(
                stage_id=str(stage.get("stage_id", f"stage_{idx + 1}") or f"stage_{idx + 1}"),
                order_index=idx,
                heading="",
                title=str(stage.get("title", "") or ""),
            )
        )
    return RawPlanDocument(
        source_file="planner",
        source_hash="",
        title="planner",
        version_label="",
        normalized_text="",
        candidate_stages=candidate_stages,
        parse_confidence=1.0,
    )


def _resolve_baseline(value: Any, *, baseline_bootstrap: dict[str, Any]) -> BaselineRef:
    payload = dict(value) if isinstance(value, dict) else {}
    defaults = {
        "snapshot_id": baseline_bootstrap.get("baseline_snapshot_id", "active-signal-v1"),
        "version": baseline_bootstrap.get("baseline_version", 1),
        "symbol": baseline_bootstrap.get("symbol", "BTCUSDT"),
        "anchor_timeframe": baseline_bootstrap.get("anchor_timeframe", "1h"),
        "execution_timeframe": baseline_bootstrap.get("execution_timeframe", "5m"),
    }
    for key, default in defaults.items():
        payload.setdefault(key, default)
    return BaselineRef(
        snapshot_id=str(payload.get("snapshot_id", "") or "").strip(),
        version=payload.get("version", 1),
        symbol=str(payload.get("symbol", "BTCUSDT") or "BTCUSDT").strip(),
        anchor_timeframe=str(payload.get("anchor_timeframe", "1h") or "1h").strip(),
        execution_timeframe=str(payload.get("execution_timeframe", "5m") or "5m").strip(),
    )
