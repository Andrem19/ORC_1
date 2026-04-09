from __future__ import annotations

import pytest

from app.execution_parsing import StructuredOutputError
from app.raw_plan_models import RawPlanDocument, RawPlanStageFragment
from app.raw_plan_parsing import parse_semantic_raw_plan_output


def _document() -> RawPlanDocument:
    return RawPlanDocument(
        source_file="raw_plans/plan_v1.md",
        source_hash="abc123",
        title="Plan v1",
        version_label="v1",
        normalized_text="# Plan v1",
        baseline_ref_hint={"snapshot_id": "active-signal-v1", "version": 1},
        candidate_stages=[
            RawPlanStageFragment(stage_id="stage_1", order_index=1, heading="ЭТАП 1", title="Stage 1"),
            RawPlanStageFragment(stage_id="stage_2", order_index=2, heading="ЭТАП 2", title="Stage 2"),
        ],
        parse_confidence=0.8,
    )


def test_parse_semantic_raw_plan_output_accepts_valid_json() -> None:
    text = """
    {
      "source_title": "Plan v1",
      "goal": "Validate funding route",
      "baseline_ref": {"snapshot_id": "active-signal-v1", "version": 1},
      "global_constraints": ["keep baseline fixed"],
      "warnings": ["llm_warning"],
      "stages": [
        {
          "stage_id": "stage_1",
          "title": "Check data",
          "objective": "Verify data exists",
          "actions": ["Inspect funding catalog"],
          "success_criteria": ["Catalog rows observed"],
          "tool_hints": ["events"],
          "depends_on": []
        },
        {
          "stage_id": "stage_2",
          "title": "Summarize",
          "objective": "Summarize findings",
          "actions": ["Write verdict"],
          "success_criteria": ["Verdict emitted"],
          "tool_hints": ["analysis"],
          "depends_on": ["stage_1"]
        }
      ]
    }
    """

    parsed = parse_semantic_raw_plan_output(text, document=_document())

    assert parsed.goal == "Validate funding route"
    assert parsed.stages[0].tool_hints == ["events"]
    assert parsed.stages[1].depends_on == ["stage_1"]
    assert parsed.warnings == ["llm_warning"]


def test_parse_semantic_raw_plan_output_rejects_forward_dependency() -> None:
    text = """
    {
      "source_title": "Plan v1",
      "goal": "Invalid deps",
      "baseline_ref": {"snapshot_id": "active-signal-v1", "version": 1},
      "global_constraints": [],
      "stages": [
        {
          "stage_id": "stage_1",
          "title": "Stage 1",
          "objective": "One",
          "actions": ["Do one"],
          "success_criteria": ["One done"],
          "depends_on": ["stage_2"]
        },
        {
          "stage_id": "stage_2",
          "title": "Stage 2",
          "objective": "Two",
          "actions": ["Do two"],
          "success_criteria": ["Two done"]
        }
      ]
    }
    """

    with pytest.raises(StructuredOutputError, match="semantic_stage_dependency_must_point_backward"):
        parse_semantic_raw_plan_output(text, document=_document())
