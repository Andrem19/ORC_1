"""
Unit tests for app.planner_semantic_parsing.
"""

import json
import pytest

from app.execution_parsing import StructuredOutputError
from app.planner_semantic_parsing import parse_planner_semantic_output


def _valid_payload(**overrides) -> dict:
    base = {
        "source_title": "Test Plan",
        "goal": "Test goal",
        "baseline_ref": {
            "snapshot_id": "snap-1",
            "version": 2,
            "symbol": "BTCUSDT",
            "anchor_timeframe": "1h",
            "execution_timeframe": "5m",
        },
        "global_constraints": ["keep baseline fixed"],
        "stages": [
            {
                "stage_id": "stage_1",
                "title": "Setup",
                "objective": "Initialize project",
                "actions": ["Create research project"],
                "success_criteria": ["Project created"],
                "tool_hints": ["research_memory"],
                "policy_tags": ["setup"],
                "depends_on": [],
                "required": True,
                "parallelizable": False,
            }
        ],
    }
    base.update(overrides)
    return base


def _baseline_bootstrap() -> dict:
    return {
        "baseline_snapshot_id": "snap-1",
        "baseline_version": 2,
        "symbol": "BTCUSDT",
        "anchor_timeframe": "1h",
        "execution_timeframe": "5m",
    }


class TestParsePlannerSemanticOutputValid:
    def test_valid_json_produces_semantic_plan(self):
        payload = _valid_payload()
        text = json.dumps(payload)
        doc, plan = parse_planner_semantic_output(
            text, goal="Test goal", baseline_bootstrap=_baseline_bootstrap()
        )
        assert plan.goal == "Test goal"
        assert len(plan.stages) == 1
        assert plan.stages[0].stage_id == "stage_1"
        assert plan.stages[0].tool_hints == ["research_memory"]

    def test_synthetic_document_has_planner_source(self):
        text = json.dumps(_valid_payload())
        doc, plan = parse_planner_semantic_output(
            text, goal="G", baseline_bootstrap=_baseline_bootstrap()
        )
        assert doc.source_file == "planner"
        assert doc.source_hash == ""
        assert doc.parse_confidence == 1.0

    def test_multiple_stages(self):
        payload = _valid_payload()
        payload["stages"].append(
            {
                "stage_id": "stage_2",
                "title": "Analyze",
                "objective": "Run analysis",
                "actions": ["Run backtest analysis"],
                "success_criteria": ["Report generated"],
                "tool_hints": ["analysis"],
                "depends_on": ["stage_1"],
            }
        )
        text = json.dumps(payload)
        doc, plan = parse_planner_semantic_output(
            text, goal="G", baseline_bootstrap=_baseline_bootstrap()
        )
        assert len(plan.stages) == 2
        assert plan.stages[1].depends_on == ["stage_1"]

    def test_goal_from_caller_overrides_empty_llm_goal(self):
        payload = _valid_payload(goal="")
        text = json.dumps(payload)
        doc, plan = parse_planner_semantic_output(
            text, goal="Caller goal", baseline_bootstrap=_baseline_bootstrap()
        )
        assert plan.goal == "Caller goal"

    def test_baseline_from_bootstrap_when_llm_empty(self):
        payload = _valid_payload(baseline_ref={})
        text = json.dumps(payload)
        doc, plan = parse_planner_semantic_output(
            text, goal="G", baseline_bootstrap=_baseline_bootstrap()
        )
        assert plan.baseline_ref.snapshot_id == "snap-1"
        assert plan.baseline_ref.version == 2

    def test_tool_hints_default_to_analysis(self):
        payload = _valid_payload()
        payload["stages"][0]["tool_hints"] = []
        text = json.dumps(payload)
        doc, plan = parse_planner_semantic_output(
            text, goal="G", baseline_bootstrap=_baseline_bootstrap()
        )
        assert plan.stages[0].tool_hints == ["analysis"]


class TestParsePlannerSemanticOutputErrors:
    def test_missing_required_field(self):
        payload = _valid_payload()
        del payload["goal"]
        text = json.dumps(payload)
        with pytest.raises(StructuredOutputError, match="planner_semantic_missing_fields"):
            parse_planner_semantic_output(
                text, goal="G", baseline_bootstrap=_baseline_bootstrap()
            )

    def test_empty_stages(self):
        payload = _valid_payload(stages=[])
        text = json.dumps(payload)
        with pytest.raises(StructuredOutputError, match="planner_semantic_requires_non_empty_stages"):
            parse_planner_semantic_output(
                text, goal="G", baseline_bootstrap=_baseline_bootstrap()
            )

    def test_duplicate_stage_ids(self):
        payload = _valid_payload()
        payload["stages"].append(dict(payload["stages"][0]))
        text = json.dumps(payload)
        with pytest.raises(StructuredOutputError, match="planner_semantic_duplicate_stage_id"):
            parse_planner_semantic_output(
                text, goal="G", baseline_bootstrap=_baseline_bootstrap()
            )

    def test_forward_dependency(self):
        payload = _valid_payload()
        payload["stages"][0]["depends_on"] = ["stage_2"]
        payload["stages"].append(
            {
                "stage_id": "stage_2",
                "title": "Later",
                "objective": "Do later",
                "actions": ["Act"],
                "success_criteria": ["Done"],
                "tool_hints": ["analysis"],
            }
        )
        text = json.dumps(payload)
        with pytest.raises(StructuredOutputError, match="planner_semantic_dependency_must_point_backward"):
            parse_planner_semantic_output(
                text, goal="G", baseline_bootstrap=_baseline_bootstrap()
            )

    def test_unknown_dependency(self):
        payload = _valid_payload()
        payload["stages"][0]["depends_on"] = ["stage_99"]
        text = json.dumps(payload)
        with pytest.raises(StructuredOutputError, match="planner_semantic_unknown_dependency"):
            parse_planner_semantic_output(
                text, goal="G", baseline_bootstrap=_baseline_bootstrap()
            )

    def test_invalid_json(self):
        with pytest.raises(StructuredOutputError):
            parse_planner_semantic_output(
                "not json at all", goal="G", baseline_bootstrap=_baseline_bootstrap()
            )

    def test_stage_missing_required_fields(self):
        payload = _valid_payload()
        payload["stages"][0] = {"stage_id": "stage_1"}
        text = json.dumps(payload)
        with pytest.raises(StructuredOutputError, match="semantic_stage_missing_fields"):
            parse_planner_semantic_output(
                text, goal="G", baseline_bootstrap=_baseline_bootstrap()
            )

    def test_stage_empty_objective(self):
        payload = _valid_payload()
        payload["stages"][0]["objective"] = ""
        text = json.dumps(payload)
        with pytest.raises(StructuredOutputError, match="semantic_stage_requires_non_empty_id_title_objective"):
            parse_planner_semantic_output(
                text, goal="G", baseline_bootstrap=_baseline_bootstrap()
            )
