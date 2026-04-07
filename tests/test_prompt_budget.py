"""Tests for global revision prompt budget."""

from __future__ import annotations

from app.plan_prompt_budget import (
    REVISION_PROMPT_BUDGET,
    SECTION_BUDGETS,
    apply_global_budget,
    compact_reports_for_revision,
    truncate_text,
)
from app.plan_models import TaskReport


class TestApplyGlobalBudget:
    def test_within_budget_no_changes(self):
        sections = {
            "goal": "Short goal",
            "context": "Some context",
            "worker_reports": "r1: ok",
        }
        result = apply_global_budget(sections)
        assert result == sections

    def test_truncates_oversized_section(self):
        long_text = "x" * 5000
        sections = {"context": long_text}
        result = apply_global_budget(sections)
        assert len(result["context"]) <= SECTION_BUDGETS["context"] + 20  # truncation marker

    def test_proportional_reduction_when_total_exceeds(self):
        sections = {
            "goal": "g" * 5000,
            "context": "c" * 5000,
            "worker_reports": "w" * 5000,
            "research_history": "h" * 5000,
        }
        result = apply_global_budget(sections)
        total = sum(len(v) for v in result.values())
        assert total <= REVISION_PROMPT_BUDGET

    def test_preserves_fixed_sections(self):
        sections = {
            "revision_context": "Fixed text that should not be truncated even if long" * 50,
            "goal": "short",
        }
        result = apply_global_budget(sections)
        assert result["revision_context"] == sections["revision_context"]

    def test_empty_sections_preserved(self):
        sections = {"goal": "", "context": "data"}
        result = apply_global_budget(sections)
        assert result["goal"] == ""

    def test_multiple_sections_truncated_individually(self):
        sections = {
            "mcp_problems": "p" * 5000,
            "validation_warnings": "w" * 5000,
            "anti_patterns": "a" * 5000,
        }
        result = apply_global_budget(sections)
        assert len(result["mcp_problems"]) <= SECTION_BUDGETS["mcp_problems"] + 20
        assert len(result["validation_warnings"]) <= SECTION_BUDGETS["validation_warnings"] + 20
        assert len(result["anti_patterns"]) <= SECTION_BUDGETS["anti_patterns"] + 20


class TestCompactReportsReduced:
    def test_max_reports_is_3(self):
        reports = [
            TaskReport(
                task_id=f"t{i}", worker_id="w1", status="success",
                verdict="PROMOTE", plan_version=1,
                what_was_done=f"Did thing {i}",
            )
            for i in range(6)
        ]
        result = compact_reports_for_revision(reports)
        # Should have 3 report entries + 1 "3 more reports omitted" line
        report_lines = [l for l in result.split("\n") if l.startswith("- stage")]
        assert len(report_lines) == 3
        assert "3 more reports omitted" in result

    def test_what_was_done_truncated_at_160(self):
        reports = [
            TaskReport(
                task_id="t1", worker_id="w1", status="success",
                verdict="PROMOTE", plan_version=1,
                what_was_done="A" * 300,
            )
        ]
        result = compact_reports_for_revision(reports)
        for line in result.split("\n"):
            if line.startswith("  done:"):
                assert len(line) <= 175  # "  done: " prefix + 160 + truncation marker
                break
