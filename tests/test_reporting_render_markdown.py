from __future__ import annotations

from app.execution_models import BaselineRef
from app.reporting.models import NarrativeSectionsRu, PlanBatchReport, RunSummaryReport, SequenceExecutionReport
from app.reporting.render_markdown import (
    render_plan_batch_report,
    render_run_summary_report,
    render_sequence_report,
)


def test_render_plan_batch_report_includes_operator_notes() -> None:
    report = PlanBatchReport(
        plan_id="plan_1",
        plan_source_kind="planner",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        narrative_sections_ru=NarrativeSectionsRu(
            executive_summary_ru="Краткая сводка.",
            operator_notes_ru=["Проверить спорные метрики перед следующим запуском."],
        ),
    )

    markdown = render_plan_batch_report(report)

    assert "## Операторские заметки" in markdown
    assert "Проверить спорные метрики" in markdown


def test_render_sequence_report_includes_operator_notes() -> None:
    report = SequenceExecutionReport(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash_1",
        sequence_id="compiled_plan_v1",
        narrative_sections_ru=NarrativeSectionsRu(
            executive_summary_ru="Sequence завершена.",
            operator_notes_ru=["Проверить compile warnings перед повтором."],
        ),
    )

    markdown = render_sequence_report(report)

    assert "## Операторские заметки" in markdown
    assert "Проверить compile warnings" in markdown


def test_render_run_summary_report_prefers_narrative_operator_notes_then_fallback() -> None:
    report = RunSummaryReport(
        run_id="run_1",
        plan_source="planner",
        goal="Goal",
        executive_summary_ru="Прогон завершён.",
        operator_notes_ru=["Fallback operator note."],
    )

    fallback_markdown = render_run_summary_report(report)
    assert "## Операторские заметки" in fallback_markdown
    assert "Fallback operator note." in fallback_markdown

    report.narrative_sections_ru = NarrativeSectionsRu(
        executive_summary_ru="Прогон завершён.",
        operator_notes_ru=["Narrative operator note."],
    )
    narrative_markdown = render_run_summary_report(report)

    assert "Narrative operator note." in narrative_markdown
    assert "Fallback operator note." not in narrative_markdown
