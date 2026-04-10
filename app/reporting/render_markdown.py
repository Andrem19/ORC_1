"""
Fixed-template Russian markdown rendering for canonical reports.
"""

from __future__ import annotations

from app.reporting.models import PlanBatchReport, RunSummaryReport, SequenceExecutionReport


def render_plan_batch_report(report: PlanBatchReport) -> str:
    lines = [
        f"# Отчёт по batch plan `{report.plan_id}`",
        "",
        f"- Статус: `{report.status}`",
        f"- Источник: `{report.plan_source_kind}`",
        f"- Последовательность: `{report.source_sequence_id or '-'}`",
        f"- Исходный raw plan: `{report.source_raw_plan or '-'}`",
        f"- Начало: `{report.started_at or '-'}`",
        f"- Завершение: `{report.finished_at or '-'}`",
        f"- Длительность: `{report.duration_ms}` ms",
        f"- Итоговый вердикт: `{report.final_verdict or 'PENDING'}`",
        "",
        "## Кратко",
        report.narrative_sections_ru.executive_summary_ru or report.final_summary or "Итоговая сводка недоступна.",
        "",
        "## Слайсы",
    ]
    for item in report.slice_results:
        lines.append(f"- `{item.slice_id}` | `{item.status}` | `{item.verdict or 'PENDING'}` | {item.summary or '-'}")
    if report.narrative_sections_ru.key_findings_ru:
        lines.extend(["", "## Ключевые выводы"])
        lines.extend(f"- {item}" for item in report.narrative_sections_ru.key_findings_ru)
    if report.narrative_sections_ru.important_failures_ru:
        lines.extend(["", "## Важные проблемы"])
        lines.extend(f"- {item}" for item in report.narrative_sections_ru.important_failures_ru)
    if report.narrative_sections_ru.recommended_next_actions_ru:
        lines.extend(["", "## Что делать дальше"])
        lines.extend(f"- {item}" for item in report.narrative_sections_ru.recommended_next_actions_ru)
    if report.narrative_sections_ru.operator_notes_ru:
        lines.extend(["", "## Операторские заметки"])
        lines.extend(f"- {item}" for item in report.narrative_sections_ru.operator_notes_ru)
    return "\n".join(lines).strip() + "\n"


def render_sequence_report(report: SequenceExecutionReport) -> str:
    lines = [
        f"# Отчёт по sequence `{report.sequence_id}`",
        "",
        f"- Исходный файл: `{report.source_file}`",
        f"- Статус sequence: `{report.sequence_status}`",
        f"- Статус компиляции: `{report.compile_status or '-'}`",
        f"- Начало: `{report.started_at or '-'}`",
        f"- Завершение: `{report.finished_at or '-'}`",
        f"- Длительность: `{report.duration_ms}` ms",
        "",
        "## Кратко",
        report.narrative_sections_ru.executive_summary_ru or report.executive_summary_ru or "Итоговая сводка недоступна.",
        "",
        "## Batch-результаты",
    ]
    for item in report.batch_results:
        lines.append(
            f"- `{item.get('plan_id', '-')}` | `{item.get('status', '-')}` | "
            f"`{item.get('final_verdict', 'PENDING')}` | {item.get('summary', '-')}"
        )
    if report.narrative_sections_ru.key_findings_ru:
        lines.extend(["", "## Подтверждённые выводы"])
        lines.extend(f"- {item}" for item in report.narrative_sections_ru.key_findings_ru)
    elif report.confirmed_findings:
        lines.extend(["", "## Подтверждённые выводы"])
        lines.extend(f"- {item}" for item in report.confirmed_findings)
    if report.narrative_sections_ru.important_failures_ru:
        lines.extend(["", "## Важные сбои и риски"])
        lines.extend(f"- {item}" for item in report.narrative_sections_ru.important_failures_ru)
    elif report.blockers:
        lines.extend(["", "## Важные сбои и риски"])
        lines.extend(f"- {item}" for item in report.blockers)
    if report.narrative_sections_ru.recommended_next_actions_ru:
        lines.extend(["", "## Рекомендуемые действия"])
        lines.extend(f"- {item}" for item in report.narrative_sections_ru.recommended_next_actions_ru)
    elif report.recommended_next_actions:
        lines.extend(["", "## Рекомендуемые действия"])
        lines.extend(f"- {item}" for item in report.recommended_next_actions)
    if report.narrative_sections_ru.operator_notes_ru:
        lines.extend(["", "## Операторские заметки"])
        lines.extend(f"- {item}" for item in report.narrative_sections_ru.operator_notes_ru)
    return "\n".join(lines).strip() + "\n"


def render_run_summary_report(report: RunSummaryReport) -> str:
    lines = [
        f"# Итоговый отчёт по run `{report.run_id}`",
        "",
        f"- Источник планов: `{report.plan_source}`",
        f"- Причина остановки: `{report.stop_reason or '-'}`",
        f"- Начало: `{report.started_at or '-'}`",
        f"- Завершение: `{report.finished_at or '-'}`",
        f"- Длительность: `{report.duration_ms}` ms",
        "",
        "## Итог",
        report.narrative_sections_ru.executive_summary_ru or report.executive_summary_ru or "Итоговая сводка недоступна.",
        "",
        "## Очередь",
        f"- Всего raw plans: {report.total_raw_plans}",
        f"- Скомпилированных sequence: {report.compiled_sequences}",
        f"- Исполненных sequence: {report.executed_sequences}",
        f"- Завершено: {report.completed_sequences}",
        f"- С ошибкой: {report.failed_sequences}",
        f"- Частично: {report.partial_sequences}",
        f"- Пропущено: {report.skipped_sequences}",
        "",
        "## Direct Execution Metrics",
        f"- Direct completed slices: {report.direct_metrics.direct_completed}",
        f"- Direct blocked slices: {report.direct_metrics.direct_blocked}",
        f"- Direct failed slices: {report.direct_metrics.direct_failed}",
        f"- Direct parse retries: {report.direct_metrics.direct_parse_retries}",
        f"- Direct tool calls observed: {report.direct_metrics.direct_tool_calls_observed}",
        f"- Direct incidents: {report.direct_metrics.direct_incidents}",
        "",
        "## По sequence",
    ]
    for item in report.per_sequence_table:
        lines.append(
            f"- `{item.get('sequence_id', '-')}` | `{item.get('status', '-')}` | "
            f"`{item.get('top_verdict', 'PENDING')}` | {item.get('summary', '-')}"
        )
    if report.narrative_sections_ru.key_findings_ru:
        lines.extend(["", "## Ключевые результаты"])
        lines.extend(f"- {item}" for item in report.narrative_sections_ru.key_findings_ru)
    if report.narrative_sections_ru.important_failures_ru:
        lines.extend(["", "## Нерешённые проблемы"])
        lines.extend(f"- {item}" for item in report.narrative_sections_ru.important_failures_ru)
    elif report.unresolved_blockers:
        lines.extend(["", "## Нерешённые проблемы"])
        lines.extend(f"- {item}" for item in report.unresolved_blockers)
    actions = report.narrative_sections_ru.recommended_next_actions_ru or (
        report.continue_items + report.rerun_items + report.drop_items
    )
    if actions:
        lines.extend(["", "## Что делать дальше"])
        lines.extend(f"- {item}" for item in actions)
    operator_notes = report.narrative_sections_ru.operator_notes_ru or report.operator_notes_ru
    if operator_notes:
        lines.extend(["", "## Операторские заметки"])
        lines.extend(f"- {item}" for item in operator_notes)
    return "\n".join(lines).strip() + "\n"
