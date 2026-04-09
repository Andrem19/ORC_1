from __future__ import annotations

import asyncio

import pytest

from app.adapters.base import AdapterResponse, BaseAdapter
from app.reporting.narrative import NarrativeGenerationError, ReportNarrativeService


class _StubAdapter(BaseAdapter):
    def __init__(self, raw_output: str) -> None:
        self.raw_output = raw_output

    def invoke(self, prompt: str, timeout: int = 120, **kwargs):
        del prompt, timeout, kwargs
        return AdapterResponse(success=True, raw_output=self.raw_output)

    def is_available(self) -> bool:
        return True

    def name(self) -> str:
        return "claude_planner_cli"


def test_report_narrative_service_accepts_russian_payload() -> None:
    service = ReportNarrativeService(
        adapter=_StubAdapter(
            """
            {
              "executive_summary_ru": "Этот plan завершился успешно и дал полезные сигналы.",
              "key_findings_ru": ["Подтверждена готовность данных"],
              "important_failures_ru": [],
              "recommended_next_actions_ru": ["Запустить следующий backtest"],
              "operator_notes_ru": ["Проверить orthogonality на следующем шаге"]
            }
            """
        ),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )

    sections = asyncio.run(service.generate(report_kind="plan_batch", payload={"plan_id": "plan_1"}))

    assert sections.executive_summary_ru.startswith("Этот plan")
    assert sections.recommended_next_actions_ru == ["Запустить следующий backtest"]


def test_report_narrative_service_rejects_non_russian_payload() -> None:
    service = ReportNarrativeService(
        adapter=_StubAdapter(
            """
            {
              "executive_summary_ru": "This plan completed successfully.",
              "key_findings_ru": ["Data was ready"],
              "important_failures_ru": [],
              "recommended_next_actions_ru": ["Run the next backtest"],
              "operator_notes_ru": ["Check orthogonality next"]
            }
            """
        ),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )

    with pytest.raises(NarrativeGenerationError, match="narrative_not_russian"):
        asyncio.run(service.generate(report_kind="plan_batch", payload={"plan_id": "plan_1"}))


def test_report_narrative_service_rejects_english_summary_even_with_russian_other_fields() -> None:
    service = ReportNarrativeService(
        adapter=_StubAdapter(
            """
            {
              "executive_summary_ru": "Completed successfully with strong metrics.",
              "key_findings_ru": ["Подтверждена готовность данных"],
              "important_failures_ru": [],
              "recommended_next_actions_ru": ["Запустить следующий backtest"],
              "operator_notes_ru": ["Проверить orthogonality на следующем шаге"]
            }
            """
        ),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )

    with pytest.raises(NarrativeGenerationError, match="narrative_not_russian"):
        asyncio.run(service.generate(report_kind="plan_batch", payload={"plan_id": "plan_1"}))


def test_report_narrative_service_accepts_russian_summary_with_technical_terms() -> None:
    service = ReportNarrativeService(
        adapter=_StubAdapter(
            """
            {
              "executive_summary_ru": "Этот backtest для BTCUSDT завершился успешно и показал рост Sharpe.",
              "key_findings_ru": ["Подтверждена готовность feature pipeline"],
              "important_failures_ru": [],
              "recommended_next_actions_ru": ["Запустить следующий backtest"],
              "operator_notes_ru": ["Проверить drift после обновления features"]
            }
            """
        ),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )

    sections = asyncio.run(service.generate(report_kind="run", payload={"run_id": "run_1"}))

    assert sections.executive_summary_ru.startswith("Этот backtest")
