"""
Prompt builders for Russian narrative generation over canonical reports.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any


NARRATIVE_SCHEMA = """{
  "executive_summary_ru": "2-5 предложений на русском",
  "key_findings_ru": ["короткий вывод на русском"],
  "important_failures_ru": ["важный сбой или риск на русском"],
  "recommended_next_actions_ru": ["что делать дальше на русском"],
  "operator_notes_ru": ["короткая операторская заметка на русском"]
}"""


def build_narrative_prompt(*, report_kind: str, payload: dict[str, Any]) -> str:
    compact_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return dedent(
        f"""
        Ты готовишь финальную операторскую сводку по canonical JSON report.
        Пиши ТОЛЬКО на русском языке. Не смешивай английский и русский, кроме неизбежных идентификаторов вроде plan_id, run_id, sequence_id, tool names и file paths.
        Не выдумывай факты. Используй только данные из JSON ниже.
        Нужен только структурированный JSON-ответ по схеме, без markdown и без пояснений.

        Тип отчёта: {report_kind}

        JSON report:
        {compact_json}

        Верни JSON строго по схеме:
        {NARRATIVE_SCHEMA}
        """
    ).strip()


SEQUENCE_NARRATIVE_SCHEMA = """{
  "executive_summary_ru": "3-7 предложений на русском: что за sequence, какие батчи прошли, общий результат",
  "key_findings_ru": ["подробный вывод на русском: какие features/signals исследовались, результаты backtest-сравнений, метрики"],
  "important_failures_ru": ["важный сбой, заблокированный батч или риск на русском с деталями"],
  "recommended_next_actions_ru": ["конкретное следующее действие на русском с указанием feature/snapshot/run_id"],
  "operator_notes_ru": ["операторская заметка на русском: рекомендации по дальнейшим исследованиям"]
}"""


def build_sequence_narrative_prompt(*, payload: dict[str, Any]) -> str:
    compact_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return dedent(
        f"""
        Ты — аналитик торговых стратегий. Готовишь развёрнутую операторскую сводку по итогам выполнения compiled sequence.
        Пиши ТОЛЬКО на русском языке. Английские идентификаторы (plan_id, run_id, sequence_id, snapshot_id, feature names, tool names) оставляй как есть.

        Твои задачи:
        1. Проанализировать что было сделано: какие slices выполнялись, какие tools использовались, какие features/signals исследовались.
        2. Описать результаты: backtest-сравнения, метрики (Sharpe, net PnL, max DD, trade count), были ли improvements относительно baseline.
        3. Выявить провалы и блокировки: какие slices упали, почему, что не получилось.
        4. Предложить конкретные следующие шаги для дальнейшего исследования.

        Не выдумывай факты. Используй только данные из JSON ниже. Если данных недостаточно для вывода — так и скажи.

        JSON sequence report:
        {compact_json}

        Верни JSON строго по схеме:
        {SEQUENCE_NARRATIVE_SCHEMA}
        """
    ).strip()
