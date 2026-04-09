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
