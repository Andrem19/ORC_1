"""
Optional Russian narrative generation for canonical reports.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.adapters.base import BaseAdapter
from app.execution_parsing import StructuredOutputError, extract_json_object
from app.reporting.models import NarrativeSectionsRu
from app.reporting.prompts import build_narrative_prompt
from app.services.direct_execution.invocation import AdapterInvocationError, invoke_adapter_with_retries


class NarrativeGenerationError(RuntimeError):
    """Raised when Russian narrative generation fails validation."""


class ReportNarrativeService:
    def __init__(
        self,
        *,
        adapter: BaseAdapter | None,
        timeout_seconds: int,
        retry_attempts: int,
        retry_backoff_seconds: float,
        enabled: bool = True,
    ) -> None:
        self.adapter = adapter
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.enabled = enabled and adapter is not None and self._adapter_supported(adapter) and self._adapter_available(adapter)

    async def generate(self, *, report_kind: str, payload: dict[str, Any]) -> NarrativeSectionsRu:
        if not self.enabled or self.adapter is None:
            raise NarrativeGenerationError("narrative_generation_disabled")
        prompt = build_narrative_prompt(report_kind=report_kind, payload=payload)
        try:
            response = await invoke_adapter_with_retries(
                adapter=self.adapter,
                prompt=prompt,
                timeout_seconds=self.timeout_seconds,
                max_attempts=self.retry_attempts,
                base_backoff_seconds=self.retry_backoff_seconds,
            )
        except AdapterInvocationError as exc:
            raise NarrativeGenerationError(str(exc)) from exc
        if not response.success:
            raise NarrativeGenerationError(response.error or "narrative_invoke_failed")
        return _parse_narrative(response.raw_output)

    @staticmethod
    def _adapter_supported(adapter: BaseAdapter) -> bool:
        try:
            name = adapter.name()
        except Exception:
            return False
        return name not in {"fake_planner"}

    @staticmethod
    def _adapter_available(adapter: BaseAdapter) -> bool:
        try:
            return bool(adapter.is_available())
        except Exception:
            return False


def _parse_narrative(text: str) -> NarrativeSectionsRu:
    try:
        payload = extract_json_object(text)
    except StructuredOutputError as exc:
        raise NarrativeGenerationError(str(exc)) from exc
    required = {
        "executive_summary_ru",
        "key_findings_ru",
        "important_failures_ru",
        "recommended_next_actions_ru",
        "operator_notes_ru",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise NarrativeGenerationError(f"narrative_missing_fields:{','.join(missing)}")
    sections = NarrativeSectionsRu(
        executive_summary_ru=str(payload.get("executive_summary_ru", "") or "").strip(),
        key_findings_ru=_string_list(payload.get("key_findings_ru", [])),
        important_failures_ru=_string_list(payload.get("important_failures_ru", [])),
        recommended_next_actions_ru=_string_list(payload.get("recommended_next_actions_ru", [])),
        operator_notes_ru=_string_list(payload.get("operator_notes_ru", [])),
    )
    if not sections.executive_summary_ru:
        raise NarrativeGenerationError("narrative_requires_executive_summary_ru")
    if not _looks_russian(asdict(sections)):
        raise NarrativeGenerationError("narrative_not_russian")
    return sections


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise NarrativeGenerationError("narrative_field_must_be_list")
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _looks_russian(payload: dict[str, Any]) -> bool:
    executive_summary = str(payload.get("executive_summary_ru", "") or "").strip()
    if not executive_summary:
        return False
    if _cyrillic_ratio(executive_summary) < 0.3:
        return False
    text = " ".join(
        value if isinstance(value, str) else " ".join(str(item) for item in value)
        for value in payload.values()
    )
    cyrillic = sum(1 for char in text if "а" <= char.lower() <= "я" or char.lower() == "ё")
    return cyrillic >= 10


def _cyrillic_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    cyrillic = sum(1 for char in letters if "а" <= char.lower() <= "я" or char.lower() == "ё")
    return cyrillic / len(letters)
