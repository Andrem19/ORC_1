"""
Deterministic ranking for prompt known-fact compaction.
"""

from __future__ import annotations

from typing import Any

from app.services.direct_execution.handle_hygiene import DURABLE_HANDLE_FIELDS

_LOW_SIGNAL_KEYS = (
    "direct.created_ids",
    "direct.successful_tool_names",
    "direct.statuses",
    "direct.provider",
)
_BACKTESTS_PRIORITY_SUFFIXES = (
    "backtests.candidate_handles",
    "backtests.analysis_refs",
    "backtests.integration_handles",
    "backtests.integration_refs",
    "feature_long_job",
    "feature_short_job",
    "diagnostics_run",
)
_BASELINE_SUFFIXES = (
    "baseline_snapshot_id",
    "baseline_version",
    "symbol",
    "anchor_timeframe",
    "execution_timeframe",
)


def compact_known_facts_for_prompt(
    *,
    known_facts: dict[str, Any],
    dependency_ids: list[str],
    required_facts: list[str],
    limit: int = 24,
) -> list[tuple[str, Any]]:
    dependency_set = {str(item).strip() for item in dependency_ids if str(item).strip()}
    required = {str(item).strip() for item in required_facts if str(item).strip()}
    ranked: list[tuple[int, int, str, str, Any]] = []
    seen_display_keys: set[str] = set()
    for raw_key, raw_value in known_facts.items():
        key_text = str(raw_key or "").strip()
        display_key = compact_fact_key(key_text)
        if display_key in seen_display_keys:
            continue
        seen_display_keys.add(display_key)
        ranked.append(
            (
                fact_priority_score(
                    raw_key=key_text,
                    dependency_ids=dependency_set,
                    required_facts=required,
                ),
                _priority_depth(key_text),
                display_key,
                key_text,
                raw_value,
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(display_key, raw_value) for _, _, display_key, _, raw_value in ranked[:limit]]


def compact_fact_key(raw_key: str) -> str:
    parts = [segment for segment in str(raw_key or "").split(".") if segment]
    if len(parts) <= 2:
        return str(raw_key)
    return f"{parts[0]}.{parts[-1]}"


def fact_priority_score(
    *,
    raw_key: str,
    dependency_ids: set[str],
    required_facts: set[str],
) -> int:
    score = 0
    key_text = str(raw_key or "").strip()
    first_segment = key_text.split(".", 1)[0]
    suffix = key_text.rsplit(".", 1)[-1]
    if first_segment in dependency_ids:
        score += 1000
    if key_text in required_facts or suffix in required_facts:
        score += 900
    if any(key_text == item or key_text.endswith(f".{item}") for item in required_facts):
        score += 700
    if suffix in DURABLE_HANDLE_FIELDS or key_text in DURABLE_HANDLE_FIELDS:
        score += 650
    if any(key_text == item or key_text.endswith(f".{item}") for item in _BACKTESTS_PRIORITY_SUFFIXES):
        score += 600
    if any(key_text == item or key_text.endswith(f".{item}") for item in _BASELINE_SUFFIXES):
        score += 450
    if key_text.startswith("prior."):
        score -= 150
    if "compiled_plan_" in key_text and key_text.count(".") >= 2:
        score -= 120
    if any(key_text == item or key_text.endswith(f".{item}") for item in _LOW_SIGNAL_KEYS):
        score -= 500
    return score


def _priority_depth(raw_key: str) -> int:
    return len([segment for segment in str(raw_key or "").split(".") if segment])


__all__ = [
    "compact_fact_key",
    "compact_known_facts_for_prompt",
    "fact_priority_score",
]
