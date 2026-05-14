"""
Validate model abort claims against the actual tool transcript.

Weak providers (e.g. MiniMax-M2.5) sometimes abort with claims that tools
returned "empty results" when the transcript clearly shows rich data.  This
module detects such hallucinated aborts and produces a correction prompt
that can be injected into the retry to give the model a second chance with
ground-truth evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any

_EMPTY_RESULT_CLAIM_PATTERNS = (
    re.compile(r"empty.?results?", re.IGNORECASE),
    re.compile(r"no\s+(custom\s+)?features?\s+(found|available)", re.IGNORECASE),
    re.compile(r"no\s+(columns?|datasets?)\s+(found|available)", re.IGNORECASE),
    re.compile(r"no\s+data\s+available", re.IGNORECASE),
    re.compile(r"INFRASTRUCTURE_DATA_UNAVAILABLE", re.IGNORECASE),
)


def abort_claims_empty_results(reason_code: str, summary: str, raw_output: str) -> bool:
    """Return True when the abort reason claims tools returned empty results."""
    haystack = " | ".join(
        text.strip()
        for text in (str(reason_code or ""), str(summary or ""), str(raw_output or ""))
        if text and text.strip()
    )
    if not haystack:
        return False
    return any(pat.search(haystack) for pat in _EMPTY_RESULT_CLAIM_PATTERNS)


def transcript_has_successful_tool_data(transcript: list[dict[str, Any]]) -> bool:
    """Return True when the transcript contains at least one successful tool result."""
    for entry in transcript:
        if entry.get("kind") != "tool_result":
            continue
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        if payload.get("error") or payload.get("ok") is False:
            continue
        return True
    return False


def extract_transcript_evidence_summary(transcript: list[dict[str, Any]], max_items: int = 8) -> list[str]:
    """Extract key evidence items from successful tool results in the transcript.

    Returns a list of human-readable strings summarising what the tools returned,
    e.g. ["features_dataset returned 27 columns: atr_1, cl_15m, ...",
          "features_custom contract returned: entrypoint=compute_series"].
    """
    items: list[str] = []
    for entry in transcript:
        if entry.get("kind") != "tool_result":
            continue
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        if payload.get("error") or payload.get("ok") is False:
            continue
        args = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        tool_name = str(entry.get("tool") or args.get("tool_name") or "").strip()
        summary = _summarise_tool_result(tool_name, args, payload)
        if summary:
            items.append(summary)
        if len(items) >= max_items:
            break
    return items


def build_transcript_correction_prompt(transcript: list[dict[str, Any]]) -> str:
    """Build a correction prompt section when the model hallucinated empty results.

    Returns an empty string when no correction is warranted.
    """
    evidence = extract_transcript_evidence_summary(transcript)
    if not evidence:
        return ""
    lines = [
        "## CRITICAL: Transcript validation correction",
        "",
        "The previous attempt **incorrectly** claimed that tools returned empty results.",
        "The transcript below proves that tools DID return data.  Do NOT repeat the",
        "false claim of empty results.  Use the actual data listed here:",
        "",
    ]
    for item in evidence:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Proceed with the slice objective using the data confirmed available above.")
    lines.append("Do NOT abort or return empty-results claims for these tools.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _summarise_tool_result(tool_name: str, args: dict[str, Any], payload: dict[str, Any]) -> str:
    """Produce a one-line summary of a successful tool result."""
    structured = _extract_structured_content(payload.get("payload"))
    data = structured.get("data") if isinstance(structured, dict) else None

    if tool_name == "features_dataset":
        columns = _deep_get_list(data, "columns")
        if columns:
            return f"features_dataset returned {len(columns)} columns: {', '.join(str(c) for c in columns[:12])}{'...' if len(columns) > 12 else ''}"
        datasets = _deep_get_list(data, "datasets")
        if datasets:
            return f"features_dataset returned {len(datasets)} datasets"

    if tool_name == "features_custom":
        contract = data.get("contract") if isinstance(data, dict) else None
        if isinstance(contract, dict):
            entrypoint = contract.get("preferred_entrypoint_name") or contract.get("entrypoint")
            return f"features_custom contract available: entrypoint={entrypoint}"
        features = _deep_get_list(data, "features")
        if features:
            names = [str(f.get("name", "")) for f in features if isinstance(f, dict) and f.get("name")]
            if names:
                return f"features_custom list returned {len(names)} features: {', '.join(names[:8])}"

    if tool_name == "models_dataset":
        datasets = _deep_get_list(data, "datasets")
        if datasets:
            return f"models_dataset returned {len(datasets)} datasets"
        contract = data.get("contract") if isinstance(data, dict) else None
        if contract:
            return "models_dataset contract available"

    if tool_name == "research_memory":
        matches = data.get("memory_matches") if isinstance(data, dict) else None
        if isinstance(matches, dict):
            total = matches.get("total", 0)
            results = matches.get("results") or []
            return f"research_memory returned {total} nodes ({len(results)} shown)"

    if tool_name == "datasets":
        datasets = _deep_get_list(data, "datasets")
        if datasets:
            return f"datasets returned {len(datasets)} items"

    if tool_name in ("events", "events_sync"):
        families = data.get("families") if isinstance(data, dict) else None
        if families:
            return f"{tool_name} returned families: {families}"

    # Generic fallback: report that the tool returned a successful response
    status = str(structured.get("status") or "").strip() if isinstance(structured, dict) else ""
    if status and status.lower() not in ("error", "failed"):
        return f"{tool_name} returned status={status}"

    return ""


def _extract_structured_content(raw_payload: Any) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        return {}
    structured = raw_payload.get("structuredContent")
    return structured if isinstance(structured, dict) else {}


def _deep_get_list(data: Any, key: str) -> list:
    """Extract a list from a nested dict by key (first level only)."""
    if not isinstance(data, dict):
        return []
    value = data.get(key)
    return value if isinstance(value, list) else []


__all__ = [
    "abort_claims_empty_results",
    "build_transcript_correction_prompt",
    "extract_transcript_evidence_summary",
    "transcript_has_successful_tool_data",
]
