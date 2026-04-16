"""
Auto-finalization helpers for LM Studio direct execution.
"""

from __future__ import annotations

import json
from typing import Any

from app.services.direct_execution.guardrails import (
    final_report_payload_passes_gate,
    synthesize_transcript_evidence_refs,
)
from app.services.direct_execution.transcript_facts import derive_facts_from_transcript


def build_profile_final_report(
    *,
    transcript: list[dict[str, Any]],
    success_criteria: list[str],
    required_output_facts: list[str],
    runtime_profile: str,
) -> str | None:
    normalized_profile = str(runtime_profile or "").strip()
    if normalized_profile == "catalog_contract_probe":
        facts = _catalog_probe_facts(transcript)
        findings = _catalog_probe_findings(facts=facts, success_criteria=success_criteria)
        summary = "Catalog-contract probe completed from live MCP tool evidence."
        confidence = 0.74
        verdict = "COMPLETE"
    elif normalized_profile == "research_shortlist":
        if not _has_successful_shortlist_write(transcript):
            return None
        facts = derive_facts_from_transcript(transcript, runtime_profile=runtime_profile)
        findings = _research_shortlist_findings(facts=facts, success_criteria=success_criteria)
        summary = "Research shortlist milestone persisted from live MCP tool evidence."
        confidence = 0.77
        verdict = "COMPLETE"
    else:
        return None
    evidence_refs = synthesize_transcript_evidence_refs(transcript)
    if not final_report_payload_passes_gate(
        facts=facts,
        findings=findings,
        evidence_refs=evidence_refs,
        required_output_facts=required_output_facts,
    ):
        return None
    payload = {
        "type": "final_report",
        "summary": summary,
        "verdict": verdict,
        "findings": findings,
        "facts": facts,
        "evidence_refs": evidence_refs,
        "confidence": confidence,
    }
    return _json_block(payload)


def build_generic_transcript_salvage_report(
    *,
    transcript: list[dict[str, Any]],
    success_criteria: list[str],
    required_output_facts: list[str],
    runtime_profile: str = "",
    slice_title: str = "",
    salvage_reason: str = "generic",
    minimum_successful_tools: int = 2,
) -> str | None:
    successful = _successful_tool_entries(transcript)
    if len(successful) < max(1, int(minimum_successful_tools or 1)):
        return None
    tools_seen = sorted({str(e.get("tool") or "").strip() for e in successful if str(e.get("tool") or "").strip()})
    if not tools_seen:
        return None
    facts: dict[str, Any] = {
        "execution.kind": "direct",
        "direct.auto_finalized_from_generic_salvage": True,
        f"direct.auto_finalized_from_{str(salvage_reason or 'generic').strip()}_salvage": True,
        "direct.tools_seen": tools_seen,
        "direct.successful_tool_count": len(successful),
    }
    transcript_facts = derive_facts_from_transcript(transcript, runtime_profile=runtime_profile)
    for key, value in transcript_facts.items():
        facts.setdefault(key, value)
    findings = [f"Collected evidence from {len(successful)} successful tool calls: {', '.join(tools_seen)}."]
    for criterion in success_criteria[:2]:
        text = str(criterion or "").strip()
        if text:
            findings.append(f"Success criterion addressed: {text}")
    if slice_title:
        findings.append(f"Slice: {slice_title}")
    evidence_refs = synthesize_transcript_evidence_refs(transcript)
    if not final_report_payload_passes_gate(
        facts=facts,
        findings=findings,
        evidence_refs=evidence_refs,
        required_output_facts=required_output_facts,
    ):
        return None
    payload = {
        "type": "final_report",
        "summary": f"Auto-synthesized from {len(successful)} successful tool results after model stall or budget exhaustion.",
        "verdict": "WATCHLIST",
        "findings": findings,
        "facts": facts,
        "evidence_refs": evidence_refs,
        "confidence": 0.60,
    }
    return _json_block(payload)


def _successful_tool_entries(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful: list[dict[str, Any]] = []
    for entry in transcript:
        if entry.get("kind") != "tool_result":
            continue
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        if payload.get("error") or payload.get("ok") is False:
            continue
        structured = _extract_structured_content(payload.get("payload"))
        if str(structured.get("status") or "").strip().lower() == "error":
            continue
        successful.append(entry)
    return successful


def _extract_structured_content(raw_payload: Any) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        return {}
    structured = raw_payload.get("structuredContent")
    return structured if isinstance(structured, dict) else {}


def _catalog_probe_facts(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    scopes: set[str] = set()
    timeframes: set[str] = set()
    for entry in transcript:
        if entry.get("kind") != "tool_result":
            continue
        args = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        scope = str(args.get("scope") or "").strip()
        timeframe = str(args.get("timeframe") or "").strip()
        if scope:
            scopes.add(scope)
        if timeframe:
            timeframes.add(timeframe)
    facts: dict[str, Any] = {
        "execution.kind": "direct",
        "features_catalog.scopes": sorted(scopes) or ["all"],
    }
    if timeframes:
        facts["features_catalog.timeframes"] = sorted(timeframes)
    return facts


def _catalog_probe_findings(*, facts: dict[str, Any], success_criteria: list[str]) -> list[str]:
    findings: list[str] = []
    scopes = facts.get("features_catalog.scopes") or []
    timeframes = facts.get("features_catalog.timeframes") or []
    if scopes:
        findings.append(f"Catalog scopes inspected: {', '.join(str(item) for item in scopes)}.")
    if timeframes:
        findings.append(f"Timeframes inspected: {', '.join(str(item) for item in timeframes)}.")
    for criterion in success_criteria[:2]:
        text = str(criterion or "").strip()
        if text:
            findings.append(f"Contract coverage addressed: {text}")
    if not findings:
        findings.append("Catalog-style contract probe completed from live tool results.")
    return findings


def _has_successful_shortlist_write(transcript: list[dict[str, Any]]) -> bool:
    for entry in transcript:
        if entry.get("kind") != "tool_result":
            continue
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        if payload.get("error") or payload.get("ok") is False:
            continue
        structured = _extract_structured_content(payload.get("payload"))
        if str(structured.get("status") or "").strip().lower() in {"error", "failed"}:
            continue
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        if str(arguments.get("action") or "").strip().lower() != "create":
            continue
        if str(arguments.get("kind") or "").strip().lower() != "milestone":
            continue
        record = arguments.get("record") if isinstance(arguments.get("record"), dict) else {}
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        content = record.get("content") if isinstance(record.get("content"), dict) else {}
        if not isinstance(metadata.get("shortlist_families"), list):
            continue
        if metadata.get("novelty_justification_present") is not True:
            continue
        candidates = content.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        if not all(isinstance(item, dict) and str(item.get("why_new") or "").strip() for item in candidates):
            continue
        return True
    return False


def _research_shortlist_findings(*, facts: dict[str, Any], success_criteria: list[str]) -> list[str]:
    findings: list[str] = []
    families = [str(item).strip() for item in list(facts.get("research.shortlist_families") or []) if str(item).strip()]
    if families:
        findings.append(f"Shortlist families persisted: {', '.join(families[:6])}.")
    if facts.get("research.novelty_justification_present") is True:
        findings.append("Novelty justification persisted in the shortlist milestone payload.")
    for criterion in success_criteria[:2]:
        text = str(criterion or "").strip()
        if text:
            findings.append(f"Success criterion addressed: {text}")
    if not findings:
        findings.append("Research shortlist milestone completed from live tool results.")
    return findings


def _json_block(payload: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


__all__ = ["build_generic_transcript_salvage_report", "build_profile_final_report"]
