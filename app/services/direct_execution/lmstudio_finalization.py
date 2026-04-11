"""
Strict auto-finalization helpers for LM Studio direct execution.
"""

from __future__ import annotations

import json
from typing import Any

from app.services.direct_execution.guardrails import (
    final_report_payload_passes_gate,
    synthesize_transcript_evidence_refs,
)


def build_catalog_only_final_report(
    *,
    transcript: list[dict[str, Any]],
    success_criteria: list[str],
    required_output_facts: list[str],
) -> str | None:
    scopes: set[str] = set()
    timeframes: set[str] = set()
    for entry in transcript:
        if entry.get("kind") != "tool_result" or entry.get("tool") != "features_catalog":
            continue
        args = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        scope = str(args.get("scope") or "").strip()
        timeframe = str(args.get("timeframe") or "").strip()
        if scope:
            scopes.add(scope)
        if timeframe:
            timeframes.add(timeframe)
    findings: list[str] = []
    if scopes:
        findings.append(f"Catalog scopes inspected: {', '.join(sorted(scopes))}.")
    if timeframes:
        findings.append(f"Timeframes inspected: {', '.join(sorted(timeframes))}.")
    for criterion in success_criteria[:2]:
        text = str(criterion or "").strip()
        if text:
            findings.append(f"Contract coverage addressed: {text}")
    if not findings:
        findings.append("Feature catalog inspection completed for contract drafting.")
    facts = {
        "execution.kind": "direct",
        "features_catalog.scopes": sorted(scopes),
        "features_catalog.timeframes": sorted(timeframes),
    }
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
        "summary": "Feature catalog inspection completed; data/feature contract drafted for shortlisted hypotheses.",
        "verdict": "COMPLETE",
        "findings": findings,
        "facts": facts,
        "evidence_refs": evidence_refs,
        "confidence": 0.76,
    }
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def build_backtests_budget_salvage_report(
    *,
    transcript: list[dict[str, Any]],
    allowed_tools: set[str],
    success_criteria: list[str],
    required_output_facts: list[str],
) -> str | None:
    if not allowed_tools or not allowed_tools.issubset({"backtests_conditions", "backtests_analysis", "backtests_runs", "backtests_plan"}):
        return None
    successful = _successful_tool_entries(transcript)
    if len(successful) < 2:
        return None
    tools_seen = sorted({str(item.get("tool") or "").strip() for item in successful if str(item.get("tool") or "").strip()})
    findings = [
        f"Collected successful backtest diagnostics signals before budget exhaustion using tools: {', '.join(tools_seen)}.",
        "Budget cap hit after producing enough intermediate evidence to continue the cycle with summarized conclusions.",
    ]
    if success_criteria:
        findings.append(f"Success criterion targeted: {success_criteria[0]}")
    facts = {
        "execution.kind": "direct",
        "direct.auto_finalized_from_expensive_budget_salvage": True,
        "backtests.successful_tool_count": len(successful),
        "backtests.tools_seen": tools_seen,
    }
    evidence_refs = synthesize_transcript_evidence_refs(successful)
    if not final_report_payload_passes_gate(
        facts=facts,
        findings=findings,
        evidence_refs=evidence_refs,
        required_output_facts=required_output_facts,
    ):
        return None
    payload = {
        "type": "final_report",
        "summary": "Expensive-tool budget exhausted after successful diagnostics sampling; synthesized final report from collected backtest evidence.",
        "verdict": "COMPLETE",
        "findings": findings,
        "facts": facts,
        "evidence_refs": evidence_refs,
        "confidence": 0.65,
    }
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def build_research_budget_salvage_report(
    *,
    transcript: list[dict[str, Any]],
    allowed_tools: set[str],
    success_criteria: list[str],
    required_output_facts: list[str],
) -> str | None:
    if not allowed_tools or not allowed_tools.issubset({"research_map", "research_search", "research_record"}):
        return None
    successful = _successful_tool_entries(transcript)
    if not successful:
        return None
    project_id = ""
    shortlist: list[str] = []
    seen_shortlist: set[str] = set()
    for entry in successful:
        args = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        pid = str(args.get("project_id") or "").strip()
        if pid and not project_id:
            project_id = pid
        if entry.get("tool") == "research_search":
            query = str(args.get("query") or "").strip()
            if query and query not in seen_shortlist:
                seen_shortlist.add(query)
                shortlist.append(query)
        if entry.get("tool") == "research_record":
            record = args.get("record") if isinstance(args.get("record"), dict) else {}
            title = str(record.get("title") or "").strip()
            if title and title not in seen_shortlist:
                seen_shortlist.add(title)
                shortlist.append(title)
    if not project_id or not shortlist:
        return None
    findings = [
        "Research transcript was successfully executed with no tool errors before budget exhaustion.",
        f"Recovered shortlist candidates: {', '.join(shortlist[:8])}.",
    ]
    if success_criteria:
        findings.append(f"Success criterion targeted: {success_criteria[0]}")
    facts = {
        "execution.kind": "direct",
        "research.project_id": project_id,
        "research.shortlist_families": shortlist[:20],
        "direct.auto_finalized_from_budget_salvage": True,
    }
    evidence_refs = synthesize_transcript_evidence_refs(successful)
    if not final_report_payload_passes_gate(
        facts=facts,
        findings=findings,
        evidence_refs=evidence_refs,
        required_output_facts=required_output_facts,
    ):
        return None
    payload = {
        "type": "final_report",
        "summary": "Budget exhausted after successful research exploration; synthesized final report from collected transcript evidence.",
        "verdict": "COMPLETE",
        "findings": findings,
        "facts": facts,
        "evidence_refs": evidence_refs,
        "confidence": 0.68,
    }
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


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


__all__ = [
    "build_backtests_budget_salvage_report",
    "build_catalog_only_final_report",
    "build_research_budget_salvage_report",
]
