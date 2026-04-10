"""
Derive reusable downstream facts from direct tool transcripts.
"""

from __future__ import annotations

from typing import Any


def derive_facts_from_transcript(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    shortlist: list[str] = []
    hypothesis_refs: list[str] = []
    for item in transcript:
        if item.get("kind") != "tool_result":
            continue
        tool = str(item.get("tool") or "").strip()
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        structured = _extract_structured(payload.get("payload"))
        data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
        if tool == "research_map":
            project_id = str(data.get("project_id") or "").strip()
            if project_id:
                facts.setdefault("research.project_id", project_id)
            dims = data.get("dimensions")
            if isinstance(dims, list):
                names = [str(dim.get("name") or "").strip() for dim in dims if isinstance(dim, dict) and str(dim.get("name") or "").strip()]
                if names:
                    facts.setdefault("atlas_dimensions", names)
        if tool == "research_record":
            kind = str(args.get("kind") or "").strip()
            record = data.get("record") if isinstance(data.get("record"), dict) else {}
            record_project_id = str(record.get("project_id") or data.get("project_id") or "").strip()
            if record_project_id:
                facts.setdefault("research.project_id", record_project_id)
            node_id = str(record.get("node_id") or ((data.get("record_refs") or {}).get("memory_node_id") if isinstance(data.get("record_refs"), dict) else "") or "").strip()
            title = str(record.get("title") or "").strip()
            if kind == "hypothesis":
                if node_id and node_id not in hypothesis_refs:
                    hypothesis_refs.append(node_id)
                family = _family_name_from_title(title)
                if family and family not in shortlist:
                    shortlist.append(family)
            elif kind == "milestone":
                content_text = _extract_record_text(record)
                for family in _families_from_text(content_text):
                    if family not in shortlist:
                        shortlist.append(family)
    if shortlist:
        facts["research.shortlist_families"] = shortlist[:20]
    if hypothesis_refs:
        facts["research.hypothesis_refs"] = hypothesis_refs[:20]
    return facts


def _extract_structured(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    structured = payload.get("structuredContent")
    return structured if isinstance(structured, dict) else {}


def _family_name_from_title(title: str) -> str:
    text = str(title or "").strip()
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    return text


def _extract_record_text(record: dict[str, Any]) -> str:
    content = record.get("content")
    if isinstance(content, dict):
        return str(content.get("text") or "").strip()
    return ""


def _families_from_text(text: str) -> list[str]:
    values: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if ". " in stripped and stripped[:1].isdigit():
            candidate = stripped.split(". ", 1)[1].split(" - ", 1)[0].strip()
            if candidate and candidate not in values:
                values.append(candidate)
    return values


__all__ = ["derive_facts_from_transcript"]
