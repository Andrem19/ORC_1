"""
Rule-based preprocessing for raw markdown research plans.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.raw_plan_models import RawPlanDocument, RawPlanStageFragment

_STAGE_HEADING_RE = re.compile(
    r"(?mi)^(#{1,3})\s*(?:этап|etap|stage)\s+(\d+)\s*[\.\:\-]?\s*(.+?)\s*$"
)
_SECTION_HEADING_RE = re.compile(r"(?mi)^#{2,4}\s+(.+?)\s*$")
_TITLE_RE = re.compile(r"(?m)^#\s+(.+?)\s*$")
_BASELINE_RE = re.compile(r"`(?P<snapshot>[a-z0-9\-]+)@(?P<version>\d+)`", re.IGNORECASE)


def parse_raw_plan_file(path: str | Path) -> RawPlanDocument:
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    title = _extract_title(normalized) or source_path.stem
    stage_matches = list(_STAGE_HEADING_RE.finditer(normalized))
    pre_stage_context = normalized[: stage_matches[0].start()].strip() if stage_matches else normalized
    candidate_stages = _extract_candidate_stages(normalized, stage_matches)
    parser_warnings: list[str] = []
    if not candidate_stages:
        parser_warnings.append("stage_headings_not_found")
    parse_confidence = _parse_confidence(candidate_stages)
    if parse_confidence < 0.45:
        parser_warnings.append("low_stage_parse_confidence")
    return RawPlanDocument(
        source_file=str(source_path),
        source_hash=digest,
        title=title,
        version_label=_extract_version_label(source_path.stem, title),
        normalized_text=normalized,
        pre_stage_context=pre_stage_context,
        baseline_ref_hint=_extract_baseline_hint(normalized),
        global_sections=_extract_global_sections(pre_stage_context),
        candidate_stages=candidate_stages,
        parser_warnings=parser_warnings,
        parse_confidence=parse_confidence,
    )


def _extract_title(text: str) -> str:
    match = _TITLE_RE.search(text)
    return str(match.group(1)).strip() if match else ""


def _extract_version_label(stem: str, title: str) -> str:
    for value in (stem, title):
        version_match = re.search(r"(v\d+)", value, re.IGNORECASE)
        if version_match:
            return version_match.group(1).lower()
    return stem


def _extract_baseline_hint(text: str) -> dict[str, str | int]:
    match = _BASELINE_RE.search(text)
    if not match:
        return {}
    return {
        "snapshot_id": str(match.group("snapshot")).strip(),
        "version": int(match.group("version")),
        "symbol": "BTCUSDT",
        "anchor_timeframe": "1h",
        "execution_timeframe": "5m",
    }


def _extract_global_sections(pre_stage_context: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_title = "context"
    buffer: list[str] = []
    for line in pre_stage_context.splitlines():
        heading = _SECTION_HEADING_RE.match(line)
        if heading:
            if buffer:
                sections[current_title] = "\n".join(buffer).strip()
                buffer = []
            current_title = heading.group(1).strip().lower()
            continue
        buffer.append(line)
    if buffer:
        sections[current_title] = "\n".join(buffer).strip()
    return {key: value for key, value in sections.items() if value}


def _extract_candidate_stages(text: str, matches: list[re.Match[str]]) -> list[RawPlanStageFragment]:
    stages: list[RawPlanStageFragment] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        stage_title = match.group(3).strip()
        stages.append(
            RawPlanStageFragment(
                stage_id=f"stage_{index + 1}",
                order_index=index + 1,
                heading=match.group(0).strip(),
                title=stage_title,
                objective_hint=_extract_stage_objective(block),
                actions_hint=_extract_stage_actions(block),
                success_criteria_hint=_extract_stage_completion(block),
                result_table_fields=_extract_result_table_fields(block),
                raw_markdown=block,
                section_titles=_extract_stage_section_titles(block),
            )
        )
    return stages


def _extract_stage_objective(block: str) -> str:
    objective = _extract_named_section(block, {"цель", "goal", "objective"})
    return objective.splitlines()[0].strip() if objective else ""


def _extract_stage_actions(block: str) -> list[str]:
    actions = _extract_named_section(block, {"что сделать", "what to do", "steps", "actions"})
    if not actions:
        return []
    results: list[str] = []
    for line in actions.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            results.append(cleaned)
    return results


def _extract_stage_completion(block: str) -> list[str]:
    completion = _extract_named_section(block, {"критерий завершения этапа", "completion criteria", "success criteria"})
    if not completion:
        return []
    return [line.strip("-* ").strip() for line in completion.splitlines() if line.strip("-* ").strip()]


def _extract_result_table_fields(block: str) -> list[str]:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        if "|" in line and index + 1 < len(lines) and set(lines[index + 1].replace("|", "").strip()) <= {"-", " "}:
            return [cell.strip() for cell in line.strip().strip("|").split("|") if cell.strip()]
    return []


def _extract_stage_section_titles(block: str) -> list[str]:
    return [match.group(1).strip() for match in _SECTION_HEADING_RE.finditer(block)]


def _extract_named_section(block: str, candidates: set[str]) -> str:
    lines = block.splitlines()
    active = False
    buffer: list[str] = []
    for line in lines:
        heading = _SECTION_HEADING_RE.match(line)
        if heading:
            title = heading.group(1).strip().lower()
            if active:
                break
            active = title in candidates
            continue
        if active:
            buffer.append(line)
    return "\n".join(buffer).strip()


def _parse_confidence(stages: list[RawPlanStageFragment]) -> float:
    if not stages:
        return 0.1
    score = 0.35
    score += min(0.3, len(stages) * 0.04)
    if all(stage.title for stage in stages):
        score += 0.15
    if sum(bool(stage.objective_hint) for stage in stages) >= max(1, len(stages) // 2):
        score += 0.1
    if sum(bool(stage.actions_hint) for stage in stages) >= max(1, len(stages) // 2):
        score += 0.1
    return min(0.95, score)

