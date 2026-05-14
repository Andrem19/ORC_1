"""
Offline conversion pipeline for raw markdown plans.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.raw_plan_compiler import build_failed_sequence, compile_semantic_raw_plan
from app.raw_plan_models import CompiledPlanSequence, RawPlanDocument, RawPlanStageFragment, SemanticRawPlan, SemanticStage
from app.raw_plan_parser import parse_raw_plan_file
from app.raw_plan_semantic_service import RawPlanSemanticError, RawPlanSemanticService
from app.execution_models import BaselineRef
from app.services.mcp_catalog.models import McpCatalogSnapshot

logger = logging.getLogger("orchestrator.converter")


class RawPlanConverterService:
    def __init__(
        self,
        *,
        semantic_service: RawPlanSemanticService | None,
        use_llm: bool,
        catalog_snapshot: McpCatalogSnapshot,
    ) -> None:
        self.semantic_service = semantic_service
        self.use_llm = use_llm
        self.catalog_snapshot = catalog_snapshot

    async def convert_path(self, path: str | Path) -> CompiledPlanSequence:
        document = parse_raw_plan_file(path)
        semantic_method = "llm" if self.use_llm else "rule_fallback"
        try:
            semantic_plan = await self._semantic_plan(document)
        except RawPlanSemanticError as exc:
            return build_failed_sequence(
                document,
                semantic_method=semantic_method,
                errors=[str(exc)],
                mcp_catalog_hash=self.catalog_snapshot.schema_hash,
            )
        return compile_semantic_raw_plan(
            document,
            semantic_plan,
            semantic_method=semantic_method,
            catalog_snapshot=self.catalog_snapshot,
        )

    async def _semantic_plan(self, document: RawPlanDocument) -> SemanticRawPlan:
        if self.use_llm and self.semantic_service is not None:
            return await self.semantic_service.extract(
                document,
                mcp_tool_catalog=self.catalog_snapshot.to_prompt_catalog(),
            )
        return _fallback_semantic_plan(document)


def _fallback_semantic_plan(document: RawPlanDocument) -> SemanticRawPlan:
    stages = document.candidate_stages or [_fallback_stage(document)]
    semantic_stages = [
        SemanticStage(
            stage_id=str(stage.stage_id),
            title=str(stage.title),
            objective=str(getattr(stage, "objective_hint", "") or stage.title),
            actions=list(getattr(stage, "actions_hint", []) or [f"Work through {stage.title} using the source markdown."]),
            success_criteria=list(getattr(stage, "success_criteria_hint", []) or ["Stage evidence collected"]),
            tool_hints=["analysis"],
            policy_tags=["rule_fallback"],
            raw_stage_ref=str(stage.stage_id),
        )
        for stage in stages
    ]
    baseline = document.baseline_ref_hint or {
        "snapshot_id": "active-signal-v1",
        "version": 1,
        "symbol": "BTCUSDT",
        "anchor_timeframe": "1h",
        "execution_timeframe": "5m",
    }
    return SemanticRawPlan(
        source_file=document.source_file,
        source_hash=document.source_hash,
        source_title=document.title,
        goal=document.global_sections.get("главная цель цикла", "") or document.title,
        baseline_ref=BaselineRef(
            snapshot_id=str(baseline.get("snapshot_id", "active-signal-v1")),
            version=baseline.get("version", 1),
            symbol=str(baseline.get("symbol", "BTCUSDT")),
            anchor_timeframe=str(baseline.get("anchor_timeframe", "1h")),
            execution_timeframe=str(baseline.get("execution_timeframe", "5m")),
        ),
        global_constraints=[
            "preserve source-stage ordering",
            "compiler fallback path used",
        ],
        stages=semantic_stages,
        warnings=list(document.parser_warnings) + ["semantic_extraction_fallback_used"],
        parse_confidence=document.parse_confidence,
    )


def _fallback_stage(document: RawPlanDocument) -> RawPlanStageFragment:
    return RawPlanStageFragment(
        stage_id="stage_1",
        order_index=1,
        heading=document.title,
        title=document.title,
        objective_hint="Convert the raw plan into an executable sequence.",
        raw_markdown=document.normalized_text,
    )
