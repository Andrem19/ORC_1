from __future__ import annotations

import asyncio
from pathlib import Path

from app.execution_models import BaselineRef
from app.raw_plan_converter_service import RawPlanConverterService
from app.raw_plan_models import SemanticRawPlan, SemanticStage
from app.raw_plan_semantic_service import RawPlanSemanticError


class _SemanticService:
    def __init__(self, plan: SemanticRawPlan | None = None, *, error: str = "") -> None:
        self.plan = plan
        self.error = error
        self.calls = 0

    async def extract(self, document, **kwargs) -> SemanticRawPlan:
        del document
        self.calls += 1
        if self.error:
            raise RawPlanSemanticError(self.error)
        assert self.plan is not None
        return self.plan


def _raw_plan(tmp_path: Path) -> Path:
    path = tmp_path / "raw_plans" / "plan_v1.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Plan v1\n\n## Главная цель цикла\nПроверить funding route.\n\n# ЭТАП 1. Проверка\n## Цель\nПроверить данные\n",
        encoding="utf-8",
    )
    return path


def _semantic_plan(source_file: str, source_hash: str) -> SemanticRawPlan:
    return SemanticRawPlan(
        source_file=source_file,
        source_hash=source_hash,
        source_title="Plan v1",
        goal="Validate funding route",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        stages=[
            SemanticStage(
                stage_id="stage_1",
                title="Check data",
                objective="Verify funding data exists",
                actions=["Inspect funding catalog"],
                success_criteria=["Catalog rows observed"],
                tool_hints=["events"],
            )
        ],
    )


def test_converter_service_compiles_with_llm_semantic_service(tmp_path) -> None:
    raw_file = _raw_plan(tmp_path)
    document_hash = ""
    from app.raw_plan_parser import parse_raw_plan_file

    document = parse_raw_plan_file(raw_file)
    document_hash = document.source_hash
    service = RawPlanConverterService(
        semantic_service=_SemanticService(_semantic_plan(str(raw_file), document_hash)),
        use_llm=True,
    )

    sequence = asyncio.run(service.convert_path(raw_file))

    assert sequence.report.compile_status == "compiled"
    assert sequence.report.compiled_plan_count == 1
    tools = sequence.plans[0].slices[0].allowed_tools
    assert "events" in tools
    assert "research_record" in tools


def test_converter_service_returns_failed_sequence_on_semantic_error(tmp_path) -> None:
    raw_file = _raw_plan(tmp_path)
    service = RawPlanConverterService(
        semantic_service=_SemanticService(error="semantic_raw_plan_missing_fields:goal"),
        use_llm=True,
    )

    sequence = asyncio.run(service.convert_path(raw_file))

    assert sequence.report.compile_status == "failed"
    assert "semantic_raw_plan_missing_fields:goal" in sequence.report.errors
