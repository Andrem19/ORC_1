"""
Post-run reporting builder for canonical JSON and Russian markdown summaries.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from app.adapters.base import BaseAdapter
from app.config import OrchestratorConfig
from app.execution_models import ExecutionStateV2
from app.reporting.collector import ReportCollector
from app.reporting.models import PlanBatchReport, RunSummaryReport, SequenceExecutionReport
from app.reporting.narrative import NarrativeGenerationError, ReportNarrativeService
from app.reporting.render_markdown import (
    render_plan_batch_report,
    render_run_summary_report,
    render_sequence_report,
)
from app.reporting.store import ReportStore

logger = logging.getLogger("orchestrator.reporting")


class PostRunReportBuilder:
    def __init__(
        self,
        *,
        config: OrchestratorConfig,
        planner_adapter: BaseAdapter | None,
        run_id: str,
        skip_llm: bool = False,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.collector = ReportCollector(config=config, run_id=run_id)
        self.store = ReportStore(plan_root=config.plan_dir, run_id=run_id)
        self.narrative = ReportNarrativeService(
            adapter=planner_adapter,
            timeout_seconds=config.planner_decision_timeout_seconds,
            retry_attempts=config.planner_decision_retry_attempts,
            retry_backoff_seconds=config.decision_retry_backoff_seconds,
            enabled=not skip_llm,
        )

    async def build(self, *, state: ExecutionStateV2) -> tuple[dict[str, Any], RunSummaryReport]:
        plan_reports = self.collector.collect_plan_reports(state)
        for report in plan_reports:
            await self._populate_narrative(report_kind="plan_batch", report=report)
            json_path = self.store.active_root / "plan_reports" / f"{report.plan_id}.json"
            md_path = self.store.active_root / "plan_reports" / f"{report.plan_id}.md"
            report.supporting_paths["report_json"] = str(json_path)
            report.supporting_paths["report_md"] = str(md_path)
            self.store.save_plan_report(
                plan_id=report.plan_id,
                payload=report,
                markdown=render_plan_batch_report(report),
            )

        sequence_reports = self.collector.collect_sequence_reports(state=state, plan_reports=plan_reports)
        for report in sequence_reports:
            await self._populate_narrative(report_kind="sequence", report=report)
            self.store.save_sequence_report(
                sequence_id=report.sequence_id,
                payload=report,
                markdown=render_sequence_report(report),
            )

        run_report = self.collector.collect_run_report(
            state=state,
            plan_reports=plan_reports,
            sequence_reports=sequence_reports,
        )
        run_report.sequence_report_paths = [
            str(self.store.active_root / "sequence_reports" / f"{item.sequence_id}.json")
            for item in sequence_reports
        ]
        run_report.plan_report_paths = [
            str(self.store.active_root / "plan_reports" / f"{item.plan_id}.json")
            for item in plan_reports
        ]
        await self._populate_narrative(report_kind="run", report=run_report)
        run_json, run_md = self.store.save_run_report(
            payload=run_report,
            markdown=render_run_summary_report(run_report),
        )
        return {
            "plan_reports": len(plan_reports),
            "sequence_reports": len(sequence_reports),
            "run_report_json": str(run_json),
            "run_report_md": str(run_md),
        }, run_report

    async def _populate_narrative(self, *, report_kind: str, report: PlanBatchReport | SequenceExecutionReport | RunSummaryReport) -> None:
        payload = asdict(report)
        try:
            sections = await self.narrative.generate(report_kind=report_kind, payload=payload)
        except NarrativeGenerationError as exc:
            report.narrative_status = "failed" if self.narrative.enabled else "skipped"
            if hasattr(report, "warnings") and isinstance(getattr(report, "warnings"), list):
                getattr(report, "warnings").append(f"narrative:{exc}")
            if self.narrative.enabled:
                logger.warning("Narrative generation degraded for %s: %s", report_kind, exc)
            return
        report.narrative_status = "generated"
        report.narrative_sections_ru = sections
        if isinstance(report, SequenceExecutionReport):
            report.executive_summary_ru = sections.executive_summary_ru
        elif isinstance(report, RunSummaryReport):
            report.executive_summary_ru = sections.executive_summary_ru
            report.operator_notes_ru = list(sections.operator_notes_ru)
