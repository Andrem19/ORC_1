"""
Sequence completion report builder with optional LLM narrative enrichment.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import OrchestratorConfig
    from app.execution_models import ExecutionPlan, ExecutionStateV2
    from app.plan_sources import CompiledPlanSource
    from app.services.notification_service import NotificationService

logger = logging.getLogger("orchestrator.sequence_reporter")


async def build_and_send_sequence_report(
    *,
    plan: ExecutionPlan,
    state: ExecutionStateV2,
    config: OrchestratorConfig,
    plan_source: CompiledPlanSource,
    notification_service: NotificationService,
) -> None:
    """Build a sequence report, enrich with LLM narrative, and send to Telegram."""
    from app.plan_sources import CompiledPlanSource
    from app.reporting.collector import ReportCollector
    from app.reporting.narrative import NarrativeGenerationError, ReportNarrativeService
    from app.runtime_factory import create_sequence_report_adapter

    if not isinstance(plan_source, CompiledPlanSource):
        return
    if not plan_source.is_sequence_complete(plan, state):
        return
    seq_id = plan.source_sequence_id or ""
    if not seq_id:
        return
    try:
        c = ReportCollector(config=config, run_id=config.current_run_id)
        prs = c.collect_plan_reports(state)
        srs = c.collect_sequence_reports(state=state, plan_reports=prs)
        sr = next((r for r in srs if r.sequence_id == seq_id), None)
        if sr is None:
            return
        sr_cfg = config.sequence_report
        try:
            adapter = create_sequence_report_adapter(config)
            narrative_svc = ReportNarrativeService(
                adapter=adapter, timeout_seconds=sr_cfg.timeout_seconds,
                retry_attempts=sr_cfg.retry_attempts, retry_backoff_seconds=sr_cfg.retry_backoff_seconds,
            )
            sections = await narrative_svc.generate_for_sequence(payload=asdict(sr))
            sr.narrative_status = "generated"
            sr.narrative_sections_ru = sections
            sr.executive_summary_ru = sections.executive_summary_ru
        except (NarrativeGenerationError, Exception) as exc:
            sr.narrative_status = "failed"
            logger.warning("LLM narrative failed for sequence %s: %s", seq_id, exc)
        notification_service.send_sequence_complete(sr)
        logger.info("Sent sequence completion report for %s", seq_id)
    except Exception as exc:
        logger.warning("Failed to build sequence report for %s: %s", seq_id, exc)
