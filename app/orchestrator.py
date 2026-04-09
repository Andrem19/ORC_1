"""
Broker-only orchestrator shell.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import Any

from app.adapters.base import BaseAdapter
from app.config import OrchestratorConfig
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import ExecutionStateV2
from app.execution_store import ExecutionStateStore
from app.models import OrchestratorEvent, OrchestratorState, StopReason
from app.runtime_console.controller import ConsoleRuntimeController
from app.services.notification_service import NotificationService

logger = logging.getLogger("orchestrator")


class Orchestrator:
    _DEBUG_EVENTS = frozenset({OrchestratorEvent.STATE_SAVED})

    def __init__(
        self,
        config: OrchestratorConfig,
        planner_adapter: BaseAdapter,
        worker_adapter: BaseAdapter,
        notification_service: NotificationService | None = None,
        console_controller: ConsoleRuntimeController | None = None,
    ) -> None:
        self.config = config
        self.execution_store = ExecutionStateStore(
            config.execution_state_path,
            run_id=getattr(config, "current_run_id", ""),
        )
        self.artifact_store = ExecutionArtifactStore(
            getattr(config, "plan_dir", "plans"),
            run_id=getattr(config, "current_run_id", ""),
        )
        self.planner_adapter = planner_adapter
        self.worker_adapter = worker_adapter
        self.notification_service = notification_service or NotificationService()
        self.console_controller = console_controller
        self.state = OrchestratorState(goal=config.goal)
        self.execution_state = ExecutionStateV2(goal=config.goal)
        self._finish_completed = False
        self._research_context_text: str | None = None
        self._stop_requested = False
        self._drain_mode = False
        self._drain_started_at: float | None = None
        self._plan_service: Any = None
        self._log_event(OrchestratorEvent.STARTED)

    def _log_event(self, event: OrchestratorEvent, detail: str = "") -> None:
        msg = f"[{event.value}] {detail}" if detail else f"[{event.value}]"
        if event in self._DEBUG_EVENTS:
            logger.debug(msg)
        else:
            logger.info(msg)

    def load_research_context(self) -> None:
        if not self.config.research_config:
            return
        try:
            from app.research_context import format_research_context_for_planner, load_research_context

            ctx = load_research_context(self.config.state_dir)
            if ctx:
                summaries = [result.summary for result in self.state.results if result.status == "success" and result.summary][-30:]
                self._research_context_text = format_research_context_for_planner(ctx, completed_summaries=summaries)
                logger.info("Research context loaded (%d chars)", len(self._research_context_text))
        except Exception as exc:
            logger.error("Failed to load research context: %s", exc)

    def load_state(self) -> bool:
        saved = self.execution_store.load()
        if saved is None:
            archived = self.execution_store.archive_legacy_runtime(
                legacy_state_path=self.config.state_path,
                legacy_plan_dir=self.config.plan_dir,
            )
            if archived is not None:
                self.notification_service.send_lifecycle("cutover", f"Archived legacy runtime state to {archived}")
                logger.warning("Archived legacy runtime artifacts for broker cutover: %s", archived)
            return False
        self.execution_state = saved
        self.state.status = saved.status
        self.state.updated_at = saved.updated_at
        if saved.stop_reason:
            try:
                self.state.stop_reason = StopReason(saved.stop_reason)
            except ValueError:
                self.state.stop_reason = None
        self._log_event(OrchestratorEvent.STATE_RESTORED, f"plans={len(saved.plans)}")
        return True

    def save_state(self) -> None:
        self.execution_state.status = self.state.status
        self.execution_store.save(self.execution_state)
        self._log_event(OrchestratorEvent.STATE_SAVED)

    def run(self) -> StopReason:
        from app.services.brokered_execution import BrokeredExecutionService

        svc = BrokeredExecutionService(orch=self)
        self._plan_service = svc
        try:
            return asyncio.run(svc.run())
        except KeyboardInterrupt:
            asyncio.run(self._finish_async(StopReason.GRACEFUL_STOP, "Interrupted by user (Ctrl+C)"))
            return StopReason.GRACEFUL_STOP
        finally:
            self._plan_service = None

    def _finish(self, reason: StopReason, summary: str = "") -> None:
        asyncio.run(self._finish_async(reason, summary))

    async def _finish_async(self, reason: StopReason, summary: str = "") -> None:
        if self._finish_completed:
            return
        self._finish_completed = True
        self.state.status = "finished"
        self.state.stop_reason = reason
        self.execution_state.status = "finished"
        self.execution_state.stop_reason = reason.value
        await asyncio.to_thread(self.execution_store.save, self.execution_state)
        report_summary, run_report = await self._build_postrun_reports()
        self.notification_service.flush()
        if run_report is not None:
            self.notification_service.send_run_complete(run_report)
        else:
            self.notification_service.send_lifecycle("finished", f"Reason: {reason.value}. {summary[:200]}")
        self._log_event(OrchestratorEvent.FINISHED, f"reason={reason.value} summary={summary[:100]}")
        logger.info("Orchestrator finished: %s. %s %s", reason.value, summary, report_summary)

    def request_stop(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        logger.info("Stop requested via signal")

    def request_drain(self) -> None:
        if self._drain_mode:
            return
        self._drain_mode = True
        self._drain_started_at = _time.monotonic()
        logger.info("Drain mode requested")
        console_controller = getattr(self, "console_controller", None)
        if console_controller is not None:
            console_controller.on_drain_requested()
        self.notification_service.flush()
        self.notification_service.send_lifecycle("draining", "Drain mode: waiting for in-flight broker work to finish")

    async def _build_postrun_reports(self):
        """Build post-run reports. Returns (summary_str, RunSummaryReport | None)."""
        try:
            from app.reporting import PostRunReportBuilder

            builder = PostRunReportBuilder(
                config=self.config,
                planner_adapter=self.planner_adapter,
                run_id=getattr(self.config, "current_run_id", "") or "",
            )
            result, run_report = await builder.build(state=self.execution_state)
        except Exception as exc:
            logger.warning("Post-run reporting degraded: %s", exc)
            return f"reporting_degraded:{exc}", None
        summary = f"plan_reports={result.get('plan_reports', 0)} sequence_reports={result.get('sequence_reports', 0)}"
        return summary, run_report
