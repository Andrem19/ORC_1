"""
Direct execution orchestrator shell.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import Any

from app.adapters.base import BaseAdapter
from app.config import OrchestratorConfig
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import ExecutionPlan, ExecutionStateV2, PlanSlice
from app.execution_store import ExecutionStateStore
from app.models import OrchestratorEvent, OrchestratorState, StopReason
from app.runtime_console.controller import ConsoleRuntimeController
from app.services.mcp_catalog.models import McpCatalogDiff, McpCatalogSnapshot
from app.services.direct_execution.process_registry import ProcessRegistry
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
        self.process_registry = ProcessRegistry()
        self.state = OrchestratorState(goal=config.goal)
        self.execution_state = ExecutionStateV2(goal=config.goal)
        self.mcp_catalog_snapshot: McpCatalogSnapshot | None = None
        self.mcp_catalog_diff: McpCatalogDiff | None = None
        self.mcp_catalog_saved_paths: dict[str, str] = {}
        self._finish_completed = False
        self._research_context_text: str | None = None
        self._stop_requested = False
        self._stop_now_requested = False
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
                logger.warning("Archived legacy runtime artifacts for direct cutover: %s", archived)
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
        from app.services.direct_execution import DirectExecutionService

        svc = DirectExecutionService(orch=self)
        self._plan_service = svc
        try:
            return asyncio.run(svc.run())
        except (KeyboardInterrupt, asyncio.CancelledError):
            try:
                asyncio.run(self._finish_async(StopReason.GRACEFUL_STOP, "Interrupted by user (Ctrl+C)"))
            except Exception:
                pass
            return StopReason.GRACEFUL_STOP
        finally:
            self._plan_service = None

    def _finish(self, reason: StopReason, summary: str = "") -> None:
        asyncio.run(self._finish_async(reason, summary))

    async def _finish_async(self, reason: StopReason, summary: str = "") -> None:
        if self._finish_completed:
            return
        self._finish_completed = True
        self.persist_terminal_snapshot(reason=reason, summary=summary)
        # Notify operator immediately about non-graceful terminal states.
        if reason in (StopReason.RECOVERABLE_BLOCKED, StopReason.GOAL_IMPOSSIBLE, StopReason.MAX_ERRORS):
            failed_plans = [p for p in self.execution_state.plans if p.status in {"failed", "stopped"}]
            if failed_plans:
                last = failed_plans[-1]
                self.notification_service.send_lifecycle(
                    "run_stopped",
                    f"Run stopped: {reason.value}. Last failed: {last.plan_id} ({last.last_error or 'unknown'}). "
                    f"Total failed plans: {len(failed_plans)}. {summary[:80]}",
                )
        terminal_summary = self._terminal_summary(summary)
        report_summary = ""
        run_report = None
        if not self._stop_now_requested:
            report_summary, run_report = await self._build_postrun_reports()
            self.notification_service.flush()
            if run_report is not None:
                self.notification_service.send_run_complete(run_report)
            else:
                self.notification_service.send_lifecycle("finished", f"Reason: {reason.value}. {terminal_summary[:200]}")
        self._log_event(OrchestratorEvent.FINISHED, f"reason={reason.value} summary={terminal_summary[:100]}")
        logger.info("Orchestrator finished: %s. %s %s", reason.value, terminal_summary, report_summary)

    def request_stop(self) -> None:
        self.request_stop_now()

    def request_stop_now(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        self._stop_now_requested = True
        logger.info("Immediate stop requested via signal")
        self.persist_terminal_snapshot(reason=StopReason.GRACEFUL_STOP, summary="signal_forced_stop")
        plan_service = getattr(self, "_plan_service", None)
        if plan_service is not None and hasattr(plan_service, "request_shutdown"):
            plan_service.request_shutdown(immediate=True)
        self.process_registry.terminate_all(grace_seconds=0.2, force_after=0.5)

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
        self.notification_service.send_lifecycle("draining", "Drain mode: waiting for in-flight direct work to finish")

    def terminate_runtime_processes(self, *, force: bool = False) -> None:
        self.process_registry.terminate_all(
            grace_seconds=0.0 if force else 0.2,
            force_after=0.2 if force else 0.5,
        )

    def has_live_runtime_processes(self) -> bool:
        return self.process_registry.has_live_processes()

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

    def persist_terminal_snapshot(self, *, reason: StopReason, summary: str = "") -> None:
        if not hasattr(self, "state") or not hasattr(self, "execution_state"):
            return
        if not hasattr(self, "execution_store") or not hasattr(self, "artifact_store"):
            return
        self.state.status = "finished"
        self.state.stop_reason = reason
        self.execution_state.status = "finished"
        self.execution_state.stop_reason = reason.value
        self.execution_state.touch()
        self.state.updated_at = self.execution_state.updated_at
        self._terminalize_nonterminal_plans(reason=reason, summary=summary)
        self.execution_store.save(self.execution_state)
        self._log_event(OrchestratorEvent.STATE_SAVED)

    def _terminal_summary(self, fallback: str = "") -> str:
        current_plan_id = str(getattr(self.execution_state, "current_plan_id", "") or "").strip()
        if current_plan_id:
            for plan in self.execution_state.plans:
                if plan.plan_id == current_plan_id:
                    return f"plan={plan.plan_id} status={plan.status}"
        for plan in reversed(self.execution_state.plans):
            if str(plan.status or "").strip():
                return f"plan={plan.plan_id} status={plan.status}"
        return fallback

    def _terminalize_nonterminal_plans(self, *, reason: StopReason, summary: str) -> None:
        for plan in self.execution_state.plans:
            if plan.is_terminal:
                continue
            self._terminalize_plan(plan=plan, reason=reason, summary=summary)
            self.artifact_store.save_plan(plan)

    def _terminalize_plan(self, *, plan: ExecutionPlan, reason: StopReason, summary: str) -> None:
        plan.status = "stopped" if reason == StopReason.GRACEFUL_STOP else "failed"
        if reason == StopReason.GRACEFUL_STOP:
            plan.last_error = "user_stop"
        else:
            plan.last_error = f"terminalized:{reason.value}" if hasattr(reason, "value") else f"terminalized:{reason}"
        for slice_obj in plan.slices:
            self._terminalize_running_slice(slice_obj=slice_obj, reason=reason, summary=summary)
        plan.touch()

    @staticmethod
    def _terminalize_running_slice(*, slice_obj: PlanSlice, reason: StopReason, summary: str) -> None:
        if slice_obj.status != "running":
            return
        if reason == StopReason.GRACEFUL_STOP:
            slice_obj.status = "checkpointed"
            slice_obj.last_checkpoint_status = "blocked"
            slice_obj.last_checkpoint_summary = (
                slice_obj.last_checkpoint_summary
                or slice_obj.last_summary
                or "Interrupted by operator stop before the slice finished."
            )
            if not slice_obj.last_summary:
                slice_obj.last_summary = slice_obj.last_checkpoint_summary
            if not slice_obj.last_error:
                slice_obj.last_error = "graceful_stop"
        else:
            slice_obj.status = "failed"
            slice_obj.last_error = slice_obj.last_error or reason.value
            if not slice_obj.last_summary:
                slice_obj.last_summary = summary or reason.value
            slice_obj.last_checkpoint_status = ""
        slice_obj.active_operation_ref = ""
        slice_obj.active_operation_status = ""
        slice_obj.active_operation_arguments = {}
