"""Result processing — convert TaskResults to reports, retry logic, baseline capture."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.models import (
    OrchestratorEvent,
    TaskStatus,
)
from app.plan_models import PlanTask

if TYPE_CHECKING:
    from app.models import TaskResult

logger = logging.getLogger("orchestrator.plan")


class ResultProcessingMixin:
    """Processes plan task results and updates plan baseline metrics."""

    # ---------------------------------------------------------------
    # Result processing + retry
    # ---------------------------------------------------------------

    def _process_plan_results(self, new_results: list[TaskResult]) -> None:
        """Convert TaskResults to TaskReports, apply retry logic for failures."""
        orch = self.orch

        for result in new_results:
            task = self.state.find_task(result.task_id)
            if not task:
                continue

            if not task.metadata.get("plan_mode"):
                # Non-plan result — already handled by _collect_results → _handle_task_result
                continue

            report = result.plan_report
            if report is None:
                logger.warning(
                    "Plan task %s missing structured plan_report; using degraded fallback from raw output",
                    result.task_id,
                )
                from app.result_parser import parse_task_report

                report = parse_task_report(
                    result.raw_output,
                    task_id=result.task_id,
                    worker_id=result.worker_id,
                    plan_version=task.metadata.get("plan_version", 0),
                )
            if self._plan_store:
                self._plan_store.save_report(report)

            # Update PlanTask in current plan
            if not self._current_plan:
                continue

            stage_num = task.metadata.get("stage_number", -1)
            pt = self._current_plan.get_task_by_stage(stage_num)
            if not pt:
                continue

            pt.results_table_rows = report.results_table
            pt.verdict = report.verdict
            pt.assigned_worker_id = task.assigned_worker_id
            pt.completed_at = datetime.now(timezone.utc).isoformat()

            if result.status == "success":
                pt.status = TaskStatus.COMPLETED
                self._maybe_update_plan_baseline(pt, report)
                # Integration validation for integration stages
                if pt.stage_name and "integration" in pt.stage_name.lower():
                    self._check_integration_result(pt, report)
                orch._log_event(
                    OrchestratorEvent.WORKER_COMPLETED,
                    f"task={task.task_id} stage={stage_num}",
                )
            elif result.status == "partial":
                pt.status = TaskStatus.COMPLETED
                self._maybe_update_plan_baseline(pt, report)
                logger.warning(
                    "Stage %d completed with partial status (verdict=%s)",
                    stage_num, pt.verdict,
                )
                # Scan partial results for MCP failure indicators
                if self._is_mcp_failure(report):
                    self._mcp_healthy = False
                    self.state.mcp_consecutive_failures += 1
                    logger.warning(
                        "MCP failure detected in partial result for stage %d — "
                        "marking MCP unhealthy (consecutive=%d)",
                        stage_num, self.state.mcp_consecutive_failures,
                    )
                # Integration validation for integration stages
                if pt.stage_name and "integration" in pt.stage_name.lower():
                    self._check_integration_result(pt, report)
                orch._log_event(
                    OrchestratorEvent.WORKER_COMPLETED,
                    f"task={task.task_id} stage={stage_num} (partial)",
                )
            else:
                # Error — retry logic
                max_attempts = task.max_attempts
                if task.attempts < max_attempts:
                    task.attempts += 1
                    # Track per-stage retry count
                    self._stage_retry_counts[stage_num] = (
                        self._stage_retry_counts.get(stage_num, 0) + 1
                    )
                    # Reset for re-dispatch
                    task.status = TaskStatus.PENDING
                    task.assigned_worker_id = None
                    pt.status = TaskStatus.PENDING
                    pt.assigned_worker_id = None
                    pt.completed_at = None

                    mcp_fail = self._is_mcp_failure(report)
                    if mcp_fail:
                        # Don't mark MCP unhealthy — the retry gets a fresh
                        # subprocess with a fresh MCP connection.
                        logger.info(
                            "MCP failure on stage %d, retrying with fresh "
                            "subprocess (attempt %d/%d)",
                            pt.stage_number, task.attempts, max_attempts,
                        )
                    else:
                        logger.info(
                            "Stage %d failed, retrying (attempt %d/%d)",
                            pt.stage_number, task.attempts, max_attempts,
                        )
                    self.memory_service.record_event(
                        self.state,
                        f"Retrying plan stage {pt.stage_number} "
                        f"(attempt {task.attempts}/{max_attempts})",
                    )
                else:
                    # Max attempts exhausted — permanent failure
                    pt.status = TaskStatus.FAILED
                    if self._is_mcp_failure(report):
                        self._mcp_healthy = False
                        self.state.mcp_consecutive_failures += 1
                        logger.warning(
                            "MCP failure on stage %d after %d retries — "
                            "marking MCP unhealthy (consecutive=%d)",
                            pt.stage_number, task.attempts,
                            self.state.mcp_consecutive_failures,
                        )
                    orch._log_event(
                        OrchestratorEvent.WORKER_FAILED,
                        f"task={task.task_id} stage={stage_num} "
                        f"permanently failed after {task.attempts} attempts",
                    )

            self._persist_current_plan()

    @staticmethod
    def _is_mcp_failure(report: Any) -> bool:
        """Detect MCP-specific failures from a TaskReport."""
        mcp_indicators = [
            "tool not found in registry",
            "mcp tools are not available",
            "mcp dev_space1",
            "mcp server not connected",
            "server may not be running",
        ]
        text = f"{getattr(report, 'error', '')} {getattr(report, 'raw_output', '')}".lower()
        return any(indicator in text for indicator in mcp_indicators)

    def _maybe_update_plan_baseline(self, plan_task: PlanTask, report: Any) -> None:
        """Capture the measured baseline from ETAP 0 so later plans use real metrics."""
        if not self._current_plan or plan_task.stage_number != 0 or report.status not in ("success", "partial"):
            return

        baseline_row = report.results_table[0] if report.results_table else {}
        baseline_run_id = None
        if isinstance(baseline_row, dict):
            baseline_run_id = baseline_row.get("run_id")

        if not baseline_run_id:
            for artifact in report.artifacts:
                if isinstance(artifact, str) and artifact.startswith("run_id:"):
                    baseline_run_id = artifact.split(":", 1)[1].strip()
                    break

        baseline_snapshot_ref = None
        if isinstance(baseline_row, dict) and baseline_row.get("snapshot_id"):
            snapshot_id = str(baseline_row["snapshot_id"])
            version = baseline_row.get("version")
            baseline_snapshot_ref = f"{snapshot_id}@{version}" if version else snapshot_id
        if not baseline_snapshot_ref:
            for artifact in report.artifacts:
                if isinstance(artifact, str) and artifact.startswith("snapshot:"):
                    baseline_snapshot_ref = artifact.split(":", 1)[1].strip()
                    break

        if baseline_run_id:
            self._current_plan.baseline_run_id = baseline_run_id
        if baseline_snapshot_ref:
            self._current_plan.baseline_snapshot_ref = baseline_snapshot_ref
        if report.key_metrics:
            self._current_plan.baseline_metrics = dict(report.key_metrics)

    def _check_integration_result(self, plan_task: PlanTask, report: Any) -> None:
        """Validate that an integration stage actually modified signal conditions."""
        from app.plan_validation import validate_integration_result

        warnings = validate_integration_result(report)
        if warnings:
            for w in warnings:
                logger.warning(
                    "Integration validation: stage %d — %s",
                    plan_task.stage_number, w,
                )
            if plan_task.verdict == "PROMOTE":
                plan_task.verdict = "WATCHLIST"
                logger.warning(
                    "Integration stage %d verdict downgraded PROMOTE → WATCHLIST "
                    "due to validation warnings",
                    plan_task.stage_number,
                )
