"""Task dispatch — resolve symbolic refs, pick workers, dispatch tasks."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.models import OrchestratorEvent, TaskStatus
from app.plan_models import PlanStep, PlanTask, plan_task_to_task
from app.plan_symbolic_refs import resolve_stage_references_in_value

if TYPE_CHECKING:
    pass

logger = logging.getLogger("orchestrator.plan")


class TaskDispatchMixin:
    """Dispatches plan tasks to workers with MCP health gating and symbolic reference resolution."""

    def _dispatch_plan_tasks(self, plan_tasks: list[PlanTask]) -> None:
        """Dispatch PlanTasks to workers (respects MCP health).

        Two channels provide dependency context to workers:
        1. **Symbolic ref resolution** (_resolve_plan_task_steps): Replaces
           ``{{stage:N.field}}`` placeholders in step instructions/args with
           concrete values extracted from upstream task reports.  This is the
           *programmatic* channel — it injects IDs directly into tool_call args.
        2. **dependency_reports**: Full ``TaskReport`` objects passed through
           to the worker prompt.  This is the *informational* channel — it
           gives the worker complete context (results tables, verdicts, metrics)
           for human-like decision making.
        """
        if self._drain_mode:
            logger.info("Drain mode active — skipping dispatch of %d plan tasks", len(plan_tasks))
            return

        orch = self.orch
        reports_by_stage = self._reports_by_stage()

        for pt in plan_tasks:
            resolved_steps = self._resolve_plan_task_steps(pt, reports_by_stage)
            if resolved_steps is None:
                continue

            # MCP health gate
            instructions_text = " ".join(
                f"{step.instruction} {step.tool_name or ''} {step.args}"
                for step in resolved_steps
            )
            if self._is_mcp_instructions(instructions_text) and not self._mcp_healthy:
                skip_count = self._mcp_skip_counts.get(pt.stage_number, 0) + 1
                self._mcp_skip_counts[pt.stage_number] = skip_count
                if skip_count >= 3:
                    logger.error(
                        "Stage %d MCP-dependent task skipped %d times — "
                        "marking FAILED to unblock plan revision",
                        pt.stage_number, skip_count,
                    )
                    pt.status = TaskStatus.FAILED
                    pt.completed_at = datetime.now(timezone.utc).isoformat()
                    self._persist_current_plan()
                else:
                    logger.warning(
                        "Skipping MCP-dependent stage %d — MCP unhealthy "
                        "(skip %d/3)",
                        pt.stage_number, skip_count,
                    )
                continue

            # Per-stage retry limit — skip stages that exceeded max retries
            stage_retries = self._stage_retry_counts.get(pt.stage_number, 0)
            if stage_retries >= self._max_stage_retries:
                logger.warning(
                    "Skipping stage %d — exceeded max per-stage retries (%d)",
                    pt.stage_number, self._max_stage_retries,
                )
                pt.status = TaskStatus.FAILED
                continue

            worker_id = self._pick_worker()
            if not worker_id:
                logger.warning("No workers available for plan task stage %d", pt.stage_number)
                continue

            pt.assigned_worker_id = worker_id
            pt.status = TaskStatus.RUNNING
            pt.completed_at = None

            task = plan_task_to_task(pt)
            task.mark_running()
            self.state.add_task(task)

            self.state.plan_task_dispatch_map[str(pt.stage_number)] = task.task_id

            # Gather dependency reports for this task's depends_on stages
            dep_reports = [
                reports_by_stage[dep_stage]
                for dep_stage in pt.depends_on
                if dep_stage in reports_by_stage
            ]

            process_info = self.worker_service.start_plan_task(
                task=task,
                plan_version=pt.plan_version,
                stage_number=pt.stage_number,
                stage_name=pt.stage_name,
                theory=pt.theory,
                agent_instructions=[step.instruction for step in resolved_steps],
                steps=resolved_steps,
                results_table_columns=pt.results_table_columns,
                dependency_reports=dep_reports if dep_reports else None,
            )
            self.state.processes.append(process_info)

            # Progress tracking — worker spinner (green)
            from app.rich_handler import ProgressManager
            pm = ProgressManager._instance
            if pm and pm.is_active():
                pm.add_worker(
                    task.task_id, worker_id,
                    pid=process_info.pid,
                    description=f"Stage {pt.stage_number}: {pt.stage_name}",
                )

            orch._log_event(
                OrchestratorEvent.WORKER_LAUNCHED,
                f"[plan] stage={pt.stage_number} task={task.task_id} worker={worker_id}",
            )

        self.state.last_change_at = datetime.now(timezone.utc).isoformat()
        self.state.empty_cycles = 0
        self._persist_current_plan()

    @staticmethod
    def _is_mcp_instructions(text: str) -> bool:
        """Check if task instructions reference MCP tools."""
        keywords = [
            "backtest", "snapshot", "features_", "custom_features",
            "models_", "strategy", "dataset", "mcp",
            "cf_", "heatmap", "walk-forward",
            "diagnostics", "research_project", "research_record",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    def _reports_by_stage(self) -> dict[int, Any]:
        reports_by_stage: dict[int, Any] = {}
        if not self._current_plan or not self._plan_store:
            return reports_by_stage

        task_to_stage = {
            pt.task_id: pt.stage_number
            for pt in self._current_plan.tasks
        }
        for report in self._plan_store.load_reports_for_plan(self._current_plan.version):
            stage_number = task_to_stage.get(report.task_id)
            if stage_number is not None:
                reports_by_stage[stage_number] = report
        return reports_by_stage

    def _resolve_plan_task_steps(
        self,
        plan_task: PlanTask,
        reports_by_stage: dict[int, Any],
    ) -> list[PlanStep] | None:
        resolved_steps: list[PlanStep] = []
        for step in plan_task.normalized_steps():
            resolved_instruction, instruction_errors = resolve_stage_references_in_value(
                step.instruction,
                reports_by_stage,
            )
            resolved_args, arg_errors = resolve_stage_references_in_value(
                step.args,
                reports_by_stage,
            )
            resolved_notes, note_errors = resolve_stage_references_in_value(
                step.notes,
                reports_by_stage,
            )
            unresolved = instruction_errors + arg_errors + note_errors
            if unresolved:
                blocking_errors = []
                runtime_errors = []
                for err in unresolved:
                    if err.stage_number not in reports_by_stage:
                        blocking_errors.append(err)
                    else:
                        runtime_errors.append(err)

                if runtime_errors:
                    message = "; ".join(err.message for err in runtime_errors)
                    logger.error(
                        "Stage %d symbolic refs failed after dependencies resolved: %s",
                        plan_task.stage_number,
                        message,
                    )
                    plan_task.status = TaskStatus.FAILED
                    plan_task.completed_at = datetime.now(timezone.utc).isoformat()
                    # Write a minimal failure report so downstream stages that
                    # depend on this one via symbolic refs can see a report
                    # entry instead of being permanently deadlocked.
                    self._save_symbolic_ref_failure_report(plan_task, message)
                    self.memory_service.record_error(
                        self.state,
                        f"Plan stage {plan_task.stage_number} symbolic ref error: {message}",
                    )
                    self.state.total_errors += 1
                    self._persist_current_plan()
                    return None

                logger.info(
                    "Stage %d waiting on unresolved deps %s — symbolic refs: %s",
                    plan_task.stage_number,
                    plan_task.depends_on,
                    ", ".join(err.raw for err in blocking_errors),
                )
                return None

            resolved_steps.append(PlanStep(
                step_id=step.step_id,
                kind=step.kind,
                instruction=str(resolved_instruction),
                tool_name=step.tool_name,
                args=resolved_args if isinstance(resolved_args, dict) else {},
                binds=list(step.binds),
                decision_outputs=list(step.decision_outputs),
                notes=str(resolved_notes),
            ))

        return resolved_steps

    def _save_symbolic_ref_failure_report(
        self,
        plan_task: PlanTask,
        error_message: str,
    ) -> None:
        """Write a minimal error report so downstream stages see a report entry."""
        from app.plan_models import TaskReport
        report = TaskReport(
            task_id=plan_task.task_id,
            worker_id="",
            plan_version=plan_task.plan_version,
            status="error",
            error=f"symbolic ref resolution failed: {error_message}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        if self._plan_store:
            self._plan_store.save_report(report)

    def _pick_worker(self) -> str | None:
        active_workers = {t.assigned_worker_id for t in self.state.active_tasks()}
        idle = [w for w in self._worker_ids if w not in active_workers]
        if idle:
            worker_id = idle[self._next_worker_idx % len(idle)]
            self._next_worker_idx += 1
            return worker_id
        if self._worker_ids:
            worker_id = self._worker_ids[self._next_worker_idx % len(self._worker_ids)]
            self._next_worker_idx += 1
            return worker_id
        return None
