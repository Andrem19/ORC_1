"""Plan lifecycle — create, repair, revise, and process plan data."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.models import TaskStatus
from app.plan_models import (
    AntiPattern,
    DecisionGate,
    PlanStep,
    PlanTask,
    ResearchPlan,
    decision_gate_from_dict,
)
from app.plan_validation import (
    PlanValidationError,
    PlanValidationResult,
    validate_plan,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("orchestrator.plan")


class PlanLifecycleMixin:
    """Handles plan creation, repair, revision, and processing of planner output."""

    def _create_plan(self) -> None:
        """Call planner to create a new research plan."""
        if not self._plan_store:
            return

        anti_patterns = self._plan_store.load_all_anti_patterns()
        cumulative = self._plan_store.load_cumulative_summary()

        prev_md = None
        if self.state.current_plan_version > 0:
            prev_md = self._plan_store.load_plan_markdown(self.state.current_plan_version)

        plan_version = self.state.current_plan_version + 1
        self.state.current_plan_attempt = 1
        self.state.current_plan_attempt_type = "create"
        self.state.current_plan_validation_errors = []
        logger.info("Creating plan (v%d) via planner", plan_version)
        # Progress tracking — planner spinner (yellow)
        from app.rich_handler import ProgressManager
        pm = ProgressManager._instance
        if pm and pm.is_active():
            pm.start_planner_wait(
                model=self.config.planner_adapter.model,
                action=f"Creating research plan v{plan_version} (attempt 1)",
            )
        self.planner_service.start_plan_creation(
            goal=self.config.goal,
            research_context=self._build_planner_context(),
            anti_patterns=anti_patterns if anti_patterns else None,
            cumulative_summary=cumulative,
            worker_ids=self._worker_ids,
            mcp_problem_summary=self._get_mcp_summary(),
            previous_plan_markdown=prev_md,
            plan_version=plan_version,
            attempt_number=1,
        )


    def _repair_plan(self) -> None:
        """Call planner to repair the latest rejected create-plan attempt."""
        if not self._plan_store:
            return

        repair_request = self._build_repair_request()
        if repair_request is None:
            logger.warning("Repair requested but no rejected plan payload is available; falling back to create")
            self._clear_invalid_plan_state()
            self._create_plan()
            return

        self.state.current_plan_attempt = repair_request.attempt_number
        self.state.current_plan_attempt_type = "repair"
        logger.info(
            "Repairing rejected plan v%d via planner (attempt %d/%d)",
            repair_request.plan_version,
            repair_request.attempt_number,
            self._max_plan_attempts,
        )
        from app.rich_handler import ProgressManager
        pm = ProgressManager._instance
        if pm and pm.is_active():
            pm.start_planner_wait(
                model=self.config.planner_adapter.model,
                action=(
                    f"Repairing plan v{repair_request.plan_version} "
                    f"(attempt {repair_request.attempt_number}/{self._max_plan_attempts})"
                ),
            )
        self.planner_service.start_plan_repair(
            repair_request=repair_request,
            research_context=self._build_planner_context(),
            worker_ids=self._worker_ids,
            mcp_problem_summary=self._get_mcp_summary(),
        )


    def _revise_plan(self) -> None:
        """Call planner to revise the current plan based on collected reports."""
        if not self._plan_store or not self._current_plan:
            return

        if len(self._current_plan.tasks) == 0:
            logger.warning(
                "Cannot revise empty plan v%d — clearing for re-creation",
                self._current_plan.version,
            )
            self._current_plan = None
            return

        reports = self._plan_store.load_reports_for_plan(self._current_plan.version)
        anti_patterns = self._plan_store.load_all_anti_patterns()

        logger.info(
            "Revising plan v%d → v%d (%d reports)",
            self._current_plan.version,
            self._current_plan.version + 1,
            len(reports),
        )
        # Progress tracking — planner spinner (yellow)
        from app.rich_handler import ProgressManager
        pm = ProgressManager._instance
        if pm and pm.is_active():
            pm.start_planner_wait(
                model=self.config.planner_adapter.model,
                action=f"Revising plan v{self._current_plan.version} → v{self._current_plan.version + 1}",
            )
        self.planner_service.start_plan_revision(
            goal=self.config.goal,
            current_plan=self._current_plan,
            reports=reports,
            research_context=self._build_planner_context(),
            anti_patterns=anti_patterns if anti_patterns else None,
            worker_ids=self._worker_ids,
            mcp_problem_summary=self._get_mcp_summary(),
        )


    def _process_plan_data(self, data: dict) -> None:
        """Process parsed plan data from the planner."""
        action = data.get("plan_action", "create")
        request_type = str(data.get("_request_type", self.state.current_plan_attempt_type or "create"))
        request_version = int(data.get("_request_version", self.state.current_plan_version + 1) or (self.state.current_plan_version + 1))
        attempt_number = int(data.get("_attempt_number", self.state.current_plan_attempt or 1) or 1)
        failure_class = str(data.get("_failure_class", "none") or "none")
        version = int(data.get("plan_version", request_version) or request_version)
        if data.get("_parse_failed"):
            version = request_version

        planner_run_artifact = self._persist_completed_planner_run()
        structured_payload = data.get("_structured_payload")
        if failure_class in {"transport_error", "parse_error"}:
            self._handle_planner_output_failure(
                plan_version=request_version,
                request_type=request_type,
                attempt_number=attempt_number,
                failure_class=failure_class,
                parsed_data=data,
                raw_output=self.planner_service.last_plan_raw_output,
                planner_run_artifact=planner_run_artifact,
                transport_errors=data.get("_transport_errors", []),
                structured_payload=structured_payload if isinstance(structured_payload, dict) else None,
            )
            return

        plan = ResearchPlan(
            schema_version=int(data.get("schema_version", 1) or 1),
            version=version,
            frozen_base=data.get("frozen_base", ""),
            baseline_run_id=data.get("baseline_run_id"),
            baseline_snapshot_ref=data.get("baseline_snapshot_ref"),
            baseline_metrics=data.get("baseline_metrics", {}) if isinstance(data.get("baseline_metrics"), dict) else {},
            goal=self.config.goal,
            principles=data.get("principles", []),
            cumulative_summary=data.get("cumulative_summary", ""),
            plan_markdown=data.get("plan_markdown", ""),
            status="active",
        )
        plan.planner_run_artifact = planner_run_artifact

        for t_data in data.get("tasks", []):
            gates = []
            for g in t_data.get("decision_gates", []):
                if isinstance(g, dict):
                    gates.append(decision_gate_from_dict(g))
                elif isinstance(g, DecisionGate):
                    gates.append(g)

            steps = []
            for idx, s_data in enumerate(t_data.get("steps", []), 1):
                if not isinstance(s_data, dict):
                    continue
                steps.append(PlanStep(
                    step_id=str(s_data.get("step_id", f"step_{idx}")),
                    kind=str(s_data.get("kind", "work")),
                    instruction=str(s_data.get("instruction", "")),
                    tool_name=s_data.get("tool_name"),
                    args=s_data.get("args", {}) if isinstance(s_data.get("args"), dict) else {},
                    binds=s_data.get("binds", []) if isinstance(s_data.get("binds"), list) else [],
                    decision_outputs=(
                        s_data.get("decision_outputs", [])
                        if isinstance(s_data.get("decision_outputs"), list) else []
                    ),
                    notes=str(s_data.get("notes", "")),
                ))

            pt = PlanTask(
                plan_version=version,
                stage_number=t_data.get("stage_number", 0),
                stage_name=t_data.get("stage_name", ""),
                theory=t_data.get("theory", ""),
                depends_on=t_data.get("depends_on", []),
                steps=steps,
                agent_instructions=t_data.get("agent_instructions", []),
                results_table_columns=t_data.get("results_table_columns", []),
                decision_gates=gates,
                verdict="PENDING",
            )
            plan.tasks.append(pt)

        for ap_data in data.get("anti_patterns_new", []):
            plan.anti_patterns.append(AntiPattern(
                category=ap_data.get("category", ""),
                description=ap_data.get("description", ""),
                evidence_count=ap_data.get("evidence_count", 0),
                evidence_summary=ap_data.get("evidence_summary", ""),
                verdict="REJECTED",
                source_plan_version=version,
            ))

        # Carry forward anti-patterns from previous plan
        if self._current_plan:
            existing_ids = {ap.pattern_id for ap in plan.anti_patterns}
            for ap in self._current_plan.anti_patterns:
                if ap.pattern_id not in existing_ids:
                    plan.anti_patterns.append(ap)
                    existing_ids.add(ap.pattern_id)
            if not plan.baseline_run_id:
                plan.baseline_run_id = self._current_plan.baseline_run_id
            if not plan.baseline_snapshot_ref:
                plan.baseline_snapshot_ref = self._current_plan.baseline_snapshot_ref
            if not plan.baseline_metrics:
                plan.baseline_metrics = dict(self._current_plan.baseline_metrics)

        plan.execution_order = data.get("tasks_to_dispatch", [])

        # --- Reject empty plans ---
        if len(plan.tasks) == 0:
            validation = PlanValidationResult(errors=[
                PlanValidationError(
                    stage_number=-1,
                    code="empty_plan",
                    message=(
                        "Planner returned 0 tasks"
                        + (
                            f" (parse_failed={data.get('_parse_failed', False)}; "
                            f"reason={str(data.get('reason', ''))[:200]})"
                        )
                    ),
                )
            ])
            self._handle_invalid_plan(
                plan_version=version,
                request_type=request_type,
                attempt_number=attempt_number,
                parsed_data=data,
                validation=validation,
                raw_output=self.planner_service.last_plan_raw_output,
                planner_run_artifact=planner_run_artifact,
                failure_class="invalid_content",
                structured_payload=structured_payload if isinstance(structured_payload, dict) else None,
            )
            return

        validation = self._validate_plan(plan)
        if not validation.is_acceptable:
            self._handle_invalid_plan(
                plan_version=version,
                request_type=request_type,
                attempt_number=attempt_number,
                parsed_data=data,
                validation=validation,
                raw_output=self.planner_service.last_plan_raw_output,
                planner_run_artifact=planner_run_artifact,
                failure_class="invalid_content",
                structured_payload=structured_payload if isinstance(structured_payload, dict) else None,
            )
            return
        elif validation.soft_errors:
            logger.warning(
                "Plan v%d accepted with %d soft warnings: %s",
                version,
                len(validation.soft_errors),
                validation.summary(),
            )
            self.state.current_plan_validation_errors = validation.soft_errors

        # Legacy fallback for older planner outputs
        if plan.schema_version < 2 and not plan.execution_order:
            plan.execution_order = [
                t.stage_number for t in sorted(plan.tasks, key=lambda t: t.stage_number)
            ]

        self._current_plan = plan
        self.state.current_plan_version = version
        self._stage_retry_counts.clear()
        self._persist_current_plan()
        self._clear_invalid_plan_state()

        self.memory_service.record_event(
            self.state,
            f"Plan v{version} {action}d: {len(plan.tasks)} tasks, "
            f"{len(plan.tasks)} stages with explicit dependencies",
        )
        self.notification_service.send_lifecycle(
            "plan_created" if action == "create" else "plan_revised",
            f"Plan v{version}: {len(plan.tasks)} stages",
        )
        self.state.last_change_at = datetime.now(timezone.utc).isoformat()
        self.state.empty_cycles = 0


