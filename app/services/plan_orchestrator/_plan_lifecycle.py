"""Plan lifecycle — create, repair, revise, and process plan data."""

from __future__ import annotations

import logging
import re
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

# Policy-locked defaults auto-injected into tool call args when missing.
POLICY_DEFAULTS: dict[tuple[str, str], dict[str, str]] = {
    ("backtests_plan", "plan"): {
        "symbol": "BTCUSDT",
        "anchor_timeframe": "1h",
        "execution_timeframe": "5m",
        "version": "1",
    },
    ("backtests_runs", "start"): {
        "symbol": "BTCUSDT",
        "anchor_timeframe": "1h",
        "execution_timeframe": "5m",
        "version": "1",
    },
    ("backtests_walkforward", "start"): {
        "symbol": "BTCUSDT",
        "anchor_timeframe": "1h",
        "execution_timeframe": "5m",
    },
    ("backtests_conditions", "run"): {
        "symbol": "BTCUSDT",
    },
}


def _inject_policy_defaults(plan: ResearchPlan) -> None:
    """Auto-fill POLICY-LOCKED args into tool_call steps when missing.

    Also fixes common arg-name mistakes (e.g. snapshot_id instead of
    source_snapshot_id for backtests_strategy clone) and auto-fills
    missing action parameters from DEFAULT_ACTION_BY_TOOL.
    """
    from app.planner_contract import ACTION_PARAM_BY_TOOL, DEFAULT_ACTION_BY_TOOL

    for task in plan.tasks:
        for step in task.steps:
            if step.kind != "tool_call" or not step.tool_name or not isinstance(step.args, dict):
                continue

            # Auto-fix common arg-name mistakes
            if step.tool_name == "backtests_strategy" and step.args.get("action") == "clone":
                if "snapshot_id" in step.args and "source_snapshot_id" not in step.args:
                    step.args["source_snapshot_id"] = step.args.pop("snapshot_id")

            # Resolve action parameter for this tool
            action_param = ACTION_PARAM_BY_TOOL.get(step.tool_name, "action")
            resolved_action = step.args.get(action_param)

            if resolved_action is None:
                # Auto-fill missing action from defaults
                default_action = DEFAULT_ACTION_BY_TOOL.get(step.tool_name)
                if default_action is not None:
                    step.args[action_param] = default_action
                    resolved_action = default_action
                    logger.debug(
                        "Auto-filled %s.%s='%s' from DEFAULT_ACTION_BY_TOOL",
                        step.tool_name, action_param, default_action,
                    )

            action = resolved_action
            if not action:
                continue
            defaults = POLICY_DEFAULTS.get((step.tool_name, str(action)))
            if defaults:
                for key, value in defaults.items():
                    if key not in step.args:
                        step.args[key] = value


def _topological_sort_stages(tasks: list[PlanTask]) -> list[int]:
    """Return stage numbers in valid execution order based on depends_on."""
    deps_map: dict[int, set[int]] = {t.stage_number: set(t.depends_on) for t in tasks}
    known_stages = set(deps_map)
    result: list[int] = []
    visited: set[int] = set()
    visiting: set[int] = set()

    def visit(stage: int) -> None:
        if stage in visited:
            return
        if stage in visiting:
            logger.warning(
                "Cycle detected in stage dependencies involving stage %d — skipping",
                stage,
            )
            return
        if stage not in known_stages:
            logger.warning(
                "Stage depends on unknown stage %d which does not exist in plan — ignoring",
                stage,
            )
            return
        visiting.add(stage)
        for dep in deps_map.get(stage, set()):
            visit(dep)
        visiting.discard(stage)
        visited.add(stage)
        result.append(stage)

    for s in sorted(deps_map):
        visit(s)
    return result


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
            validation_warnings=self._get_validation_warnings(),
            research_history=self._plan_store.load_all_reports_compact(
                current_plan_version=self.state.current_plan_version,
            ) if self._plan_store else None,
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
        compressed_reports_text = self._report_compressor.compress_reports(reports)
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
            validation_warnings=self._get_validation_warnings(),
            research_history=self._plan_store.load_all_reports_compact(
                current_plan_version=self._current_plan.version,
            ) if self._plan_store else None,
            compressed_reports=compressed_reports_text,
        )

    def _get_validation_warnings(self) -> list[dict] | None:
        """Get previous plan validation warnings as dicts for prompt inclusion."""
        errors = getattr(self.state, "current_plan_validation_errors", None)
        if not errors:
            return None
        from app.plan_validation import PlanValidationResult
        if isinstance(errors, PlanValidationResult):
            return errors.as_dicts()
        if isinstance(errors, list):
            return [
                e if isinstance(e, dict) else {
                    "stage_number": getattr(e, "stage_number", "?"),
                    "code": getattr(e, "code", "?"),
                    "message": getattr(e, "message", ""),
                }
                for e in errors
            ]
        return None

    def _process_plan_data(self, data: dict) -> None:
        """Process parsed plan data from the planner."""
        action = data.get("plan_action", "create")
        request_type = str(data.get("_request_type", self.state.current_plan_attempt_type or "create"))
        request_version = int(data.get("_request_version", self.state.current_plan_version + 1) or (self.state.current_plan_version + 1))
        attempt_number = int(data.get("_attempt_number", self.state.current_plan_attempt or 1) or 1)
        failure_class = str(data.get("_failure_class", "none") or "none")
        # Always use the orchestrator's request_version — the planner's
        # plan_version output is unreliable (the schema example hardcodes 1
        # and models frequently echo it regardless of actual version).
        version = request_version
        if data.get("_parse_failed"):
            version = request_version
        data["plan_version"] = version

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

        raw_order = data.get("tasks_to_dispatch", [])
        if raw_order:
            plan.execution_order = raw_order
        else:
            plan.execution_order = _topological_sort_stages(plan.tasks)

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

        # Auto-fill POLICY-LOCKED defaults before validation
        _inject_policy_defaults(plan)

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


        self._current_plan = plan
        self.state.current_plan_version = version
        self._stage_retry_counts.clear()
        self._mcp_reconnect_stage_counts.clear()
        self._mcp_skip_counts.clear()
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

        # Execution time prediction via LMStudio
        if self._lmstudio and self._lmstudio.is_available():
            self._predict_and_notify(plan)

    def _predict_and_notify(self, plan: ResearchPlan) -> None:
        """Predict execution time for a new plan and send notification."""
        try:
            from app.services.lmstudio_assistant import TaskHistoryEntry

            history: list[TaskHistoryEntry] = []
            for task in self.state.tasks:
                if task.status != TaskStatus.COMPLETED:
                    continue
                if not task.metadata.get("plan_mode"):
                    continue
                created = task.created_at
                updated = task.updated_at
                if not created or not updated:
                    continue
                try:
                    c = datetime.fromisoformat(created)
                    u = datetime.fromisoformat(updated)
                    duration = (u - c).total_seconds() / 60.0
                except (ValueError, TypeError):
                    duration = 0.0
                history.append(TaskHistoryEntry(
                    stage_number=task.metadata.get("stage_number", 0),
                    stage_name=task.metadata.get("stage_name", ""),
                    execution_minutes=max(duration, 0.0),
                ))

            prediction = self._lmstudio.predict_execution_time(history, plan)  # type: ignore[union-attr]
            if prediction:
                self.notification_service.send_execution_prediction(
                    prediction.raw_response, plan.version,
                )
                logger.info(
                    "Execution prediction sent for plan v%d (%.1f min estimated)",
                    plan.version, prediction.total_estimated_minutes,
                )
        except Exception as e:
            logger.warning("Execution prediction failed: %s", e)
