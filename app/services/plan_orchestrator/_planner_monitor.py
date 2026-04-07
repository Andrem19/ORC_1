"""Planner monitoring — watchdog, timeout, error handling, repair requests."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.models import StopReason
from app.plan_validation import (
    PlanRepairRequest,
    PlanValidationError,
    PlanValidationResult,
    validate_plan,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("orchestrator.plan")


class PlannerMonitorMixin:
    """Monitors planner subprocess health and handles output failures / invalid plans."""

    # ---------------------------------------------------------------
    # Planner runtime state sync
    # ---------------------------------------------------------------

    def _sync_planner_runtime_state(self, clear: bool = False) -> None:
        snapshot = None if clear else self.planner_service.plan_runtime_snapshot()
        if snapshot is None:
            self.state.planner_started_at = None
            self.state.planner_first_output_at = None
            self.state.planner_last_output_at = None
            self.state.planner_output_bytes = 0
            self.state.planner_stderr_bytes = 0
            return

        self.state.planner_started_at = snapshot.started_at_iso
        self.state.planner_first_output_at = snapshot.first_output_at_iso
        self.state.planner_last_output_at = snapshot.last_output_at_iso
        self.state.planner_output_bytes = snapshot.output_bytes
        self.state.planner_stderr_bytes = snapshot.stderr_bytes

    # ---------------------------------------------------------------
    # Planner watchdog
    # ---------------------------------------------------------------

    def _check_planner_watchdog(self) -> None:
        snapshot = self.planner_service.plan_runtime_snapshot()
        if snapshot is None:
            return

        adapter_cfg = getattr(self.config, "planner_adapter", None)
        no_first_byte_seconds = max(30, int(getattr(adapter_cfg, "no_first_byte_seconds", 180) or 180))
        soft_stall_seconds = max(no_first_byte_seconds, int(getattr(adapter_cfg, "soft_timeout_seconds", 300) or 300))
        hard_stall_seconds = max(soft_stall_seconds, int(getattr(adapter_cfg, "hard_timeout_seconds", 900) or 900))

        elapsed = snapshot.elapsed_seconds
        if not snapshot.has_first_byte and elapsed >= no_first_byte_seconds and not snapshot.no_first_byte_warning_sent:
            snapshot.no_first_byte_warning_sent = True
            logger.warning(
                "planner_silent: type=%s version=%d attempt=%d elapsed=%.0fs prompt=%d chars stdout=0 stderr=%d",
                snapshot.request_type,
                snapshot.request_version,
                snapshot.attempt_number,
                elapsed,
                snapshot.prompt_length,
                snapshot.stderr_bytes,
            )

        last_output_age = None
        if snapshot.last_output_at_monotonic is not None:
            last_output_age = max(0.0, time.monotonic() - snapshot.last_output_at_monotonic)

        if snapshot.has_any_output and elapsed >= soft_stall_seconds and (last_output_age is None or last_output_age < soft_stall_seconds):
            if not snapshot.slow_active_warning_sent:
                snapshot.slow_active_warning_sent = True
                stdout_state = self._planner_output_state(snapshot)
                logger.warning(
                    "planner_slow_active: type=%s version=%d attempt=%d elapsed=%.0fs state=%s stdout=%d stderr=%d prompt=%d structured=%dB delta=%dB",
                    snapshot.request_type,
                    snapshot.request_version,
                    snapshot.attempt_number,
                    elapsed,
                    stdout_state,
                    snapshot.output_bytes,
                    snapshot.stderr_bytes,
                    snapshot.prompt_length,
                    snapshot.structured_payload_bytes,
                    snapshot.structured_delta_bytes,
                )
                if not snapshot.slow_notification_sent:
                    snapshot.slow_notification_sent = True
                    self.notification_service.send_error(
                        (
                            f"Planner slow_active: {snapshot.request_type} v{snapshot.request_version} "
                            f"attempt {snapshot.attempt_number}, elapsed {elapsed:.0f}s, "
                            f"stdout={snapshot.output_bytes}B, stderr={snapshot.stderr_bytes}B, "
                            f"prompt={snapshot.prompt_length} chars"
                        ),
                        context="plan_mode",
                    )

        if last_output_age is not None and last_output_age >= soft_stall_seconds and not snapshot.stalled_warning_sent:
            snapshot.stalled_warning_sent = True
            stdout_state = self._planner_output_state(snapshot)
            logger.warning(
                "planner_stalled: type=%s version=%d attempt=%d elapsed=%.0fs last_output_age=%.0fs state=%s stdout=%d stderr=%d prompt=%d structured=%dB delta=%dB",
                snapshot.request_type,
                snapshot.request_version,
                snapshot.attempt_number,
                elapsed,
                last_output_age,
                stdout_state,
                snapshot.output_bytes,
                snapshot.stderr_bytes,
                snapshot.prompt_length,
                snapshot.structured_payload_bytes,
                snapshot.structured_delta_bytes,
            )
            if not snapshot.stalled_notification_sent:
                snapshot.stalled_notification_sent = True
                self.notification_service.send_error(
                    (
                        f"Planner stalled: {snapshot.request_type} v{snapshot.request_version} "
                        f"attempt {snapshot.attempt_number}, elapsed {elapsed:.0f}s, "
                        f"last_output={last_output_age:.0f}s ago, stdout={snapshot.output_bytes}B, "
                        f"stderr={snapshot.stderr_bytes}B"
                    ),
                    context="plan_mode",
                )

        if elapsed >= hard_stall_seconds:
            self._handle_planner_timeout(snapshot)

    @staticmethod
    def _planner_output_state(snapshot: Any) -> str:
        if getattr(snapshot, "structured_payload_source", "none") in {"tool_use_input", "input_json_delta"} or getattr(snapshot, "structured_delta_bytes", 0) > 0:
            return "structured_output_stream_active"
        if getattr(snapshot, "rendered_output", ""):
            return "text_stream_active"
        if getattr(snapshot, "output_bytes", 0) > 0:
            return "raw_stream_only"
        if getattr(snapshot, "stderr_bytes", 0) > 0:
            return "stderr_only"
        return "no_output"

    # ---------------------------------------------------------------
    # Planner timeout
    # ---------------------------------------------------------------

    def _handle_planner_timeout(self, snapshot: Any) -> None:
        final_snapshot = self.planner_service.terminate_plan_run("hard_timeout")
        if final_snapshot is None:
            return

        self._sync_planner_runtime_state(clear=True)
        self.state.planner_timeout_count += 1
        artifact_path = self._persist_planner_run(final_snapshot)
        summary = (
            f"Planner timeout: {final_snapshot.request_type} v{final_snapshot.request_version} "
            f"attempt {final_snapshot.attempt_number}, elapsed {final_snapshot.elapsed_seconds:.0f}s, "
            f"stdout={final_snapshot.output_bytes}B, stderr={final_snapshot.stderr_bytes}B"
        )

        if final_snapshot.timeout_retry_count < 1 and self.planner_service.restart_plan_request():
            logger.warning("%s — retrying once", summary)
            self.notification_service.send_error(
                f"{summary}. Retrying once. Artifact: {artifact_path}",
                context="plan_mode",
            )
            self.state.empty_cycles = 0
            self._sync_planner_runtime_state()
            return

        logger.error("%s — stopping orchestrator", summary)
        self.notification_service.send_error(
            f"{summary}. Stopping. Artifact: {artifact_path}",
            context="plan_mode",
        )
        self._terminal_stop_reason = StopReason.PLANNER_TIMEOUT
        self._terminal_stop_summary = summary

    def _persist_planner_run(self, snapshot: Any) -> str | None:
        if not self._plan_store:
            return None
        path = self._plan_store.save_planner_run(
            request_type=snapshot.request_type,
            request_version=snapshot.request_version,
            attempt_number=snapshot.attempt_number,
            execution_seq=getattr(snapshot, "execution_seq", 1),
            payload=snapshot.to_dict(),
        )
        return str(path)

    # ---------------------------------------------------------------
    # Invalid plan handling
    # ---------------------------------------------------------------

    def _handle_invalid_plan(
        self,
        plan_version: int,
        request_type: str,
        attempt_number: int,
        parsed_data: dict[str, Any],
        validation: PlanValidationResult,
        raw_output: str,
        failure_class: str = "invalid_content",
        structured_payload: dict[str, Any] | None = None,
        planner_run_artifact: str | None = None,
    ) -> None:
        """Persist rejection details and either trigger repair or stop."""
        attempt_number = max(1, attempt_number)
        attempt_type = request_type or self.state.current_plan_attempt_type or "create"
        first_error = validation.errors[0] if validation.errors else None
        summary = validation.summary()

        logger.warning(
            "Rejecting plan v%d attempt=%d type=%s: %d validation errors (%s)",
            plan_version,
            attempt_number,
            attempt_type,
            len(validation.errors),
            summary,
        )

        artifact_path = None
        if self._plan_store:
            artifact_path = self._plan_store.save_rejected_plan_attempt(
                plan_version=plan_version,
                attempt_number=attempt_number,
                attempt_type=attempt_type,
                execution_seq=int(parsed_data.get("_execution_seq", 1) or 1),
                raw_output=raw_output,
                parsed_data=parsed_data,
                validation_errors=validation.as_dicts(),
                failure_class=failure_class,
                request_type=request_type,
                structured_payload=structured_payload,
                failure_detail=summary,
                planner_run_artifact=planner_run_artifact,
            )

        self.state.current_plan_attempt = attempt_number
        self.state.current_plan_attempt_type = attempt_type
        self.state.current_plan_validation_errors = validation.as_dicts()
        self.state.last_rejected_plan_version = plan_version
        self.state.last_rejected_plan_attempt_at = datetime.now(timezone.utc).isoformat()
        self.state.last_rejected_plan_artifact = str(artifact_path) if artifact_path else None
        self.state.empty_cycles = 0
        self.state.last_change_at = datetime.now(timezone.utc).isoformat()

        if first_error:
            memory_text = (
                f"Plan v{plan_version} repair needed: {len(validation.errors)} errors, "
                f"stage {first_error.stage_number} {first_error.code}: {first_error.message}"
            )
        else:
            memory_text = f"Plan v{plan_version} repair needed: validation failed"
        self.memory_service.record_event(self.state, memory_text)

        # Convergence check: stop repair if errors did not decrease
        current_error_count = len(validation.errors)
        if attempt_number > 1 and self._last_repair_error_count > 0:
            if current_error_count >= self._last_repair_error_count:
                reason_word = "diverged" if current_error_count > self._last_repair_error_count else "stalled"
                logger.warning(
                    "Repair %s: %d errors → %d errors. Stopping repair loop.",
                    reason_word,
                    self._last_repair_error_count,
                    current_error_count,
                )
                final_message = (
                    f"Plan v{plan_version} repair {reason_word}: "
                    f"{self._last_repair_error_count} → {current_error_count} errors. {summary}"
                )
                self.notification_service.send_error(final_message, context="plan_mode")
                self._terminal_stop_reason = StopReason.INVALID_PLAN_LOOP
                self._terminal_stop_summary = final_message
                self._last_repair_error_count = current_error_count
                return
        self._last_repair_error_count = current_error_count

        if attempt_number >= self._max_plan_attempts:
            final_message = (
                f"Planner produced invalid plan v{plan_version} "
                f"{attempt_number} times: {summary}"
            )
            self.notification_service.send_error(final_message, context="plan_mode")
            self._terminal_stop_reason = StopReason.INVALID_PLAN_LOOP
            self._terminal_stop_summary = final_message
            return

        next_attempt = attempt_number + 1
        self.notification_service.send_error(
            (
                f"Plan v{plan_version} rejected on attempt {attempt_number}. "
                f"Scheduling repair attempt {next_attempt}/{self._max_plan_attempts}. "
                f"{summary}"
            ),
            context="plan_mode",
        )
        self.state.current_plan_attempt = next_attempt
        self.state.current_plan_attempt_type = "repair"
        self._repair_plan()

    # ---------------------------------------------------------------
    # Planner output failure (transport / parse errors)
    # ---------------------------------------------------------------

    def _handle_planner_output_failure(
        self,
        *,
        plan_version: int,
        request_type: str,
        attempt_number: int,
        failure_class: str,
        parsed_data: dict[str, Any],
        raw_output: str,
        planner_run_artifact: str | None,
        transport_errors: list[Any] | None = None,
        structured_payload: dict[str, Any] | None = None,
    ) -> None:
        """Handle transport/parse failures.

        Parse errors are routed through the repair loop (up to _max_plan_attempts)
        so the planner gets feedback about what went wrong. Transport errors get
        one simple retry with the same prompt.
        """
        attempt_number = max(1, attempt_number)
        request_type = request_type or self.state.current_plan_attempt_type or "create"
        transport_errors = [str(err) for err in (transport_errors or []) if str(err).strip()]
        summary = "; ".join(transport_errors[:2]) or str(parsed_data.get("reason", "planner output could not be recovered"))

        logger.warning(
            "Planner %s failure on %s v%d attempt=%d: %s",
            failure_class,
            request_type,
            plan_version,
            attempt_number,
            summary,
        )

        artifact_path = None
        if self._plan_store:
            # For parse errors, add a synthetic validation error so the repair
            # prompt tells the planner what went wrong.
            validation_errors_for_artifact: list[dict[str, Any]] = []
            if failure_class == "parse_error":
                validation_errors_for_artifact = [{
                    "stage_number": -1,
                    "code": "json_parse_error",
                    "message": (
                        "The plan JSON could not be parsed. Common causes: "
                        "trailing commas, unquoted keys, single quotes instead "
                        "of double quotes, or extra text outside the JSON. "
                        f"Parser error: {summary}"
                    ),
                    "offending_text": raw_output[:500] if raw_output else "",
                }]
            artifact_path = self._plan_store.save_rejected_plan_attempt(
                plan_version=plan_version,
                attempt_number=attempt_number,
                attempt_type=request_type,
                execution_seq=int(parsed_data.get("_execution_seq", 1) or 1),
                raw_output=raw_output,
                parsed_data=parsed_data,
                validation_errors=validation_errors_for_artifact,
                failure_class=failure_class,
                request_type=request_type,
                structured_payload=structured_payload,
                failure_detail=summary,
                planner_run_artifact=planner_run_artifact,
            )

        self.state.current_plan_attempt = attempt_number
        self.state.current_plan_attempt_type = request_type
        self.state.current_plan_validation_errors = []
        self.state.last_rejected_plan_version = plan_version
        self.state.last_rejected_plan_attempt_at = datetime.now(timezone.utc).isoformat()
        self.state.last_rejected_plan_artifact = str(artifact_path) if artifact_path else None
        self.state.empty_cycles = 0
        self.state.last_change_at = datetime.now(timezone.utc).isoformat()

        self.memory_service.record_event(
            self.state,
            f"Planner {failure_class} on {request_type} v{plan_version} attempt {attempt_number}: {summary}",
        )

        # Parse errors: route through repair loop (planner gets feedback)
        next_attempt = attempt_number + 1
        if failure_class == "parse_error" and next_attempt <= self._max_plan_attempts:
            logger.info(
                "Routing parse_error to repair loop (attempt %d/%d)",
                next_attempt, self._max_plan_attempts,
            )
            self.state.current_plan_attempt = next_attempt
            self.state.current_plan_attempt_type = "repair"
            self._repair_plan()
            self.notification_service.send_error(
                (
                    f"Planner parse error on v{plan_version} attempt {attempt_number}. "
                    f"Launching repair attempt {next_attempt}."
                ),
                context="plan_mode",
            )
            self._sync_planner_runtime_state()
            return

        # Transport errors: one simple retry with the same prompt
        transport_retry_count = getattr(self.planner_service, "plan_transport_retry_count", 0)
        if transport_retry_count < 1 and self.planner_service.restart_plan_request(reason="transport_error"):
            self.notification_service.send_error(
                (
                    f"Planner {failure_class} on {request_type} v{plan_version} "
                    f"attempt {attempt_number}. Retrying same request once. {summary}"
                ),
                context="plan_mode",
            )
            self._sync_planner_runtime_state()
            return

        final_message = (
            f"Planner {failure_class} on {request_type} v{plan_version} attempt {attempt_number}: {summary}"
        )
        self.notification_service.send_error(final_message, context="plan_mode")
        self._terminal_stop_reason = StopReason.INVALID_OUTPUT
        self._terminal_stop_summary = final_message

    # ---------------------------------------------------------------
    # Persist completed planner run
    # ---------------------------------------------------------------

    def _persist_completed_planner_run(self) -> str | None:
        """Consume and persist the completed planner run snapshot."""
        if not self._plan_store:
            return None
        snapshot = self.planner_service.consume_last_completed_plan_runtime()
        if snapshot is None:
            return None
        path = self._plan_store.save_planner_run(
            request_type=snapshot.request_type,
            request_version=snapshot.request_version,
            attempt_number=snapshot.attempt_number,
            execution_seq=getattr(snapshot, "execution_seq", 1),
            payload=snapshot.to_dict(),
        )
        return str(path)

    # ---------------------------------------------------------------
    # Repair eligibility
    # ---------------------------------------------------------------

    def _should_attempt_plan_repair(self) -> bool:
        return (
            self.state.current_plan_version == 0
            and self.state.last_rejected_plan_version is not None
            and self.state.current_plan_attempt_type in {"create", "repair"}
            and 1 < self.state.current_plan_attempt <= self._max_plan_attempts
            and bool(self.state.last_rejected_plan_artifact)
        )

    def _build_repair_request(self) -> PlanRepairRequest | None:
        artifact_path = self.state.last_rejected_plan_artifact
        if not artifact_path:
            return None
        try:
            payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load rejected plan artifact %s: %s", artifact_path, exc)
            return None

        errors = [
            PlanValidationError(
                stage_number=int(err.get("stage_number", -1)),
                instruction_index=err.get("instruction_index"),
                code=str(err.get("code", "")),
                message=str(err.get("message", "")),
                offending_text=str(err.get("offending_text", "")),
            )
            for err in payload.get("validation_errors", [])
            if isinstance(err, dict)
        ]

        return PlanRepairRequest(
            goal=self.config.goal,
            plan_version=int(payload.get("plan_version", self.state.last_rejected_plan_version or 1)),
            attempt_number=int(payload.get("attempt_number", 1)) + 1,
            invalid_plan_data=payload.get("parsed_data", {}),
            validation_errors=errors,
        )
