"""
Async broker that owns tool bootstrap, validation, autopoll, and incident capture.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from typing import Any

from app.broker.contract_traps import AppliedToolRewrite, apply_known_contract_traps
from app.broker.response_normalization import NormalizedToolPayload, normalize_tool_payload
from app.broker.schema_coercion import coerce_arguments_to_schema
from app.broker.semantic_contracts import SemanticContractError, apply_semantic_contract_rules
from app.broker.schema_validation import ToolArgumentValidationError, validate_arguments
from app.broker.transport import AsyncBrokerTransport, BrokerTransportConfig, McpBrokerTransport
from app.broker.tool_policies import infer_tool_policy, normalize_wait_argument, policy_for_call
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import BrokerHealth, ToolDefinition, ToolPolicy, ToolResultEnvelope, make_id
from app.runtime_incidents import LocalIncidentStore

logger = logging.getLogger("orchestrator.broker")

_RUNNING_STATUSES = {"queued", "pending", "running", "started", "in_progress"}
_RETRYABLE_ERROR_CLASSES = {"transport_error", "timeout", "server_error"}
_HIDDEN_PUBLIC_TOOLS = {"signal_api_binding_apply"}


class BrokerServiceError(RuntimeError):
    """Raised when the broker cannot execute a requested tool call."""


class BrokerService:
    def __init__(
        self,
        *,
        transport_config: BrokerTransportConfig,
        artifact_store: ExecutionArtifactStore,
        incident_store: LocalIncidentStore,
        autopoll_budget_seconds: float = 8.0,
        autopoll_interval_seconds: float = 1.0,
        transport: AsyncBrokerTransport | None = None,
        console_controller: Any | None = None,
    ) -> None:
        self.transport: AsyncBrokerTransport = transport or McpBrokerTransport(transport_config)
        self.artifact_store = artifact_store
        self.incident_store = incident_store
        self.autopoll_budget_seconds = autopoll_budget_seconds
        self.autopoll_interval_seconds = autopoll_interval_seconds
        self._tool_definitions: dict[str, ToolDefinition] = {}
        self._bootstrapped = False
        self._capturing_incident = False
        self._transport_config = transport_config
        self._tool_surface_generation = ""
        self.console_controller = console_controller

    def validate_runtime_requirements(self) -> None:
        self.transport.validate_runtime_requirements()

    async def close(self) -> None:
        await self.transport.close()

    async def bootstrap(self) -> BrokerHealth:
        return await self._bootstrap(force_refresh=True)

    async def _bootstrap(self, *, force_refresh: bool) -> BrokerHealth:
        self.validate_runtime_requirements()
        await self.transport.open()
        bootstrap_result = await self._safe_call("system_bootstrap", {"view": "summary"})
        # Public dev_space1 schema currently declares validate_tools as a string, not a boolean.
        health_result = await self._safe_call("system_health", {"validate_tools": "true", "view": "summary"})
        tools_result = await self.transport.list_tools()
        tools = getattr(tools_result, "tools", None)
        if tools is None and isinstance(tools_result, dict):
            tools = tools_result.get("tools", [])
        tools = tools or []
        self._tool_definitions = {}
        for tool in tools:
            if isinstance(tool, dict):
                name = str(tool.get("name", "") or "")
                title = str(tool.get("title", "") or "")
                description = str(tool.get("description", "") or "")
                input_schema = dict(tool.get("inputSchema", {}) or {})
                output_schema = dict(tool.get("outputSchema", {}) or {})
            else:
                name = str(getattr(tool, "name", "") or "")
                title = str(getattr(tool, "title", "") or "")
                description = str(getattr(tool, "description", "") or "")
                input_schema = dict(getattr(tool, "inputSchema", {}) or {})
                output_schema = dict(getattr(tool, "outputSchema", {}) or {})
            if name in _HIDDEN_PUBLIC_TOOLS:
                continue
            definition = ToolDefinition(
                tool_name=name,
                title=title,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                policy=infer_tool_policy(name),
            )
            definition.policy.tool_name = definition.tool_name
            if definition.tool_name:
                self._tool_definitions[definition.tool_name] = definition
        self._bootstrapped = True
        self._tool_surface_generation = _extract_surface_generation(health_result)
        warnings = self._bootstrap_warnings(bootstrap_result=bootstrap_result, health_result=health_result)
        notes = self._bootstrap_notes(warnings)
        health = BrokerHealth(
            endpoint_url=self._transport_config.endpoint_url,
            bootstrapped_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            session_id=self.transport.session_id(),
            tool_count=len(self._tool_definitions),
            status="degraded" if warnings else "healthy",
            summary=self._summarize_bootstrap(bootstrap_result, health_result, warnings=warnings),
            notes=notes,
            warnings=warnings,
        )
        if not self._tool_definitions:
            raise BrokerServiceError("broker_bootstrap_discovered_no_tools")
        return health

    def allowlist(self) -> set[str]:
        return set(self._tool_definitions)

    def policy_for(self, tool_name: str) -> ToolPolicy:
        definition = self._tool_definitions.get(tool_name)
        return definition.policy if definition else infer_tool_policy(tool_name)

    def policy_for_call(self, tool_name: str, arguments: dict[str, Any]) -> ToolPolicy:
        return policy_for_call(tool_name=tool_name, arguments=arguments)

    async def call_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        plan_id: str = "",
        slice_id: str = "",
        slot: int = 0,
        phase: str = "call",
    ) -> ToolResultEnvelope:
        if not self._bootstrapped:
            await self.bootstrap()
        if tool_name in _HIDDEN_PUBLIC_TOOLS:
            raise BrokerServiceError(f"tool_hidden_from_orchestrator:{tool_name}")
        definition = await self._resolve_tool_definition(tool_name)
        normalized_arguments, applied_rewrites = await self._normalize_call_arguments(
            tool_name=tool_name,
            arguments=arguments,
            definition=definition,
        )
        started = time.monotonic()
        if self.console_controller is not None and slot > 0:
            self.console_controller.on_tool_call_started(
                slot=slot,
                plan_id=plan_id,
                slice_id=slice_id,
                tool_name=tool_name,
                phase=phase,
            )
        try:
            raw_result = await self.transport.call_tool(tool_name, normalized_arguments)
            raw_payload = self._to_plain(raw_result)
        except Exception as exc:
            await self._capture_incident(
                summary=f"Broker transport call failed: {tool_name}",
                error=str(exc),
                affected_tool=tool_name,
                metadata={"arguments": normalized_arguments},
                severity="high",
            )
            raise BrokerServiceError(str(exc)) from exc
        autopoll_payloads, resume_tool, resume_arguments, resume_token = await self._autopoll_if_needed(
            tool_name=tool_name,
            arguments=normalized_arguments,
            raw_payload=raw_payload,
        )
        combined_payload = raw_payload
        if autopoll_payloads:
            combined_payload = {"initial": raw_payload, "autopoll": autopoll_payloads}
        envelope = self._summarize_tool_result(
            tool_name=tool_name,
            arguments=normalized_arguments,
            raw_payload=combined_payload,
            duration_ms=int((time.monotonic() - started) * 1000),
            resume_tool=resume_tool,
            resume_token=resume_token,
            resume_arguments=resume_arguments,
        )
        if applied_rewrites:
            envelope.warnings = _merge_warnings(
                envelope.warnings,
                [f"broker_contract_trap:{item.reason_code}" for item in applied_rewrites],
            )
        raw_path = self.artifact_store.save_tool_result(envelope, combined_payload)
        envelope.raw_result_ref = str(raw_path)
        envelope.plan_id = plan_id or envelope.plan_id
        envelope.slice_id = slice_id or envelope.slice_id
        self._log_tool_call(
            envelope,
            applied_rewrites=applied_rewrites,
            original_arguments=arguments,
            normalized_arguments=normalized_arguments,
        )
        for rewrite in applied_rewrites:
            await self._capture_incident(
                summary=rewrite.reason_code,
                error=rewrite.summary,
                affected_tool=tool_name,
                metadata={
                    "original_arguments": rewrite.original_arguments,
                    "rewritten_arguments": rewrite.rewritten_arguments,
                    "reason_code": rewrite.reason_code,
                },
                severity="medium",
            )
        await self._capture_warning_incidents(tool_name=tool_name, arguments=arguments, envelope=envelope)
        if self.console_controller is not None and slot > 0:
            self.console_controller.on_tool_call_finished(slot=slot, result=envelope)
        if not envelope.ok:
            await self._capture_incident(
                summary=f"Brokered tool call failed: {tool_name}",
                error=envelope.summary or envelope.error_class,
                affected_tool=tool_name,
                metadata={"arguments": arguments, "result": asdict(envelope)},
            )
        return envelope

    async def record_worker_issues(self, issues: list[dict[str, Any]] | list[Any], *, tool_name: str = "") -> None:
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            summary = str(issue.get("summary", "") or "").strip()
            if not summary:
                continue
            await self._capture_incident(
                summary=_normalize_incident_summary(summary=summary, details=str(issue.get("details", "") or "")),
                error=str(issue.get("details", "") or "").strip(),
                affected_tool=str(issue.get("affected_tool", tool_name) or ""),
                metadata=issue,
                severity=str(issue.get("severity", "medium") or "medium"),
            )

    async def report_incident(
        self,
        *,
        summary: str,
        error: str,
        affected_tool: str,
        metadata: dict[str, Any],
        severity: str = "medium",
    ) -> None:
        await self._capture_incident(
            summary=summary,
            error=error,
            affected_tool=affected_tool,
            metadata=metadata,
            severity=severity,
        )

    async def _autopoll_if_needed(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        raw_payload: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any], str]:
        policy = self.policy_for(tool_name)
        if not policy.autopoll_enabled:
            return [], "", {}, ""
        normalized = normalize_tool_payload(raw_payload)
        status = normalized.operation_status.lower()
        if status not in _RUNNING_STATUSES:
            return [], normalized.resume_tool, normalized.resume_arguments, normalized.resume_token
        if normalized.resume_tool and normalized.resume_tool != tool_name:
            return [], normalized.resume_tool, normalized.resume_arguments, normalized.resume_token
        latest_arguments = normalized.resume_arguments or _build_resume_arguments(tool_name=tool_name, initial_arguments=arguments, raw_payload=raw_payload)
        latest_tool = normalized.resume_tool or tool_name
        latest_token = normalized.resume_token or normalized.operation_ref
        if not latest_arguments:
            return [], latest_tool, {}, latest_token
        deadline = time.monotonic() + max(0.0, self.autopoll_budget_seconds)
        payloads: list[dict[str, Any]] = []
        while latest_arguments and time.monotonic() < deadline:
            await asyncio.sleep(self.autopoll_interval_seconds)
            polled = self._to_plain(await self.transport.call_tool(latest_tool, latest_arguments))
            payloads.append(polled)
            normalized_polled = normalize_tool_payload(polled)
            polled_status = normalized_polled.operation_status.lower()
            if polled_status and polled_status not in _RUNNING_STATUSES:
                return payloads, "", {}, normalized_polled.resume_token or normalized_polled.operation_ref
            latest_tool = normalized_polled.resume_tool or latest_tool
            latest_token = normalized_polled.resume_token or normalized_polled.operation_ref or latest_token
            latest_arguments = normalized_polled.resume_arguments or _build_resume_arguments(
                tool_name=latest_tool,
                initial_arguments=latest_arguments,
                raw_payload=polled,
            )
        return payloads, latest_tool, latest_arguments or {}, latest_token

    async def _safe_call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._to_plain(await self.transport.call_tool(tool_name, arguments))
        except Exception as exc:
            await self._capture_incident(
                summary=f"Broker bootstrap call failed: {tool_name}",
                error=str(exc),
                affected_tool=tool_name,
                metadata={"arguments": arguments},
                severity="medium",
            )
            return {"error": str(exc)}

    async def _capture_incident(
        self,
        *,
        summary: str,
        error: str,
        affected_tool: str,
        metadata: dict[str, Any],
        severity: str = "medium",
    ) -> None:
        local_metadata = {"error": error, "affected_tool": affected_tool, **metadata}
        if self._capturing_incident:
            self._record_local_incident(summary=summary, metadata=local_metadata, source="broker_fallback", severity=severity)
            return
        self._capturing_incident = True
        try:
            self._record_local_incident(summary=summary, metadata=local_metadata, source="broker", severity=severity)
            if self._bootstrapped and "incidents" in self._tool_definitions and affected_tool != "incidents":
                try:
                    await self.transport.call_tool(
                        "incidents",
                        {
                            "action": "capture",
                            "summary": summary,
                            "error": error[:1000],
                            "severity": severity,
                            "affected_tool": affected_tool,
                            "service": "dev_space1",
                            "metadata": json.dumps(metadata, ensure_ascii=False),
                        },
                    )
                    return
                except Exception as exc:
                    self._record_local_incident(
                        summary=f"{summary}_remote_capture_failed",
                        metadata={**local_metadata, "incident_capture_error": str(exc)},
                        source="broker_remote_capture",
                        severity=severity,
                    )
        finally:
            self._capturing_incident = False

    def _record_local_incident(
        self,
        *,
        summary: str,
        metadata: dict[str, Any],
        source: str,
        severity: str,
    ) -> None:
        self.incident_store.record(summary=summary, metadata=metadata, source=source, severity=severity)

    def _summarize_bootstrap(self, bootstrap_result: dict[str, Any], health_result: dict[str, Any], *, warnings: list[str]) -> str:
        bootstrap_text = _extract_summary(bootstrap_result)
        health_text = _extract_summary(health_result)
        parts = [part for part in (bootstrap_text, health_text) if part]
        if warnings:
            parts.append(f"bootstrap_warnings={len(warnings)}")
        return " | ".join(parts)[:600]

    def _bootstrap_warnings(self, *, bootstrap_result: dict[str, Any], health_result: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        if _extract_error_class(bootstrap_result):
            warnings.append(f"system_bootstrap:{_extract_error_class(bootstrap_result)}")
        elif not bootstrap_result or not _extract_summary(bootstrap_result):
            warnings.append("system_bootstrap:empty_payload")
        if _extract_error_class(health_result):
            warnings.append(f"system_health:{_extract_error_class(health_result)}")
        elif not health_result or not (_extract_summary(health_result) or _extract_status(health_result)):
            warnings.append("system_health:empty_payload")
        return warnings

    @staticmethod
    def _bootstrap_notes(warnings: list[str]) -> list[str]:
        if not warnings:
            return []
        return [
            "Broker bootstrap completed with warnings.",
            f"Warnings: {', '.join(warnings[:3])}",
        ]

    def _summarize_tool_result(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        raw_payload: dict[str, Any],
        duration_ms: int,
        resume_tool: str,
        resume_token: str,
        resume_arguments: dict[str, Any],
    ) -> ToolResultEnvelope:
        normalized = normalize_tool_payload(raw_payload)
        return ToolResultEnvelope(
            call_id=make_id("tool"),
            tool=tool_name,
            ok=normalized.ok,
            retryable=bool(normalized.error_class in _RETRYABLE_ERROR_CLASSES),
            duration_ms=duration_ms,
            summary=normalized.summary or (normalized.operation_status if normalized.ok else normalized.error_class or "tool_call_failed"),
            key_facts=normalized.key_facts,
            artifact_ids=normalized.artifact_ids,
            warnings=normalized.warnings,
            error_class=normalized.error_class,
            request_arguments=arguments,
            response_status=normalized.operation_status,
            tool_response_status=normalized.tool_status,
            operation_ref=resume_token or normalized.operation_ref,
            resume_tool=resume_tool or normalized.resume_tool or tool_name,
            resume_token=resume_token or normalized.resume_token,
            resume_arguments=resume_arguments or normalized.resume_arguments,
        )

    def _log_tool_call(
        self,
        envelope: ToolResultEnvelope,
        *,
        applied_rewrites: list[AppliedToolRewrite],
        original_arguments: dict[str, Any],
        normalized_arguments: dict[str, Any],
    ) -> None:
        payload = {
            "event": "broker_tool_call",
            "tool_name": envelope.tool,
            "summary": envelope.summary[:160],
            "duration_ms": envelope.duration_ms,
            "ok": envelope.ok,
            "retryable": envelope.retryable,
            "response_status": envelope.response_status,
            "tool_response_status": envelope.tool_response_status,
            "operation_ref": envelope.operation_ref,
            "warning_count": len(envelope.warnings),
            "domain_warning_present": bool(envelope.warnings),
            "error_class": envelope.error_class,
            "rewrites_applied": [item.reason_code for item in applied_rewrites],
            "arguments_normalized": bool(applied_rewrites),
            "original_arguments": original_arguments,
            "normalized_arguments": normalized_arguments,
            "raw_result_ref": envelope.raw_result_ref,
            "plan_id": envelope.plan_id,
            "slice_id": envelope.slice_id,
        }
        logger.info(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            extra={
                "event_kind": "broker_tool_call",
                "tool_name": envelope.tool,
                "plan_id": envelope.plan_id,
                "slice_id": envelope.slice_id,
                "warning_count": len(envelope.warnings),
            },
        )

    @staticmethod
    def _to_plain(result: Any) -> dict[str, Any]:
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json", exclude_none=True)
        if isinstance(result, dict):
            return result
        return {"value": str(result)}

    async def _resolve_tool_definition(self, tool_name: str) -> ToolDefinition:
        definition = self._tool_definitions.get(tool_name)
        if definition is not None:
            return definition
        await self._refresh_tool_surface(reason=f"tool_not_registered:{tool_name}")
        definition = self._tool_definitions.get(tool_name)
        if definition is None:
            raise BrokerServiceError(f"tool_not_registered:{tool_name}")
        return definition

    async def _normalize_call_arguments(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        definition: ToolDefinition,
    ) -> tuple[dict[str, Any], list[AppliedToolRewrite]]:
        normalized_arguments = dict(arguments)
        policy = self.policy_for_call(tool_name, normalized_arguments)
        normalized_arguments, applied_rewrites = apply_known_contract_traps(
            tool_name=tool_name,
            arguments=normalized_arguments,
            policy=policy,
            supports_wait=_schema_supports_wait(definition.input_schema),
        )
        try:
            normalized_arguments, semantic_rewrites = apply_semantic_contract_rules(
                tool_name=tool_name,
                arguments=normalized_arguments,
            )
            if semantic_rewrites:
                applied_rewrites.extend(semantic_rewrites)
        except SemanticContractError as exc:
            await self._capture_incident(
                summary=f"Broker semantic contract rejected call: {tool_name}",
                error=str(exc),
                affected_tool=tool_name,
                metadata={"arguments": arguments, "reason_code": exc.reason_code},
                severity="medium",
            )
            raise BrokerServiceError(str(exc)) from exc
        try:
            normalized_arguments = normalize_wait_argument(
                tool_name=tool_name,
                policy=policy,
                arguments=normalized_arguments,
                supports_wait=_schema_supports_wait(definition.input_schema),
            )
        except ValueError as exc:
            await self._capture_incident(
                summary=f"Broker wait-policy rejected call: {tool_name}",
                error=str(exc),
                affected_tool=tool_name,
                metadata={"arguments": arguments},
                    severity="medium",
                )
            raise BrokerServiceError(str(exc)) from exc
        normalized_arguments = coerce_arguments_to_schema(
            schema=definition.input_schema,
            arguments=normalized_arguments,
        )
        try:
            validate_arguments(schema=definition.input_schema, arguments=normalized_arguments, tool_name=tool_name)
            return normalized_arguments, applied_rewrites
        except ToolArgumentValidationError as exc:
            await self._refresh_tool_surface(reason=f"schema_validation_failed:{tool_name}")
            refreshed_definition = self._tool_definitions.get(tool_name, definition)
            normalized_arguments = coerce_arguments_to_schema(
                schema=refreshed_definition.input_schema,
                arguments=normalized_arguments,
            )
            try:
                validate_arguments(schema=refreshed_definition.input_schema, arguments=normalized_arguments, tool_name=tool_name)
                return normalized_arguments, applied_rewrites
            except ToolArgumentValidationError:
                await self._capture_incident(
                    summary=f"Broker argument validation failed: {tool_name}",
                    error=str(exc),
                    affected_tool=tool_name,
                    metadata={"arguments": normalized_arguments, "input_schema": refreshed_definition.input_schema},
                    severity="medium",
                )
                raise BrokerServiceError(str(exc)) from exc

    async def _refresh_tool_surface(self, *, reason: str) -> None:
        logger.info("Refreshing broker tool surface: %s", reason)
        health_result = await self._safe_call("system_health", {"validate_tools": "true", "view": "summary"})
        new_generation = _extract_surface_generation(health_result)
        if new_generation and new_generation != self._tool_surface_generation:
            logger.info("Detected tool surface generation change: %s -> %s", self._tool_surface_generation, new_generation)
        await self._bootstrap(force_refresh=False)

    async def _capture_warning_incidents(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        envelope: ToolResultEnvelope,
    ) -> None:
        if tool_name in {"incidents"}:
            return
        for warning in envelope.warnings:
            await self._capture_incident(
                summary=_warning_incident_summary(tool_name=tool_name, warning=warning),
                error=warning,
                affected_tool=tool_name,
                metadata={"arguments": arguments, "result": asdict(envelope)},
                severity="medium",
            )


def _extract_summary(raw_payload: dict[str, Any]) -> str:
    return normalize_tool_payload(raw_payload).summary


def _extract_status(raw_payload: dict[str, Any]) -> str:
    return normalize_tool_payload(raw_payload).operation_status


def _extract_error_class(raw_payload: dict[str, Any]) -> str:
    return normalize_tool_payload(raw_payload).error_class


def _extract_key_facts(raw_payload: dict[str, Any]) -> dict[str, Any]:
    return normalize_tool_payload(raw_payload).key_facts


def _extract_artifact_ids(raw_payload: dict[str, Any]) -> list[str]:
    return normalize_tool_payload(raw_payload).artifact_ids


def _extract_operation_ref(raw_payload: dict[str, Any]) -> str:
    normalized = normalize_tool_payload(raw_payload)
    return normalized.resume_token or normalized.operation_ref


def _extract_warnings(raw_payload: dict[str, Any]) -> list[str]:
    return normalize_tool_payload(raw_payload).warnings


def _merge_warnings(existing: list[str], extra: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*(existing or []), *(extra or [])]:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            merged.append(text)
    return merged


def _warning_incident_summary(*, tool_name: str, warning: str) -> str:
    warning_lower = warning.lower()
    if tool_name == "events" and "outside the available anchor dataset coverage" in warning_lower:
        return "events_anchor_coverage_mismatch"
    return f"{tool_name}_warning"


def _normalize_incident_summary(*, summary: str, details: str) -> str:
    combined = f"{summary} {details}".lower()
    if "liquidation" in combined and "no" in combined and "family" in combined:
        return "domain_capability_gap_liquidation_events_missing"
    return summary


def _build_resume_arguments(tool_name: str, initial_arguments: dict[str, Any], raw_payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_tool_payload(raw_payload)
    if normalized.resume_tool and normalized.resume_arguments:
        return dict(normalized.resume_arguments)
    transformer = _status_request_for(tool_name=tool_name, initial_arguments=initial_arguments, raw_payload=raw_payload)
    if transformer is None:
        return {}
    return transformer() or {}


def _status_request_for(tool_name: str, initial_arguments: dict[str, Any], raw_payload: dict[str, Any]) -> Any:
    normalized = normalize_tool_payload(raw_payload)
    if normalized.resume_tool and normalized.resume_arguments:
        return lambda: dict(normalized.resume_arguments)
    data = normalized.data if isinstance(normalized.data, dict) else {}
    operation = data.get("operation") if isinstance(data, dict) else {}
    job = data.get("job") if isinstance(data, dict) else {}
    run = data.get("run") if isinstance(data, dict) else {}
    operation_id = str(data.get("operation_id", "") or (operation.get("operation_id", "") if isinstance(operation, dict) else "") or "")
    job_id = str(data.get("job_id", "") or (job.get("job_id", "") if isinstance(job, dict) else "") or "")
    run_id = str(data.get("run_id", "") or (run.get("run_id", "") if isinstance(run, dict) else "") or "")
    if tool_name == "backtests_runs" and run_id:
        return lambda: {"action": "inspect", "view": "status", "run_id": run_id}
    if tool_name in {"backtests_analysis", "backtests_conditions", "backtests_studies", "backtests_walkforward"} and job_id:
        return lambda: {"action": "status", "job_id": job_id}
    if tool_name in {"features_custom", "features_dataset", "models_dataset", "models_train", "experiments_run", "research_record"}:
        ref_key = "operation_id" if operation_id else "job_id"
        ref_value = operation_id or job_id
        if ref_value:
            return lambda: {"action": "status", ref_key: ref_value}
    if tool_name == "datasets_sync" and operation_id:
        return lambda: {"action": initial_arguments.get("action", "sync"), "symbol": initial_arguments.get("symbol", ""), "timeframes": initial_arguments.get("timeframes"), "wait": "started"}
    if tool_name == "events_sync" and operation_id:
        return lambda: {
            "family": initial_arguments.get("family", ""),
            "scope": initial_arguments.get("scope", ""),
            "symbol": initial_arguments.get("symbol"),
            "currency": initial_arguments.get("currency"),
            "wait": "started",
        }
    return None


def _extract_surface_generation(health_result: dict[str, Any]) -> str:
    normalized = normalize_tool_payload(health_result)
    data = normalized.data if isinstance(normalized.data, dict) else {}
    server_health = data.get("server_health") if isinstance(data.get("server_health"), dict) else {}
    tool_surface = server_health.get("tool_surface") if isinstance(server_health.get("tool_surface"), dict) else {}
    for key in ("server_dispatch_generation", "surface_fingerprint"):
        value = str(data.get(key, "") or "").strip()
        if value:
            return value
    for key in ("tool_surface_generation", "registry_generation", "surface_fingerprint"):
        value = str(tool_surface.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _schema_supports_wait(schema: dict[str, Any]) -> bool:
    properties = schema.get("properties", {}) or {}
    return isinstance(properties, dict) and "wait" in properties


__all__ = [
    "BrokerService",
    "BrokerServiceError",
    "_build_resume_arguments",
    "_extract_artifact_ids",
    "_extract_error_class",
    "_extract_key_facts",
    "_extract_operation_ref",
    "_extract_status",
    "_extract_summary",
    "_extract_warnings",
]
