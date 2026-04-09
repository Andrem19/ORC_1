from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from app.broker.service import (
    BrokerService,
    BrokerServiceError,
    _extract_error_class,
    _extract_key_facts,
    _extract_status,
    _extract_summary,
)
from app.broker.transport import BrokerTransportConfig, McpBrokerTransport
from app.execution_artifacts import ExecutionArtifactStore
from app.runtime_incidents import LocalIncidentStore


class _FakeTransport:
    def __init__(self, *, responses: dict[str, list[dict[str, object]]], tools: list[dict[str, object]]) -> None:
        self.responses = {key: list(value) for key, value in responses.items()}
        self.tools = tools
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.opened = False
        self.closed = False
        self.validated = False

    def validate_runtime_requirements(self) -> None:
        self.validated = True

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True

    async def initialize(self):
        return {"ok": True}

    async def list_tools(self):
        return {"tools": self.tools}

    async def call_tool(self, tool_name: str, arguments: dict[str, object] | None = None):
        args = dict(arguments or {})
        self.calls.append((tool_name, args))
        scripted = self.responses.get(tool_name, [])
        if scripted:
            return scripted.pop(0)
        return {"structuredContent": {"status": "ok", "message": f"{tool_name} ok", "data": {"status": "completed"}}}

    def session_id(self) -> str:
        return "session_test"


def _make_service(tmp_path, transport) -> BrokerService:
    return BrokerService(
        transport_config=BrokerTransportConfig(),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans"),
        incident_store=LocalIncidentStore(tmp_path / "state"),
        autopoll_budget_seconds=0.02,
        autopoll_interval_seconds=0.001,
        transport=transport,
    )


def test_transport_does_not_use_asyncio_runner() -> None:
    transport = McpBrokerTransport(BrokerTransportConfig())

    assert not hasattr(transport, "_runner")


def test_broker_fast_fails_when_sdk_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.broker.transport._load_mcp_client_session", lambda: (_ for _ in ()).throw(RuntimeError("sdk_missing")))
    monkeypatch.setattr("app.broker.transport._load_streamable_http_client", lambda: (_ for _ in ()).throw(RuntimeError("sdk_missing")))
    service = BrokerService(
        transport_config=BrokerTransportConfig(),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans"),
        incident_store=LocalIncidentStore(tmp_path / "state"),
    )

    with pytest.raises(RuntimeError, match="sdk_missing"):
        service.validate_runtime_requirements()


def test_broker_bootstrap_and_autopoll_with_fake_transport(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "features_dataset": [
                {
                    "structuredContent": {
                        "status": "ok",
                        "message": "started",
                        "data": {"status": "running", "operation_id": "op_123"},
                    }
                },
                {
                    "structuredContent": {
                        "status": "ok",
                        "message": "done",
                        "data": {"status": "completed", "operation_id": "op_123"},
                    }
                },
            ],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {
                "name": "features_dataset",
                "description": "features dataset",
                "inputSchema": {
                    "type": "object",
                    "required": ["action", "symbol", "timeframe", "columns"],
                    "properties": {
                        "action": {"type": "string"},
                        "symbol": {"type": "string"},
                        "timeframe": {"type": "string"},
                        "columns": {"type": "array"},
                    },
                },
            },
        ],
    )
    service = _make_service(tmp_path, transport)

    health = asyncio.run(service.bootstrap())
    envelope = asyncio.run(
        service.call_tool(
            tool_name="features_dataset",
            arguments={"action": "build", "symbol": "BTCUSDT", "timeframe": "1h", "columns": ["rsi_14"]},
        )
    )

    assert health.tool_count == 3
    assert transport.validated is True
    assert envelope.ok is True
    assert envelope.response_status == "completed"
    assert envelope.resume_arguments == {}
    assert transport.calls[-1] == ("features_dataset", {"action": "status", "operation_id": "op_123"})


def test_broker_extracts_nested_dataset_fact_ids(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "models_dataset": [
                {
                    "structuredContent": {
                        "status": "ok",
                        "message": "materialized",
                        "data": {
                            "final_status": "completed",
                            "dataset": {"dataset_id": "dataset_123", "row_count": 240},
                            "operation": {"operation_id": "op_123", "status": "completed"},
                        },
                    }
                }
            ],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {"name": "models_dataset", "description": "models dataset", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)

    asyncio.run(service.bootstrap())
    envelope = asyncio.run(service.call_tool(tool_name="models_dataset", arguments={"action": "status", "operation_id": "op_123"}))

    assert envelope.key_facts["dataset.dataset_id"] == "dataset_123"
    assert envelope.key_facts["dataset.row_count"] == 240
    assert envelope.key_facts["operation.operation_id"] == "op_123"


def test_broker_extracts_nested_project_fact_ids(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "research_project": [
                {
                    "structuredContent": {
                        "status": "ok",
                        "message": "created",
                        "data": {
                            "project": {"project_id": "proj_123", "root_node_id": "node_1"},
                            "state_summary": {"project_id": "proj_123"},
                            "summary": "Completed research project action create.",
                        },
                    }
                }
            ],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {"name": "research_project", "description": "research project", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)

    asyncio.run(service.bootstrap())
    envelope = asyncio.run(service.call_tool(tool_name="research_project", arguments={"action": "create"}))

    assert envelope.key_facts["project.project_id"] == "proj_123"
    assert envelope.key_facts["project.root_node_id"] == "node_1"
    assert envelope.key_facts["state_summary.project_id"] == "proj_123"


def test_broker_coerces_numeric_strings_from_schema(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "backtests_runs": [{"structuredContent": {"status": "ok", "message": "started", "data": {"final_status": "started"}}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {
                "name": "backtests_runs",
                "description": "backtests",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "starting_capital_usd": {"oneOf": [{"type": "number"}, {"type": "integer"}]},
                    },
                },
            },
        ],
    )
    service = _make_service(tmp_path, transport)

    asyncio.run(service.bootstrap())
    asyncio.run(
        service.call_tool(
            tool_name="backtests_runs",
            arguments={"action": "start", "starting_capital_usd": "1000"},
        )
    )

    assert ("backtests_runs", {"action": "start", "starting_capital_usd": 1000}) in transport.calls


def test_broker_rewrites_features_cleanup_registry_scope(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "features_cleanup": [{"structuredContent": {"status": "ok", "message": "preview ok", "data": {"final_status": "completed"}}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {"name": "features_cleanup", "description": "cleanup", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)

    asyncio.run(service.bootstrap())
    envelope = asyncio.run(
        service.call_tool(
            tool_name="features_cleanup",
            arguments={"action": "preview", "scope": "registry", "cf_patterns": "cf_*"},
        )
    )

    assert envelope.ok is True
    assert ("features_cleanup", {"action": "preview", "scope": "features_only", "cf_patterns": "cf_*"}) in transport.calls
    assert "broker_contract_trap:features_cleanup_registry_scope_rewritten" in envelope.warnings


def test_broker_rejects_research_search_without_project_id_locally(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {"name": "research_search", "description": "research", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    with pytest.raises(BrokerServiceError, match="requires a non-empty project_id"):
        asyncio.run(service.call_tool(tool_name="research_search", arguments={"query": "coverage note"}))

    assert not any(tool_name == "research_search" for tool_name, _ in transport.calls)


def test_broker_rejects_funding_events_without_symbol_locally(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {"name": "events_sync", "description": "events sync", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    with pytest.raises(BrokerServiceError, match="family='funding' requires symbol"):
        asyncio.run(service.call_tool(tool_name="events_sync", arguments={"family": "funding", "scope": "incremental"}))

    assert not any(tool_name == "events_sync" for tool_name, _ in transport.calls)


def test_broker_rejects_features_analytics_without_selector_locally(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {"name": "features_analytics", "description": "analytics", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    with pytest.raises(BrokerServiceError, match="requires one concrete feature selector"):
        asyncio.run(service.call_tool(tool_name="features_analytics", arguments={"action": "heatmap", "symbol": "BTCUSDT"}))

    assert not any(tool_name == "features_analytics" for tool_name, _ in transport.calls)


def test_broker_does_not_mirror_incidents_tool_warnings_back_into_local_store(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "incidents": [
                {
                    "structuredContent": {
                        "status": "ok",
                        "message": "incident summary",
                        "data": {"summary": "incident summary"},
                        "warnings": ["old_warning"],
                    }
                }
            ],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {"name": "incidents", "description": "incidents", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)

    asyncio.run(service.bootstrap())
    envelope = asyncio.run(service.call_tool(tool_name="incidents", arguments={"action": "inspect", "view": "summary"}))

    incidents_dir = Path(tmp_path / "state" / "incidents")
    assert envelope.ok is True
    assert envelope.warnings == ["old_warning"]
    assert list(incidents_dir.glob("*.json")) == []


def test_broker_bootstrap_uses_string_validate_tools_contract(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)

    asyncio.run(service.bootstrap())

    assert ("system_health", {"validate_tools": "true", "view": "summary"}) in transport.calls


def test_broker_hides_signal_binding_apply_from_allowlist_and_calls(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {"name": "signal_api_binding_apply", "description": "operator-only", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)

    health = asyncio.run(service.bootstrap())

    assert "signal_api_binding_apply" not in service.allowlist()
    assert health.tool_count == 2
    with pytest.raises(BrokerServiceError, match="tool_hidden_from_orchestrator:signal_api_binding_apply"):
        asyncio.run(service.call_tool(tool_name="signal_api_binding_apply", arguments={}))


def test_broker_bootstrap_partial_failure_is_reported_as_degraded(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"error": "bootstrap unavailable"}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)

    health = asyncio.run(service.bootstrap())

    assert health.status == "degraded"
    assert "system_bootstrap:server_error" in health.warnings
    assert any("warnings" in note.lower() for note in health.notes)


def test_broker_validation_and_server_errors_capture_local_incidents(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "events": [{"error": "server exploded"}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {
                "name": "events",
                "description": "events",
                "inputSchema": {
                    "type": "object",
                    "required": ["view", "symbol"],
                    "properties": {"view": {"type": "string"}, "symbol": {"type": "string"}},
                },
            },
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    with pytest.raises(BrokerServiceError, match="missing required field 'symbol'"):
        asyncio.run(service.call_tool(tool_name="events", arguments={"view": "catalog"}))

    incidents_dir = Path(tmp_path / "state" / "incidents")
    assert any(incidents_dir.glob("*.json"))

    envelope = asyncio.run(service.call_tool(tool_name="events", arguments={"view": "catalog", "symbol": "BTCUSDT"}))

    assert envelope.ok is False
    assert envelope.error_class == "server_error"
    assert len(list(incidents_dir.glob("*.json"))) >= 2


def test_broker_incident_uses_remote_incidents_tool_when_available(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "incidents": [{"structuredContent": {"status": "ok", "message": "captured"}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {"name": "incidents", "description": "incidents", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    asyncio.run(
        service._capture_incident(
            summary="contract issue",
            error="details",
            affected_tool="events",
            metadata={"k": "v"},
        )
    )

    incidents_dir = Path(tmp_path / "state" / "incidents")
    incidents = list(incidents_dir.glob("*.json"))
    assert incidents
    assert any(call[0] == "incidents" for call in transport.calls)


def test_broker_incident_falls_back_to_local_store_when_remote_capture_fails(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {"name": "incidents", "description": "incidents", "inputSchema": {"type": "object"}},
        ],
    )

    async def broken_call_tool(tool_name: str, arguments: dict[str, object] | None = None):
        if tool_name == "incidents":
            raise RuntimeError("incident capture failed")
        return await _FakeTransport.call_tool(transport, tool_name, arguments)

    transport.call_tool = broken_call_tool  # type: ignore[assignment]
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    asyncio.run(
        service._capture_incident(
            summary="contract issue",
            error="details",
            affected_tool="events",
            metadata={"k": "v"},
        )
    )

    incidents = list((Path(tmp_path) / "state" / "incidents").glob("*.json"))
    assert incidents
    payloads = [path.read_text(encoding="utf-8") for path in incidents]
    assert any("incident_capture_error" in payload for payload in payloads)


def test_broker_summary_helpers_prefer_autopoll_tail_payload() -> None:
    payload = {
        "initial": {
            "structuredContent": {
                "status": "ok",
                "message": "started",
                "data": {"status": "running", "count": 1},
            }
        },
        "autopoll": [
            {
                "structuredContent": {
                    "status": "ok",
                    "message": "done",
                    "data": {"status": "completed", "count": 5, "operation_id": "op_1"},
                }
            }
        ],
    }

    assert _extract_summary(payload) == "done"
    assert _extract_status(payload) == "completed"
    assert _extract_key_facts(payload)["count"] == 5
    assert _extract_error_class({"error": "timeout"}) == "timeout"


def test_broker_collects_nested_domain_warnings_and_records_incident(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "events": [
                {
                    "structuredContent": {
                        "status": "ok",
                        "message": "Event payload loaded.",
                        "data": {
                            "preview": {
                                "warnings": [
                                    "The local event store contains rows outside the available anchor dataset coverage for this symbol/timeframe."
                                ]
                            }
                        },
                    }
                }
            ],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {"name": "events", "description": "events", "inputSchema": {"type": "object", "properties": {"view": {"type": "string"}}, "required": ["view"]}},
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    envelope = asyncio.run(service.call_tool(tool_name="events", arguments={"view": "align_preview"}))

    assert envelope.warnings
    assert "anchor dataset coverage" in envelope.warnings[0]
    incidents = list((Path(tmp_path) / "state" / "incidents").glob("*.json"))
    assert incidents
    payload = incidents[-1].read_text(encoding="utf-8")
    assert "events_anchor_coverage_mismatch" in payload


def test_broker_logs_structured_call_record(tmp_path, caplog) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "events": [{"structuredContent": {"status": "ok", "message": "catalog ok", "data": {"status": "completed"}}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {
                "name": "events",
                "description": "events",
                "inputSchema": {"type": "object", "properties": {"view": {"type": "string"}}, "required": ["view"]},
            },
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    with patch("app.broker.service.logger.info") as info_mock:
        asyncio.run(service.call_tool(tool_name="events", arguments={"view": "catalog"}))

    record = json.loads(info_mock.call_args.args[0])
    assert record["event"] == "broker_tool_call"
    assert record["tool_name"] == "events"
    assert record["ok"] is True
    assert "duration_ms" in record
    assert "summary" in record
    assert "domain_warning_present" in record


def test_broker_logs_structured_error_record(tmp_path, caplog) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "events": [{"error": "server exploded"}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {
                "name": "events",
                "description": "events",
                "inputSchema": {"type": "object", "properties": {"view": {"type": "string"}}, "required": ["view"]},
            },
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    with patch("app.broker.service.logger.info") as info_mock:
        envelope = asyncio.run(service.call_tool(tool_name="events", arguments={"view": "catalog"}))

    record = json.loads(info_mock.call_args.args[0])
    assert envelope.ok is False
    assert record["ok"] is False
    assert record["error_class"] == "server_error"


def test_broker_rewrites_research_search_compact_level(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "research_search": [{"structuredContent": {"status": "ok", "message": "Research search payload loaded.", "data": {"level": "normal"}}}],
            "incidents": [{"structuredContent": {"status": "ok", "message": "captured"}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {
                "name": "research_search",
                "description": "search",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "level": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {"name": "incidents", "description": "incidents", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    with patch("app.broker.service.logger.info") as info_mock:
        envelope = asyncio.run(
            service.call_tool(
                tool_name="research_search",
                arguments={"query": "baseline", "level": "compact", "project_id": "proj_123"},
            )
        )

    assert envelope.ok is True
    assert ("research_search", {"query": "baseline", "level": "normal", "project_id": "proj_123"}) in transport.calls
    assert any(item == "broker_contract_trap:research_search_compact_level_rewritten" for item in envelope.warnings)
    record = json.loads(info_mock.call_args.args[0])
    assert record["arguments_normalized"] is True
    assert "research_search_compact_level_rewritten" in record["rewrites_applied"]
    incidents = list((Path(tmp_path) / "state" / "incidents").glob("*.json"))
    assert incidents
    assert "research_search_compact_level_rewritten" in incidents[-1].read_text(encoding="utf-8")


def test_broker_rewrites_expensive_async_wait_completed(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "events_sync": [{"structuredContent": {"status": "ok", "message": "sync started", "data": {"status": "running", "operation_id": "op_1"}}}],
            "incidents": [{"structuredContent": {"status": "ok", "message": "captured"}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {
                "name": "events_sync",
                "description": "events sync",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "family": {"type": "string"},
                        "scope": {"type": "string"},
                        "symbol": {"type": "string"},
                        "wait": {"type": "string"},
                    },
                    "required": ["family", "scope"],
                },
            },
            {"name": "incidents", "description": "incidents", "inputSchema": {"type": "object"}},
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    envelope = asyncio.run(
        service.call_tool(
            tool_name="events_sync",
            arguments={"family": "funding", "scope": "incremental", "symbol": "BTCUSDT", "wait": "completed"},
        )
    )

    assert envelope.ok is True
    assert ("events_sync", {"family": "funding", "scope": "incremental", "symbol": "BTCUSDT", "wait": "started"}) in transport.calls
    assert any(item == "broker_contract_trap:expensive_async_wait_completed_rewritten" for item in envelope.warnings)


def test_broker_does_not_force_wait_for_features_custom_contract_inspect(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "features_custom": [
                {
                    "structuredContent": {
                        "status": "ok",
                        "message": "Custom feature facade action completed.",
                        "tool_name": "features_custom",
                        "data": {
                            "view": "contract",
                            "contract": {
                                "required_fields_by_action": {
                                    "validate": ["description", "code", "symbol", "timeframe"],
                                    "publish": ["name", "description", "code", "symbol", "timeframe"],
                                }
                            },
                        },
                    }
                }
            ],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {
                "name": "features_custom",
                "description": "features custom",
                "inputSchema": {
                    "type": "object",
                    "properties": {"action": {"type": "string"}, "view": {"type": "string"}, "wait": {"type": "string"}},
                },
            },
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    envelope = asyncio.run(
        service.call_tool(tool_name="features_custom", arguments={"action": "inspect", "view": "contract"})
    )

    assert envelope.ok is True
    assert ("features_custom", {"action": "inspect", "view": "contract"}) in transport.calls
    assert "validate requires [description, code, symbol, timeframe]" in envelope.summary


def test_broker_does_not_force_wait_for_features_dataset_inspect_columns(tmp_path) -> None:
    transport = _FakeTransport(
        responses={
            "system_bootstrap": [{"structuredContent": {"status": "ok", "message": "bootstrap ok"}}],
            "system_health": [{"structuredContent": {"status": "ok", "message": "healthy"}}],
            "features_dataset": [{"structuredContent": {"status": "ok", "message": "Feature dataset facade action completed.", "tool_name": "features_dataset", "data": {"view": "columns"}}}],
        },
        tools=[
            {"name": "system_bootstrap", "description": "bootstrap", "inputSchema": {"type": "object"}},
            {"name": "system_health", "description": "health", "inputSchema": {"type": "object"}},
            {
                "name": "features_dataset",
                "description": "features dataset",
                "inputSchema": {
                    "type": "object",
                    "properties": {"action": {"type": "string"}, "view": {"type": "string"}, "wait": {"type": "string"}},
                },
            },
        ],
    )
    service = _make_service(tmp_path, transport)
    asyncio.run(service.bootstrap())

    envelope = asyncio.run(
        service.call_tool(tool_name="features_dataset", arguments={"action": "inspect", "view": "columns"})
    )

    assert envelope.ok is True
    assert ("features_dataset", {"action": "inspect", "view": "columns"}) in transport.calls
