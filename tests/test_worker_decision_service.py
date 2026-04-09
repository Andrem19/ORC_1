from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.adapters.base import AdapterResponse, BaseAdapter
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_parsing import parse_worker_action_output
from app.execution_models import PlanSlice
from app.services.brokered_execution.worker import WorkerDecisionError, WorkerDecisionService, WorkerParseFailureError


class _StubAdapter(BaseAdapter):
    def __init__(self, raw_output: str | list[str]) -> None:
        self.raw_output = raw_output
        self.calls = 0

    def invoke(self, prompt: str, timeout: int = 120, **kwargs):
        del prompt, timeout, kwargs
        self.calls += 1
        if isinstance(self.raw_output, list):
            index = min(self.calls - 1, len(self.raw_output) - 1)
            payload = self.raw_output[index]
        else:
            payload = self.raw_output
        return AdapterResponse(success=True, raw_output=payload)

    def is_available(self) -> bool:
        return True

    def name(self) -> str:
        return "stub_worker"


def test_worker_decision_service_saves_parse_failure_artifact(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter('{"type":"system","subtype":"init"}'),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_1",
        title="Funding",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["events"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=0,
        parallel_slot=1,
    )

    with pytest.raises(WorkerDecisionError, match="worker_action_type_invalid"):
        asyncio.run(
            service.choose_action(
                plan_id="plan_1",
                slice_obj=slice_obj,
                baseline_bootstrap={},
                known_facts={},
                recent_turn_summaries=[],
                latest_tool_summary="",
                remaining_budget={"turns_used": 0, "turns_remaining": 3, "tool_calls_used": 0, "tool_calls_remaining": 2},
                checkpoint_summary="",
                active_operation={},
            )
        )

    files = list((tmp_path / "plans" / "runs" / "run-1" / "worker_failures" / "plan_1" / "slice_1").glob("*.json"))
    assert files
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["error"] == "worker_action_type_invalid"
    assert payload["raw_output"] == '{"type":"system","subtype":"init"}'


def test_worker_decision_service_corrects_prefixed_tool_name_once(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                '{"type":"tool_call","tool":"mcp__dev_space1__features_catalog","arguments":{"scope":"available"},"reason":"inspect"}',
                '{"type":"tool_call","tool":"features_catalog","arguments":{"scope":"available"},"reason":"inspect"}',
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_1",
        title="Funding",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["features_catalog"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=0,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="",
            remaining_budget={"turns_used": 0, "turns_remaining": 3, "tool_calls_used": 0, "tool_calls_remaining": 2},
            checkpoint_summary="",
            active_operation={},
        )
    )

    assert action.tool == "features_catalog"
    files = list((tmp_path / "plans" / "runs" / "run-1" / "worker_failures" / "plan_1" / "slice_1").glob("*.json"))
    assert len(files) == 1


def test_worker_decision_service_aborts_after_corrective_retry_exhausted(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                '{"type":"tool_call","tool":"mcp__dev_space1__features_catalog","arguments":{"scope":"available"},"reason":"inspect"}',
                '{"type":"tool_call","tool":"mcp__dev_space1__features_catalog","arguments":{"scope":"available"},"reason":"inspect"}',
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_1",
        title="Funding",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["features_catalog"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=0,
        parallel_slot=1,
    )

    with pytest.raises(WorkerDecisionError, match="tool_prefixed_namespace_forbidden"):
        asyncio.run(
            service.choose_action(
                plan_id="plan_1",
                slice_obj=slice_obj,
                baseline_bootstrap={},
                known_facts={},
                recent_turn_summaries=[],
                latest_tool_summary="",
                remaining_budget={"turns_used": 0, "turns_remaining": 3, "tool_calls_used": 0, "tool_calls_remaining": 2},
                checkpoint_summary="",
                active_operation={},
            )
        )

    files = list((tmp_path / "plans" / "runs" / "run-1" / "worker_failures" / "plan_1" / "slice_1").glob("*.json"))
    assert len(files) == 2


def test_worker_decision_service_terminalizes_nonrecoverable_parse_failure(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter("Файл действия сохранён. Ожидаю следующий шаг."),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_1",
        title="Funding",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["features_catalog"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=0,
        parallel_slot=1,
    )

    with pytest.raises(WorkerParseFailureError, match="json_object_not_found"):
        asyncio.run(
            service.choose_action(
                plan_id="plan_1",
                slice_obj=slice_obj,
                baseline_bootstrap={},
                known_facts={},
                recent_turn_summaries=[],
                latest_tool_summary="",
                remaining_budget={"turns_used": 0, "turns_remaining": 3, "tool_calls_used": 0, "tool_calls_remaining": 2},
                checkpoint_summary="",
                active_operation={},
            )
        )

    files = list((tmp_path / "plans" / "runs" / "run-1" / "worker_failures" / "plan_1" / "slice_1").glob("*.json"))
    assert len(files) == 2


def test_worker_decision_service_retries_once_for_missing_json_object(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                "Файл действия сохранён. Ожидаю следующий шаг.",
                '{"type":"tool_call","tool":"features_catalog","arguments":{"scope":"available"},"reason":"inspect"}',
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_1",
        title="Funding",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["features_catalog"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=0,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="",
            remaining_budget={"turns_used": 0, "turns_remaining": 3, "tool_calls_used": 0, "tool_calls_remaining": 2},
            checkpoint_summary="",
            active_operation={},
        )
    )

    assert action.tool == "features_catalog"


def test_worker_decision_service_terminalizes_timed_out_worker_response(tmp_path) -> None:
    class _TimeoutAdapter(BaseAdapter):
        def invoke(self, prompt: str, timeout: int = 120, **kwargs):
            del prompt, timeout, kwargs
            return AdapterResponse(success=False, raw_output="", error="Timed out", timed_out=True)

        def is_available(self) -> bool:
            return True

        def name(self) -> str:
            return "timeout_worker"

    service = WorkerDecisionService(
        adapter=_TimeoutAdapter(),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_1",
        title="Funding",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["features_catalog"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=0,
        parallel_slot=1,
    )

    with pytest.raises(WorkerParseFailureError, match="worker_invoke_timeout"):
        asyncio.run(
            service.choose_action(
                plan_id="plan_1",
                slice_obj=slice_obj,
                baseline_bootstrap={},
                known_facts={},
                recent_turn_summaries=[],
                latest_tool_summary="",
                remaining_budget={"turns_used": 0, "turns_remaining": 3, "tool_calls_used": 0, "tool_calls_remaining": 2},
                checkpoint_summary="",
                active_operation={},
            )
        )

    files = list((tmp_path / "plans" / "runs" / "run-1" / "worker_failures" / "plan_1" / "slice_1").glob("*.json"))
    assert len(files) == 1


def test_parse_worker_action_accepts_abort_reason_as_summary_alias() -> None:
    action = parse_worker_action_output(
        '{"type":"abort","reason":"tools unavailable in current step","reason_code":"tool_unavailable","retryable":true}',
        allowlist={"events"},
    )

    assert action.action_type == "abort"
    assert action.summary == "tools unavailable in current step"


def test_worker_decision_service_corrects_contradictory_tool_unavailable_abort(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                json.dumps(
                    {
                        "type": "abort",
                        "reason_code": "tool_unavailable",
                        "summary": "Required event tools (events, events_sync) are not available in the tool registry.",
                        "retryable": True,
                        "reportable_issues": [
                            {
                                "summary": "tools unavailable",
                                "details": "events and events_sync not found in registry",
                                "affected_tool": "events",
                            }
                        ],
                    }
                ),
                '{"type":"tool_call","tool":"events","arguments":{"view":"catalog","family":"funding","symbol":"BTCUSDT"},"reason":"verify event coverage"}',
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_1",
        title="Events",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["events", "events_sync"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=0,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="Funding events already catalogued successfully.",
            remaining_budget={"turns_used": 1, "turns_remaining": 2, "tool_calls_used": 1, "tool_calls_remaining": 1},
            checkpoint_summary="events call succeeded",
            active_operation={},
        )
    )

    assert action.action_type == "tool_call"
    assert action.tool == "events"


def test_worker_decision_service_corrects_local_session_registry_abort(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                json.dumps(
                    {
                        "type": "abort",
                        "reason_code": "tool_name_ambiguous",
                        "summary": "Brokered tools are not available in this session.",
                        "retryable": True,
                        "reportable_issues": [
                            {
                                "summary": "Required broker execution runtime tools not available",
                                "details": "Only standard development tools are registered in this session. features_catalog and research_search are not available here.",
                                "affected_tool": "features_catalog",
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "features_catalog",
                        "arguments": {"scope": "available"},
                        "reason": "The broker owns this public tool outside the worker session.",
                    }
                ),
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_registry_abort",
        title="Catalog",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["features_catalog", "research_search"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=0,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="Feature catalog facade loaded.",
            remaining_budget={"turns_used": 0, "turns_remaining": 3, "tool_calls_used": 0, "tool_calls_remaining": 2},
            checkpoint_summary="",
            active_operation={},
        )
    )

    assert action.action_type == "tool_call"
    assert action.tool == "features_catalog"


def test_worker_decision_service_corrects_features_custom_create_semantic_drift(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "features_custom",
                        "arguments": {"action": "create", "feature_name": "cf_test"},
                        "reason": "create a new feature",
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "features_custom",
                        "arguments": {"action": "inspect", "view": "contract"},
                        "reason": "inspect the exact public contract before authoring",
                    }
                ),
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_1",
        title="Custom features",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["features_custom"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=1,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="",
            remaining_budget={"turns_used": 0, "turns_remaining": 3, "tool_calls_used": 0, "tool_calls_remaining": 2},
            checkpoint_summary="",
            active_operation={},
        )
    )

    assert action.action_type == "tool_call"
    assert action.tool == "features_custom"
    assert action.arguments == {"action": "inspect", "view": "contract"}


def test_worker_decision_service_corrects_contradictory_registry_checkpoint(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                json.dumps(
                    {
                        "type": "checkpoint",
                        "status": "blocked",
                        "summary": "events tool not found in registry; cannot continue.",
                        "pending_questions": ["Should I fall back because events is not in registry?"],
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "events_sync",
                        "arguments": {"family": "expiry", "scope": "incremental", "wait": "started"},
                        "reason": "sync fresh expiry events",
                    }
                ),
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_1",
        title="Events",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["events", "events_sync"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=1,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="Event payload loaded.",
            remaining_budget={"turns_used": 1, "turns_remaining": 2, "tool_calls_used": 1, "tool_calls_remaining": 1},
            checkpoint_summary="catalog call succeeded",
            active_operation={},
        )
    )

    assert action.action_type == "tool_call"
    assert action.tool == "events_sync"
    assert action.arguments["family"] == "expiry"


def test_worker_decision_service_corrects_events_sync_missing_required_fields(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "events_sync",
                        "arguments": {"tool": "events_sync", "symbol": "BTCUSDT", "wait": "started"},
                        "reason": "sync events before feature engineering",
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "events_sync",
                        "arguments": {"family": "funding", "scope": "incremental", "symbol": "BTCUSDT", "wait": "started"},
                        "reason": "sync funding events with required public arguments",
                    }
                ),
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_events_sync",
        title="Funding sync",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["events_sync"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=1,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="Event payload loaded.",
            remaining_budget={"turns_used": 1, "turns_remaining": 2, "tool_calls_used": 1, "tool_calls_remaining": 1},
            checkpoint_summary="catalog reviewed",
            active_operation={},
        )
    )

    assert action.action_type == "tool_call"
    assert action.tool == "events_sync"
    assert action.arguments["family"] == "funding"
    assert action.arguments["scope"] == "incremental"


def test_worker_decision_service_corrects_duplicate_events_sync_while_active_operation_running(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "events_sync",
                        "arguments": {"family": "funding", "scope": "incremental", "wait": "started"},
                        "reason": "check the same sync again",
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "events",
                        "arguments": {"view": "catalog", "family": "funding", "symbol": "BTCUSDT"},
                        "reason": "inspect cheap event catalog while sync continues in the broker",
                    }
                ),
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_events_sync_active",
        title="Funding sync",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["events", "events_sync"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=1,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="Event refresh for family=funding scope=incremental is currently running_compute.",
            remaining_budget={"turns_used": 1, "turns_remaining": 2, "tool_calls_used": 1, "tool_calls_remaining": 1},
            checkpoint_summary="sync started",
            active_operation={"tool": "events_sync", "ref": "op_1", "status": "running_compute"},
        )
    )

    assert action.action_type == "tool_call"
    assert action.tool == "events"


def test_worker_decision_service_does_not_flag_valid_tool_call_for_registry_claim_text(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "features_catalog",
                        "arguments": {"scope": "available"},
                        "reason": "Inspect the catalog even if earlier notes mentioned tool registry confusion.",
                        "reportable_issues": [
                            {
                                "code": "model_confusion",
                                "summary": "Earlier draft said features_catalog was not found in registry.",
                                "details": "This was likely stale context, not current broker state.",
                                "affected_tool": "features_catalog",
                            }
                        ],
                    }
                )
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_registry_text",
        title="Catalog",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["features_catalog"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=0,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="Feature catalog facade loaded.",
            remaining_budget={"turns_used": 0, "turns_remaining": 3, "tool_calls_used": 0, "tool_calls_remaining": 2},
            checkpoint_summary="",
            active_operation={},
        )
    )

    assert action.action_type == "tool_call"
    assert action.tool == "features_catalog"


def test_worker_decision_service_corrects_datasets_preview_missing_contract_fields(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "datasets_preview",
                        "arguments": {
                            "datasets_preview": {
                                "symbol": "BTCUSDT",
                                "timeframes": ["4h", "1d"],
                            }
                        },
                        "reason": "preview higher-timeframe datasets",
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "datasets_preview",
                        "arguments": {
                            "dataset_id": "BTCUSDT_4h",
                            "view": "rows",
                            "limit": 20,
                        },
                        "reason": "preview one canonical dataset by id",
                    }
                ),
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_datasets_preview",
        title="Dataset preview",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["datasets", "datasets_preview"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=0,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="Datasets payload loaded.",
            remaining_budget={"turns_used": 1, "turns_remaining": 2, "tool_calls_used": 1, "tool_calls_remaining": 1},
            checkpoint_summary="dataset list reviewed",
            active_operation={},
        )
    )

    assert action.action_type == "tool_call"
    assert action.tool == "datasets_preview"
    assert action.arguments["dataset_id"] == "BTCUSDT_4h"
    assert action.arguments["view"] == "rows"


def test_worker_decision_service_corrects_features_analytics_retry_after_not_ready(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "features_analytics",
                        "arguments": {
                            "action": "analytics",
                            "feature_name": "cf_vol_imbalance_12",
                            "symbol": "BTCUSDT",
                            "anchor_timeframe": "1h",
                        },
                        "reason": "retry analytics immediately",
                    }
                ),
                json.dumps(
                    {
                        "type": "checkpoint",
                        "status": "blocked",
                        "summary": "Stored analytics are not ready yet, so further analytics calls should wait for readiness or a different discovery step.",
                    }
                ),
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_features_analytics",
        title="Analytics readiness",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["features_analytics", "features_catalog"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=1,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="Stored analytics are not ready for feature 'cf_vol_imbalance_12' on BTCUSDT 1h.",
            remaining_budget={"turns_used": 1, "turns_remaining": 2, "tool_calls_used": 1, "tool_calls_remaining": 1},
            checkpoint_summary="analytics probe failed",
            active_operation={},
        )
    )

    assert action.action_type == "checkpoint"
    assert action.status == "blocked"


def test_worker_decision_service_corrects_research_search_snapshot_id_project_misuse(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "research_search",
                        "arguments": {
                            "query": "volume microstructure institutional activity",
                            "project_id": "active-signal-v1",
                            "level": "normal",
                            "limit": 10,
                        },
                        "reason": "search baseline-linked prior research",
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "research_search",
                        "arguments": {
                            "query": "volume microstructure institutional activity",
                            "level": "normal",
                            "limit": 10,
                        },
                        "reason": "search prior research without inventing a project id",
                    }
                ),
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_2",
        title="Volume microstructure",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["research_search"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=0,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={"baseline_snapshot_id": "active-signal-v1"},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="features_catalog loaded successfully",
            remaining_budget={"turns_used": 0, "turns_remaining": 3, "tool_calls_used": 0, "tool_calls_remaining": 2},
            checkpoint_summary="",
            active_operation={},
        )
    )

    assert action.action_type == "tool_call"
    assert action.tool == "research_search"
    assert "project_id" not in action.arguments


def test_worker_decision_service_corrects_features_analytics_without_selector(tmp_path) -> None:
    service = WorkerDecisionService(
        adapter=_StubAdapter(
            [
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "features_analytics",
                        "arguments": {
                            "action": "heatmap",
                            "symbol": "BTCUSDT",
                            "anchor_timeframe": "1h",
                            "bucket_count": 20,
                        },
                        "reason": "probe orthogonal candidates broadly",
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": "features_catalog",
                        "arguments": {"scope": "timeframe", "timeframe": "1h"},
                        "reason": "first identify one concrete candidate feature before analytics",
                    }
                ),
            ]
        ),
        artifact_store=ExecutionArtifactStore(tmp_path / "plans", run_id="run-1"),
        timeout_seconds=5,
        retry_attempts=1,
        retry_backoff_seconds=0.01,
    )
    slice_obj = PlanSlice(
        slice_id="slice_2",
        title="Analytics",
        hypothesis="h",
        objective="o",
        success_criteria=["x"],
        allowed_tools=["features_analytics", "features_catalog"],
        evidence_requirements=["x"],
        policy_tags=["cheap_first"],
        max_turns=3,
        max_tool_calls=2,
        max_expensive_calls=1,
        parallel_slot=1,
    )

    action = asyncio.run(
        service.choose_action(
            plan_id="plan_1",
            slice_obj=slice_obj,
            baseline_bootstrap={},
            known_facts={},
            recent_turn_summaries=[],
            latest_tool_summary="Feature catalog facade loaded.",
            remaining_budget={"turns_used": 1, "turns_remaining": 2, "tool_calls_used": 1, "tool_calls_remaining": 1},
            checkpoint_summary="catalog review started",
            active_operation={},
        )
    )

    assert action.action_type == "tool_call"
    assert action.tool == "features_catalog"
