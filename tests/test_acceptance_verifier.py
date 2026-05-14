from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.acceptance import accepted_completion_state, dependency_unblocked_by
from app.services.direct_execution.acceptance.contracts import extract_proof, is_infra_error, proof_refs, check_infra_route, FAIL, PASS, PredicateResult
from app.services.direct_execution.acceptance.proof_client import _extract_structured_payload
from app.services.direct_execution.acceptance.subjects import node_ids_from_action_and_transcript
from app.services.direct_execution.acceptance.verifier import AcceptanceVerifier
from app.services.direct_execution.executor import DirectExecutionResult
from app.services.direct_execution.fallback_executor import FallbackExecutor


class _FakeProofClient:
    def __init__(self, responses: dict[str, dict[str, Any]], *, nested_envelope: bool = False) -> None:
        self.responses = responses
        self.nested_envelope = nested_envelope
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_FakeProofClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def call_proof(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"tool": tool_name, "arguments": dict(arguments)})
        key = f"{tool_name}:{arguments.get('run_id') or arguments.get('node_id') or arguments.get('name') or arguments.get('dataset_id') or arguments.get('model_id') or '*'}"
        response = self.responses.get(key, self.responses.get(tool_name, {"proof": {"status": "fail", "blocking_reasons": ["missing_fake_response"]}}))
        if self.nested_envelope:
            return {"status": "ok", "data": response}
        return response


def _slice(**overrides: Any) -> PlanSlice:
    defaults = dict(
        slice_id="slice_1",
        title="slice",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=4,
        max_expensive_calls=0,
        parallel_slot=1,
    )
    defaults.update(overrides)
    return PlanSlice(**defaults)


def _action(**overrides: Any) -> WorkerAction:
    defaults = dict(
        action_id="action_1",
        action_type="final_report",
        summary="done",
        verdict="COMPLETE",
        confidence=0.8,
        evidence_refs=["transcript:1:research_memory"],
        facts={
            "direct.successful_tool_count": 1,
            "direct.successful_tool_names": ["research_memory"],
            "direct.successful_mutating_tool_count": 1,
            "direct.supported_evidence_refs": ["transcript:1:research_memory"],
        },
    )
    defaults.update(overrides)
    return WorkerAction(**defaults)


def _verifier(client: _FakeProofClient, *, incident_store: Any = None) -> AcceptanceVerifier:
    return AcceptanceVerifier(
        direct_config=SimpleNamespace(),
        proof_client_factory=lambda: client,
        incident_store=incident_store,
    )


# ---------- existing tests ----------


def test_accepted_ready_requires_passed_acceptance_proof() -> None:
    sl = _slice(status="completed", verdict="COMPLETE")

    assert accepted_completion_state(slice_obj=sl, verdict="COMPLETE") == "reported_terminal"
    assert dependency_unblocked_by(sl) is False

    sl.acceptance_proof = {"status": "pass"}
    sl.acceptance_state = accepted_completion_state(slice_obj=sl, verdict="COMPLETE")

    assert sl.acceptance_state == "accepted_ready"
    assert dependency_unblocked_by(sl) is True


def test_passed_proof_overrides_watchlist_verdict_under_strict_mode() -> None:
    # Regression: a strict-acceptance slice whose agent emitted WATCHLIST but
    # whose acceptance proof PASSED must still unblock downstream dependents.
    # Previously the state machine returned "reported_terminal" for WATCHLIST
    # even when the formal proof passed, blocking the entire downstream batch.
    sl = _slice(
        status="completed",
        verdict="WATCHLIST",
        dependency_unblock_mode="accepted_only",
        watchlist_allows_unblock=False,
    )
    sl.acceptance_proof = {"status": "pass", "blocking_reasons": []}

    state = accepted_completion_state(slice_obj=sl, verdict="WATCHLIST")
    sl.acceptance_state = state

    assert state == "accepted_ready"
    assert dependency_unblocked_by(sl) is True


def test_failed_proof_keeps_reported_terminal_even_with_complete_verdict() -> None:
    sl = _slice(status="completed", verdict="COMPLETE")
    sl.acceptance_proof = {"status": "fail", "blocking_reasons": ["missing_evidence"]}

    state = accepted_completion_state(slice_obj=sl, verdict="COMPLETE")

    assert state == "reported_terminal"
    sl.acceptance_state = state
    assert dependency_unblocked_by(sl) is False


def test_passed_proof_yields_advisory_only_done_under_advisory_mode() -> None:
    sl = _slice(
        status="completed",
        verdict="WATCHLIST",
        dependency_unblock_mode="advisory_only",
        watchlist_allows_unblock=False,
    )
    sl.acceptance_proof = {"status": "pass"}

    state = accepted_completion_state(slice_obj=sl, verdict="WATCHLIST")

    assert state == "advisory_only_done"
    sl.acceptance_state = state
    assert dependency_unblocked_by(sl) is True


def test_standalone_backtests_accepts_only_mcp_proof_pass() -> None:
    client = _FakeProofClient(
        {
            "backtests_runs:20260413-143910-3db43a14": {
                "proof": {
                    "status": "pass",
                    "predicates": [
                        {"id": "run_resolves", "status": "pass"},
                        {"id": "compute_terminal_completed", "status": "pass"},
                        {"id": "summary_metrics_present", "status": "pass"},
                    ],
                    "evidence_refs": ["mcp://backtests_runs/20260413-143910-3db43a14#acceptance_proof"],
                }
            }
        }
    )
    sl = _slice(
        allowed_tools=["backtests_plan", "backtests_runs"],
        runtime_profile="generic_mutation",
        requires_persisted_artifact=True,
        requires_live_handle_validation=True,
    )
    action = _action(
        evidence_refs=["20260413-143910-3db43a14"],
        facts={
            "run_id": "20260413-143910-3db43a14",
            "direct.successful_tool_count": 2,
            "direct.successful_tool_names": ["backtests_plan", "backtests_runs"],
            "direct.supported_evidence_refs": ["20260413-143910-3db43a14"],
        },
    )

    result = asyncio.run(_verifier(client).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is True
    assert client.calls[0]["tool"] == "backtests_runs"


def test_standalone_backtests_contradiction_blocks_before_mcp() -> None:
    client = _FakeProofClient({})
    sl = _slice(allowed_tools=["backtests_runs"], runtime_profile="generic_mutation")
    action = _action(
        summary="Candidates survived but no saved/active runs exist, cannot evaluate standalone backtest runs.",
        evidence_refs=["20260413-143910-3db43a14"],
        facts={"run_id": "20260413-143910-3db43a14"},
    )

    result = asyncio.run(_verifier(client).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is False
    assert result.route == "repair_only"
    assert result.blocking_reasons[0].startswith("forbidden_contradiction:")
    assert client.calls == []


def test_write_result_requires_persisted_research_node_proof() -> None:
    client = _FakeProofClient({})
    sl = _slice(runtime_profile="write_result", allowed_tools=["research_memory"])
    action = _action(facts={"direct.successful_mutating_tool_count": 1, "direct.successful_tool_count": 1})

    result = asyncio.run(_verifier(client).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is False
    assert "research_node_subject_present" in result.blocking_reasons


def test_write_result_accepts_persisted_research_node_proof() -> None:
    client = _FakeProofClient({"research_memory:node_abc": {"proof": {"status": "pass", "evidence_refs": ["mcp://research_memory/node_abc#prove"]}}})
    sl = _slice(runtime_profile="write_result", allowed_tools=["research_memory"])
    action = _action(facts={"node_id": "node_abc", "direct.successful_mutating_tool_count": 1, "direct.successful_tool_count": 1})

    result = asyncio.run(_verifier(client).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is True
    assert client.calls[0]["tool"] == "research_memory"


def test_feature_construction_requires_feature_proof() -> None:
    client = _FakeProofClient({"features_custom:cf_alpha": {"proof": {"status": "pass"}}})
    sl = _slice(
        title="Feature construction",
        objective="publish custom feature",
        allowed_tools=["features_custom", "features_dataset"],
    )
    action = _action(facts={"feature_name": "cf_alpha", "direct.successful_tool_names": ["features_custom"]})

    result = asyncio.run(_verifier(client).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is True
    assert client.calls[0]["tool"] == "features_custom"


def test_model_training_requires_dataset_and_registry_proofs() -> None:
    client = _FakeProofClient(
        {
            "models_dataset:ds_1": {"proof": {"status": "pass"}},
            "models_registry:model_1": {"proof": {"status": "pass"}},
        }
    )
    sl = _slice(
        title="Model training",
        objective="train model",
        allowed_tools=["models_dataset", "models_train", "models_registry"],
    )
    action = _action(facts={"dataset_id": "ds_1", "model_id": "model_1", "version": "v1"})

    result = asyncio.run(_verifier(client).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is True
    assert [call["tool"] for call in client.calls] == ["models_dataset", "models_registry"]


def test_fallback_router_hard_blocks_acceptance_mcp_infra_without_next_provider() -> None:
    result = DirectExecutionResult(
        action=_action(),
        artifact_path="",
        raw_output="",
        acceptance_result={
            "status": "fail",
            "route": "hard_block_infra",
            "blocking_reasons": ["acceptance_mcp_infra_unavailable"],
        },
    )

    assert FallbackExecutor._result_route(result) == "hard_block_infra"


# ---------- new tests: proof envelope unwrapping ----------


def test_extract_proof_unwraps_nested_envelope() -> None:
    payload = {
        "status": "ok",
        "data": {
            "proof": {
                "status": "fail",
                "error": "AttributeError: bad method",
                "predicates": [{"id": "p1", "status": "fail"}],
            }
        },
    }
    proof = extract_proof(payload)
    assert proof["status"] == "fail"
    assert "AttributeError" in proof["error"]
    assert len(proof["predicates"]) == 1
    assert is_infra_error(proof) is True


def test_extract_proof_direct_envelope() -> None:
    payload = {"proof": {"status": "pass", "evidence_refs": ["mcp://x"]}}
    proof = extract_proof(payload)
    assert proof["status"] == "pass"
    assert proof["evidence_refs"] == ["mcp://x"]
    assert is_infra_error(proof) is False


def test_extract_proof_flat_envelope() -> None:
    payload = {"status": "pass"}
    proof = extract_proof(payload)
    assert proof["status"] == "pass"


def test_extract_proof_non_dict_returns_defaults() -> None:
    proof = extract_proof("not a dict")
    assert proof["status"] == ""
    assert proof["predicates"] == []


def test_is_infra_error_detects_exception_signatures() -> None:
    assert is_infra_error({"status": "fail", "error": "AttributeError: no attr"}) is True
    assert is_infra_error({"status": "fail", "error": "RuntimeError: timeout"}) is True
    assert is_infra_error({"status": "fail", "error": "ConnectionError: refused"}) is True
    assert is_infra_error({"status": "fail", "error_type": "ImportError"}) is True


def test_is_infra_error_pass_is_never_infra() -> None:
    assert is_infra_error({"status": "pass", "error": "AttributeError: something"}) is False


def test_is_infra_error_plain_fail_is_not_infra() -> None:
    assert is_infra_error({"status": "fail", "error": ""}) is False
    assert is_infra_error({"status": "fail", "error": "node does not exist"}) is False


def test_proof_refs_extracts_from_dict() -> None:
    assert proof_refs({"evidence_refs": ["a", "b"]}) == ["a", "b"]
    assert proof_refs({"evidence_refs": None}) == []
    assert proof_refs({"no_key": 1}) == []
    assert proof_refs("not a dict") == []


def test_check_infra_route_triggers_after_threshold() -> None:
    p1 = PredicateResult(id="p1", status=FAIL, details={"infra_error": "AttributeError"})
    p2 = PredicateResult(id="p2", status=FAIL, details={"infra_error": "AttributeError"})

    route, count = check_infra_route([p1], 0)
    assert route == ""
    assert count == 1

    route, count = check_infra_route([p1, p2], count)
    assert route == "hard_block_infra"
    assert count == 2


def test_check_infra_route_resets_on_non_infra() -> None:
    p1 = PredicateResult(id="p1", status=FAIL, details={"infra_error": "AttributeError"})
    p2 = PredicateResult(id="p2", status=PASS, details={})

    route, count = check_infra_route([p1, p2], 1)
    assert route == ""
    assert count == 0


# ---------- new tests: integration with nested envelope ----------


def test_standalone_backtests_accepts_nested_envelope_proof() -> None:
    client = _FakeProofClient(
        {
            "backtests_runs:20260413-143910-3db43a14": {
                "proof": {
                    "status": "pass",
                    "predicates": [
                        {"id": "run_resolves", "status": "pass"},
                        {"id": "compute_terminal_completed", "status": "pass"},
                        {"id": "summary_metrics_present", "status": "pass"},
                    ],
                }
            }
        },
        nested_envelope=True,
    )
    sl = _slice(
        allowed_tools=["backtests_plan", "backtests_runs"],
        runtime_profile="generic_mutation",
        requires_persisted_artifact=True,
        requires_live_handle_validation=True,
    )
    action = _action(
        evidence_refs=["20260413-143910-3db43a14"],
        facts={
            "run_id": "20260413-143910-3db43a14",
            "direct.successful_tool_count": 2,
            "direct.successful_tool_names": ["backtests_plan", "backtests_runs"],
            "direct.supported_evidence_refs": ["20260413-143910-3db43a14"],
        },
    )

    result = asyncio.run(_verifier(client).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is True


def test_research_node_proof_detects_infra_error_in_nested_envelope() -> None:
    client = _FakeProofClient(
        {
            "research_memory:node_abc": {
                "proof": {
                    "status": "fail",
                    "error": "AttributeError: 'ResearchRecordService' object has no attribute 'get_node'",
                }
            }
        },
        nested_envelope=True,
    )
    sl = _slice(runtime_profile="write_result", allowed_tools=["research_memory"])
    action = _action(facts={"node_id": "node_abc", "direct.successful_mutating_tool_count": 1, "direct.successful_tool_count": 1})

    result = asyncio.run(_verifier(client).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is False
    assert result.route == "hard_block_infra"
    assert "research_node_proof_pass" in result.blocking_reasons
    proof_predicate = [p for p in result.predicates if p.id == "research_node_proof_pass"][0]
    assert "infra_error" in proof_predicate.details
    assert "AttributeError" in proof_predicate.details["infra_error"]


def test_research_infra_error_fast_fail() -> None:
    client = _FakeProofClient(
        {
            "research_memory:node_1": {
                "proof": {
                    "status": "fail",
                    "error": "AttributeError: 'ResearchRecordService' object has no attribute 'get_node'",
                }
            },
            "research_memory:node_2": {
                "proof": {
                    "status": "fail",
                    "error": "AttributeError: 'ResearchRecordService' object has no attribute 'get_node'",
                }
            },
            "research_memory:node_3": {
                "proof": {"status": "pass"},
            },
        },
        nested_envelope=True,
    )
    sl = _slice(runtime_profile="write_result", allowed_tools=["research_memory"])
    action = _action(
        facts={
            "node_id": "node_1",
            "direct.created_ids": ["node_1", "node_2", "node_3"],
            "direct.successful_mutating_tool_count": 1,
            "direct.successful_tool_count": 1,
        },
    )

    result = asyncio.run(_verifier(client).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is False
    assert result.route == "hard_block_infra"
    # node_3 should NOT have been called (fast-fail after 2 consecutive infra errors)
    assert len(client.calls) == 2


def test_feature_construction_with_nested_envelope() -> None:
    client = _FakeProofClient(
        {"features_custom:cf_alpha": {"proof": {"status": "pass"}}},
        nested_envelope=True,
    )
    sl = _slice(
        title="Feature construction",
        objective="publish custom feature",
        allowed_tools=["features_custom", "features_dataset"],
    )
    action = _action(facts={"feature_name": "cf_alpha", "direct.successful_tool_names": ["features_custom"]})

    result = asyncio.run(_verifier(client).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is True


def test_model_training_with_nested_envelope() -> None:
    client = _FakeProofClient(
        {
            "models_dataset:ds_1": {"proof": {"status": "pass"}},
            "models_registry:model_1": {"proof": {"status": "pass"}},
        },
        nested_envelope=True,
    )
    sl = _slice(
        title="Model training",
        objective="train model",
        allowed_tools=["models_dataset", "models_train", "models_registry"],
    )
    action = _action(facts={"dataset_id": "ds_1", "model_id": "model_1", "version": "v1"})

    result = asyncio.run(_verifier(client).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is True
    assert [call["tool"] for call in client.calls] == ["models_dataset", "models_registry"]


class _FakeIncidentStore:
    def __init__(self) -> None:
        self.incidents: list[dict[str, Any]] = []

    def record(self, *, summary: str, metadata: dict[str, Any], source: str, severity: str) -> None:
        self.incidents.append({"summary": summary, "metadata": metadata, "source": source, "severity": severity})


def test_inner_proof_infra_error_records_incident() -> None:
    client = _FakeProofClient(
        {
            "research_memory:node_1": {
                "proof": {
                    "status": "fail",
                    "error": "RuntimeError: database unavailable",
                }
            }
        },
        nested_envelope=True,
    )
    incidents = _FakeIncidentStore()
    sl = _slice(runtime_profile="write_result", allowed_tools=["research_memory"])
    action = _action(facts={"node_id": "node_1", "direct.successful_mutating_tool_count": 1, "direct.successful_tool_count": 1})

    result = asyncio.run(_verifier(client, incident_store=incidents).verify(plan=None, slice_obj=sl, action=action, transcript=[], known_facts={}, required_output_facts=[]))

    assert result.passed is False
    assert result.route == "hard_block_infra"
    assert len(incidents.incidents) == 1
    assert incidents.incidents[0]["metadata"]["reason"] == "acceptance_inner_proof_infra_error"
    assert "RuntimeError" in incidents.incidents[0]["metadata"]["infra_errors"][0]


# ---------- node ID extraction: regex boundary fix ----------


def test_node_id_regex_excludes_composite_project_ids() -> None:
    action = _action(
        facts={
            "project_id": "project-node-6958986f557c4fd39aaee8b802258c83",
            "node_id": "node_abc123",
        },
    )
    ids = node_ids_from_action_and_transcript(action, [])
    assert "node-6958986f557c4fd39aaee8b802258c83" not in ids
    assert "node_abc123" in ids


def test_node_id_regex_excludes_branch_node_prefix() -> None:
    action = _action(
        facts={
            "branch_id": "branch-node-deadbeef1234",
            "node_id": "node_correct",
        },
    )
    ids = node_ids_from_action_and_transcript(action, [])
    assert "node-deadbeef1234" not in ids
    assert "node_correct" in ids


def test_node_id_regex_accepts_standalone_hyphen_node() -> None:
    action = _action(
        facts={
            "node_id": "node-6958986f557c4fd39aaee8b802258c83",
        },
    )
    ids = node_ids_from_action_and_transcript(action, [])
    assert "node-6958986f557c4fd39aaee8b802258c83" in ids


def test_node_id_regex_scans_transcript_without_phantom_matches() -> None:
    action = _action(facts={"node_id": "node_real"})
    transcript = [
        {
            "tool": "research_memory",
            "arguments": {"project_id": "project-node-phantom123"},
            "payload": {"created": "project-node-phantom123"},
        },
    ]
    ids = node_ids_from_action_and_transcript(action, transcript)
    assert "node-phantom123" not in ids
    assert "node_real" in ids


def test_node_id_regex_excludes_json_field_names() -> None:
    """JSON keys like node_type, node_refs, node-id must not be extracted as node IDs."""
    action = _action(facts={"node_id": "node_real"})
    transcript = [
        {
            "tool": "research_memory",
            "arguments": {},
            "payload": {
                "node_type": "hypothesis",
                "node_refs": [],
                "node-id": "some-value",
                "node_ids": [],
                "node_count": 5,
                "record": {"node_id": "node-3a4e8cef2f264eaa8e065b6a15a960f8"},
            },
        },
    ]
    ids = node_ids_from_action_and_transcript(action, transcript)
    # JSON field names must be excluded
    assert "node_type" not in ids
    assert "node_refs" not in ids
    assert "node-id" not in ids
    assert "node_ids" not in ids
    assert "node_count" not in ids
    # But real node IDs from values must be preserved
    assert "node_real" in ids
    assert "node-3a4e8cef2f264eaa8e065b6a15a960f8" in ids


# ---------- _extract_structured_payload: unit tests for all payload formats ----------


def test_extract_structured_payload_parses_mcp_sdk_json_text() -> None:
    """Standard MCP SDK format: content[].text contains a JSON string."""
    raw = {
        "content": [{"type": "text", "text": '{"status":"ok","data":{"proof":{"status":"pass"}}}'}],
        "isError": False,
    }
    result = _extract_structured_payload(raw)
    assert result == {"status": "ok", "data": {"proof": {"status": "pass"}}}


def test_extract_structured_payload_parses_claude_structured_content() -> None:
    """Claude MCP protocol: structuredContent at top level."""
    raw = {"structuredContent": {"status": "ok", "data": {"proof": {"status": "pass"}}}}
    result = _extract_structured_payload(raw)
    assert result == {"status": "ok", "data": {"proof": {"status": "pass"}}}


def test_extract_structured_payload_parses_nested_claude_content() -> None:
    """Claude MCP protocol: structuredContent nested inside content[]."""
    raw = {"content": [{"structuredContent": {"status": "ok", "data": {"proof": {"status": "pass"}}}}]}
    result = _extract_structured_payload(raw)
    assert result == {"status": "ok", "data": {"proof": {"status": "pass"}}}


def test_extract_structured_payload_recurses_into_payload() -> None:
    """Recursive unwrapping: raw has a 'payload' key containing structuredContent."""
    raw = {"payload": {"structuredContent": {"status": "ok", "data": {"proof": {"status": "pass"}}}}}
    result = _extract_structured_payload(raw)
    assert result == {"status": "ok", "data": {"proof": {"status": "pass"}}}


def test_extract_structured_payload_handles_invalid_json_text() -> None:
    """Invalid JSON in content[].text falls back to returning raw dict."""
    raw = {"content": [{"type": "text", "text": "not valid json {"}]}
    result = _extract_structured_payload(raw)
    # Fallback: return raw as-is (with content and no structured match)
    assert result is raw


def test_extract_structured_payload_handles_non_text_content() -> None:
    """Non-text content (e.g. image) with no structuredContent falls back to raw."""
    raw = {"content": [{"type": "image", "data": "base64..."}]}
    result = _extract_structured_payload(raw)
    assert result is raw


def test_extract_structured_payload_non_dict_returns_raw() -> None:
    """Non-dict input returns a wrapper with 'raw' key."""
    result = _extract_structured_payload("just a string")
    assert result == {"raw": "just a string"}


# ---------- End-to-end: _extract_structured_payload + extract_proof ----------


def test_proof_extraction_end_to_end_with_mcp_sdk_format() -> None:
    """Full chain: MCP SDK content[].text -> _extract_structured_payload -> extract_proof -> pass."""
    raw = {
        "content": [{"type": "text", "text": '{"status":"ok","data":{"proof":{"status":"pass","evidence_refs":["mcp://x"]}}}'}],
        "isError": False,
    }
    payload = _extract_structured_payload(raw)
    proof = extract_proof(payload)
    assert proof["status"] == "pass"
    assert proof["evidence_refs"] == ["mcp://x"]


def test_proof_extraction_end_to_end_with_mcp_sdk_fail() -> None:
    """Full chain: MCP SDK content[].text -> _extract_structured_payload -> extract_proof -> fail."""
    raw = {
        "content": [{"type": "text", "text": '{"status":"ok","data":{"proof":{"status":"fail","blocking_reasons":["node_missing"]}}}'}],
        "isError": False,
    }
    payload = _extract_structured_payload(raw)
    proof = extract_proof(payload)
    assert proof["status"] == "fail"
    assert "node_missing" in proof["blocking_reasons"]


# ---------- _append_named_predicates required_ids contract ----------


def test_named_predicates_are_required_when_in_contract_required_predicates() -> None:
    """For standalone_backtests, each_run_matches_snapshot_symbol_timeframes is
    in required_predicates and should be marked required=True."""
    from app.services.direct_execution.acceptance.backtests import _append_named_predicates

    proof = {
        "status": "pass",
        "predicates": [
            {"id": "snapshot_matches_expected", "status": "pass"},
            {"id": "symbol_matches_expected", "status": "pass"},
            {"id": "timeframes_match_expected", "status": "pass"},
        ],
    }
    predicates: list[PredicateResult] = []
    _append_named_predicates(
        predicates, proof,
        run_id="20260416-021356-efed0832",
        required_ids=frozenset({
            "candidate_set_non_empty",
            "each_run_exists",
            "each_run_completed",
            "each_run_has_metrics",
            "each_run_matches_snapshot_symbol_timeframes",
        }),
    )
    snap_pred = next(p for p in predicates if p.id == "each_run_matches_snapshot_symbol_timeframes")
    assert snap_pred.status == "pass"
    assert snap_pred.required is True


def test_named_predicates_are_not_required_when_absent_from_contract() -> None:
    """For condition_stability_analysis, each_run_matches_snapshot_symbol_timeframes
    is NOT in required_predicates and should be marked required=False."""
    from app.services.direct_execution.acceptance.backtests import _append_named_predicates

    proof = {
        "status": "pass",
        "predicates": [
            {"id": "snapshot_matches_expected", "status": "fail"},
            {"id": "symbol_matches_expected", "status": "pass"},
            {"id": "timeframes_match_expected", "status": "pass"},
        ],
    }
    predicates: list[PredicateResult] = []
    _append_named_predicates(
        predicates, proof,
        run_id="20260416-021356-efed0832",
        required_ids=frozenset({"run_set_non_empty", "each_run_proof_pass"}),
    )
    snap_pred = next(p for p in predicates if p.id == "each_run_matches_snapshot_symbol_timeframes")
    assert snap_pred.status == "fail"
    assert snap_pred.required is False


def test_condition_stability_passes_despite_snapshot_mismatch() -> None:
    """Reproduce the stage_6 scenario: condition_stability_analysis should pass
    even when snapshot/symbol/timeframe predicates fail, because those are
    not in the contract's required_predicates."""
    client = _FakeProofClient(
        {
            "backtests_runs:20260416-021356-efed0832": {
                "proof": {
                    "status": "pass",
                    "predicates": [
                        {"id": "snapshot_matches_expected", "status": "fail"},
                        {"id": "symbol_matches_expected", "status": "pass"},
                        {"id": "timeframes_match_expected", "status": "fail"},
                    ],
                    "evidence_refs": ["mcp://backtests_runs/20260416-021356-efed0832#acceptance_proof"],
                }
            }
        }
    )
    sl = _slice(
        allowed_tools=["backtests_conditions", "backtests_analysis", "backtests_runs"],
        runtime_profile="backtests_stability_analysis",
    )
    action = _action(
        facts={"run_id": "20260416-021356-efed0832"},
    )

    result = asyncio.run(_verifier(client).verify(
        plan=None, slice_obj=sl, action=action, transcript=[],
        known_facts={}, required_output_facts=[],
    ))

    assert result.passed is True
    # The snapshot predicate should be present but not a blocker
    snap_preds = [p for p in result.predicates if p.id == "each_run_matches_snapshot_symbol_timeframes"]
    assert len(snap_preds) == 1
    assert snap_preds[0].status == "fail"
    assert "each_run_matches_snapshot_symbol_timeframes" not in result.blocking_reasons

