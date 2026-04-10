from __future__ import annotations

import json

from app.services.direct_execution.semantic_progress import (
    build_auto_final_report,
    build_semantic_loop_abort,
    build_watchdog_checkpoint,
    compact_tool_result_message,
    derive_research_write_facts,
    should_auto_finalize_research_slice,
    tool_call_signature,
    _extract_ids,
)


def _ok_payload(summary: str = "Completed research record action create.", project_id: str = "proj_1") -> dict:
    return {
        "ok": True,
        "payload": {
            "structuredContent": {
                "status": "ok",
                "summary": summary,
                "data": {"project_id": project_id, "node_id": "node_1"},
                "warnings": [],
                "error": None,
            }
        },
    }


def test_tool_call_signature_matches_same_research_record_write() -> None:
    args_a = {
        "action": "create",
        "kind": "result",
        "project_id": "proj_1",
        "record": {"title": "Final Report", "summary": "Done"},
    }
    args_b = {
        "action": "create",
        "kind": "result",
        "project_id": "proj_1",
        "record": {"title": "Final Report", "summary": "Done"},
    }

    assert tool_call_signature("research_record", args_a) == tool_call_signature("research_record", args_b)


def test_should_auto_finalize_research_slice_on_success_marker() -> None:
    arguments = {
        "action": "create",
        "kind": "result",
        "project_id": "proj_1",
        "record": {
            "title": "Final Report",
            "summary": "Wave-1 short-list complete. Success criteria met.",
            "metadata": {"success_criteria_met": True, "evidence_refs": ["node_1"]},
        },
    }

    assert should_auto_finalize_research_slice(
        tool_name="research_record",
        arguments=arguments,
        result_payload=_ok_payload(),
        success_criteria=["Wave-1 short-list exists"],
        allowed_tools={"research_record", "research_map", "research_search"},
        required_output_facts=["research.project_id", "research.hypothesis_refs"],
        prior_contract_issue=False,
    ) is True


def test_should_not_auto_finalize_without_required_output_facts() -> None:
    arguments = {
        "action": "create",
        "kind": "result",
        "project_id": "proj_1",
        "record": {
            "title": "Final Report",
            "summary": "Success criteria met.",
            "metadata": {"success_criteria_met": True},
        },
    }

    assert should_auto_finalize_research_slice(
        tool_name="research_record",
        arguments=arguments,
        result_payload=_ok_payload(project_id=""),
        success_criteria=["Wave-1 short-list exists"],
        allowed_tools={"research_record", "research_map", "research_search"},
        required_output_facts=["research.project_id", "research.hypothesis_refs", "research.shortlist_families"],
        prior_contract_issue=False,
    ) is False


def test_should_auto_finalize_even_after_prior_contract_issue_when_success_marker_present() -> None:
    arguments = {
        "action": "create",
        "kind": "result",
        "project_id": "proj_1",
        "record": {
            "title": "Final Report",
            "summary": "Success criteria met.",
            "metadata": {"success_criteria_met": True, "evidence_refs": ["node_1"]},
        },
    }

    assert should_auto_finalize_research_slice(
        tool_name="research_record",
        arguments=arguments,
        result_payload=_ok_payload(),
        success_criteria=["Wave-1 short-list exists"],
        allowed_tools={"research_record", "research_map", "research_search"},
        required_output_facts=["research.project_id", "research.hypothesis_refs"],
        prior_contract_issue=True,
    ) is True


def test_should_auto_finalize_on_terminal_evidence_marker_without_exact_criteria_match() -> None:
    arguments = {
        "action": "create",
        "kind": "milestone",
        "project_id": "proj_1",
        "record": {
            "title": "Wave-1 Shortlist - Terminal Evidence",
            "summary": "Terminal evidence recorded for shortlist completion.",
            "metadata": {"evidence_refs": ["node_1"]},
        },
    }

    assert should_auto_finalize_research_slice(
        tool_name="research_record",
        arguments=arguments,
        result_payload=_ok_payload(summary="Research record operation loaded."),
        success_criteria=["Each shortlisted hypothesis must include novelty justification text"],
        allowed_tools={"research_record", "research_map", "research_search"},
        required_output_facts=["research.project_id", "research.hypothesis_refs"],
        prior_contract_issue=False,
    ) is True


def test_compact_tool_result_message_is_small_and_actionable() -> None:
    arguments = {
        "action": "create",
        "kind": "milestone",
        "project_id": "proj_1",
        "record": {"title": "Final Report Milestone", "summary": "Success criteria met."},
    }

    message = compact_tool_result_message(
        tool_name="research_record",
        arguments=arguments,
        result_payload=_ok_payload(),
        success_criteria=["Wave-1 short-list exists"],
    )
    payload = json.loads(message)

    assert payload["tool"] == "research_record"
    assert payload["status"] == "ok"
    assert "runtime_guidance" in payload
    assert len(message) < 1200


def test_build_auto_final_report_returns_terminal_final_report_json() -> None:
    payload = build_auto_final_report(
        arguments={
            "action": "create",
            "kind": "result",
            "project_id": "proj_1",
            "record": {
                "title": "Final Report",
                "summary": "Wave-1 short-list complete. Success criteria met.",
                "metadata": {"evidence_refs": ["node_1"]},
            },
        },
        result_payload=_ok_payload(),
        success_criteria=["Wave-1 short-list exists"],
    )

    assert "\"type\": \"final_report\"" in payload
    assert "\"verdict\": \"COMPLETE\"" in payload


def test_derive_research_write_facts_extracts_downstream_ready_fields() -> None:
    facts = derive_research_write_facts(
        arguments={
            "action": "create",
            "kind": "milestone",
            "project_id": "proj_1",
            "record": {
                "title": "Final Report Milestone",
                "summary": "Signal families: funding divergence, volatility shock",
                "metadata": {"evidence_refs": ["node_1", "node_2"]},
            },
        },
        result_payload=_ok_payload(),
    )

    assert facts["research.project_id"] == "proj_1"
    assert facts["research.hypothesis_refs"] == ["node_1"]
    assert "funding divergence" in facts["research.shortlist_families"][0]


def test_build_semantic_loop_abort_returns_soft_abort_payload() -> None:
    payload = build_semantic_loop_abort(summary="loop detected")

    assert "\"type\": \"abort\"" in payload
    assert "\"reason_code\": \"direct_semantic_loop_detected\"" in payload


def test_build_watchdog_checkpoint_returns_blocked_checkpoint() -> None:
    payload = build_watchdog_checkpoint(summary="stalled", reason_code="direct_model_stalled_before_first_action")

    assert "\"type\": \"checkpoint\"" in payload
    assert "\"status\": \"blocked\"" in payload
    assert "\"reason_code\": \"direct_model_stalled_before_first_action\"" in payload


# ---------------------------------------------------------------------------
# _extract_ids — nested structure handling
# ---------------------------------------------------------------------------


def test_extract_ids_finds_project_id_in_nested_data_project() -> None:
    """MCP returns data.project.project_id — _extract_ids must find it."""
    result_payload = {
        "ok": True,
        "payload": {
            "content": [{"type": "text", "text": json.dumps({
                "status": "ok",
                "data": {
                    "action": "create",
                    "project": {
                        "project_id": "compiled-plan-v1-batch-1-22ef70ed",
                        "root_node_id": "project-node-abc",
                        "default_branch_id": "branch-xyz",
                    },
                },
            })}]
        },
    }
    ids = _extract_ids(result_payload)
    assert "compiled-plan-v1-batch-1-22ef70ed" in ids
    assert "project-node-abc" in ids


def test_extract_ids_finds_operation_id_in_nested_data_record() -> None:
    """MCP returns data.record.operation_id — _extract_ids must find it."""
    result_payload = {
        "ok": True,
        "payload": {
            "content": [{"type": "text", "text": json.dumps({
                "status": "ok",
                "data": {
                    "record": {
                        "node_id": "node-123",
                        "operation_id": "op-456",
                    },
                },
            })}]
        },
    }
    ids = _extract_ids(result_payload)
    assert "node-123" in ids
    assert "op-456" in ids
