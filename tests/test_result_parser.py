"""Tests for result parsing and validation."""

import json

from app.models import PlannerDecision, TaskResult
from app.result_parser import (
    is_duplicate_result,
    is_useless_result,
    parse_planner_output,
    parse_task_report,
    parse_worker_output,
)


# --- Planner output parsing ---

def test_parse_valid_planner_json():
    raw = json.dumps({
        "decision": "launch_worker",
        "target_worker_id": "qwen-1",
        "task_instruction": "Write code",
        "reason": "Need to start",
        "check_after_seconds": 60,
        "memory_update": "starting",
        "should_finish": False,
        "final_summary": "",
    })
    output = parse_planner_output(raw)
    assert output.decision == PlannerDecision.LAUNCH_WORKER
    assert output.target_worker_id == "qwen-1"
    assert output.task_instruction == "Write code"


def test_parse_planner_with_markdown_fences():
    raw = '```json\n{"decision": "wait", "reason": "still working"}\n```'
    output = parse_planner_output(raw)
    assert output.decision == PlannerDecision.WAIT


def test_parse_planner_with_extra_text():
    raw = 'Here is my analysis:\n{"decision": "finish", "should_finish": true, "final_summary": "Done"}\nThat is all.'
    output = parse_planner_output(raw)
    assert output.decision == PlannerDecision.FINISH
    assert output.should_finish is True


def test_parse_planner_invalid_json():
    raw = "This is not JSON at all"
    output = parse_planner_output(raw)
    assert output.decision == PlannerDecision.WAIT  # safe default
    assert "Failed to parse" in output.reason


def test_parse_planner_unknown_decision():
    raw = json.dumps({"decision": "fly_to_moon"})
    output = parse_planner_output(raw)
    assert output.decision == PlannerDecision.WAIT  # safe default


def test_parse_planner_partial_json():
    raw = json.dumps({"decision": "launch_worker"})  # missing fields
    output = parse_planner_output(raw)
    assert output.decision == PlannerDecision.LAUNCH_WORKER
    assert output.target_worker_id is None
    assert output.check_after_seconds == 300  # default


# --- Worker output parsing ---

def test_parse_valid_worker_json():
    raw = json.dumps({
        "status": "success",
        "summary": "Created files",
        "artifacts": ["a.py", "b.py"],
        "confidence": 0.9,
        "error": "",
    })
    result = parse_worker_output(raw, task_id="t1", worker_id="w1")
    assert result.status == "success"
    assert result.summary == "Created files"
    assert len(result.artifacts) == 2
    assert result.confidence == 0.9


def test_parse_worker_error_response():
    raw = json.dumps({"status": "error", "error": "Something broke"})
    result = parse_worker_output(raw, task_id="t1", worker_id="w1")
    assert result.status == "error"
    assert "Something broke" in result.error


def test_parse_worker_no_json():
    raw = "Just plain text output"
    result = parse_worker_output(raw, task_id="t1", worker_id="w1")
    assert result.status == "error"
    assert "No parseable JSON" in result.error


def test_parse_worker_invalid_status():
    raw = json.dumps({"status": "flying"})
    result = parse_worker_output(raw, task_id="t1", worker_id="w1")
    assert result.status == "error"  # normalized


def test_parse_worker_preserves_full_raw_output():
    payload = {
        "status": "success",
        "summary": "x" * 2500,
        "artifacts": [],
        "confidence": 0.9,
        "error": "",
    }
    raw = json.dumps(payload)
    result = parse_worker_output(raw, task_id="t1", worker_id="w1")
    assert result.raw_output == raw


def test_parse_plan_task_report_with_preamble_and_fenced_json():
    payload = {
        "status": "success",
        "what_was_requested": "run ETAP 4",
        "what_was_done": "completed ETAP 4 with funding filter",
        "results_table": [
            {"run_id": "run-1", "net_pnl": 10, "verdict": "PROMOTE"},
            {"run_id": "run-2", "net_pnl": 5, "verdict": "BASELINE"},
        ],
        "key_metrics": {"sharpe": 1.99, "trades": 185},
        "artifacts": ["run-1", "snapshot:v2-funding@2"],
        "verdict": "PROMOTE",
        "confidence": 0.85,
        "error": "",
        "mcp_problems": [],
    }
    raw = (
        "Все шаги выполнены. Финальный результат ниже.\n\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "```"
    )
    report = parse_task_report(raw, task_id="t1", worker_id="w1", plan_version=1)
    assert report.status == "success"
    assert report.verdict == "PROMOTE"
    assert report.what_was_done == payload["what_was_done"]
    assert len(report.results_table) == 2
    assert report.raw_output == raw


def test_parse_plan_task_report_rejects_truncated_nested_fragment():
    payload = {
        "status": "success",
        "what_was_requested": "run ETAP 1",
        "what_was_done": "completed ETAP 1",
        "results_table": [
            {"run_id": "run-1", "snapshot_id": "base", "net_pnl": 1, "verdict": "BASELINE"},
            {"run_id": "run-2", "snapshot_id": "variant", "net_pnl": 2, "verdict": "WATCHLIST"},
        ],
        "key_metrics": {"net_pnl": 2, "sharpe": 0.9},
        "artifacts": ["run-2"],
        "verdict": "WATCHLIST",
        "confidence": 0.8,
        "error": "",
        "mcp_problems": [],
    }
    raw = (
        "Now let me compile the final results JSON:\n\n```json\n"
        f"{json.dumps(payload)}\n```"
    )
    truncated = raw[:220]
    report = parse_task_report(truncated, task_id="t1", worker_id="w1", plan_version=1)
    assert report.status == "error"
    assert "No parseable JSON" in report.error


# --- Duplicate/useless detection ---

def test_duplicate_detection():
    r1 = TaskResult(task_id="t1", worker_id="w1", status="success", summary="Same")
    r2 = TaskResult(task_id="t1", worker_id="w1", status="success", summary="Same")
    assert is_duplicate_result(r1, r2)


def test_not_duplicate_different_summary():
    r1 = TaskResult(task_id="t1", worker_id="w1", status="success", summary="First")
    r2 = TaskResult(task_id="t1", worker_id="w1", status="success", summary="Second")
    assert not is_duplicate_result(r1, r2)


def test_not_duplicate_none_previous():
    r = TaskResult(task_id="t1", worker_id="w1", status="success", summary="X")
    assert not is_duplicate_result(None, r)


def test_useless_result_empty():
    r = TaskResult(task_id="t1", worker_id="w1", status="success")
    assert r.is_empty
    assert is_useless_result(r)


def test_useless_result_error_no_message():
    r = TaskResult(task_id="t1", worker_id="w1", status="error", error="")
    assert is_useless_result(r)


def test_not_useless_has_content():
    r = TaskResult(task_id="t1", worker_id="w1", status="success", summary="Did stuff")
    assert not is_useless_result(r)
