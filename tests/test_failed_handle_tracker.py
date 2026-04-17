"""Tests for failed_handle_tracker module."""

import json
import pytest

from app.services.direct_execution.failed_handle_tracker import (
    FailedHandleTracker,
    _is_runtime_failure,
    extract_failed_handles,
)


class TestIsRuntimeFailure:
    def test_operation_failed_token(self):
        payload = {"ok": True, "payload": {"error": {"code": "operation_failed"}}}
        assert _is_runtime_failure(payload) is True

    def test_walkforward_failed_status(self):
        payload = {
            "ok": True,
            "payload": {
                "content": [
                    {"type": "text", "text": json.dumps({
                        "message": "Walk-forward job abc ended with status 'failed' before publishing result."
                    })}
                ]
            },
        }
        assert _is_runtime_failure(payload) is True

    def test_cancelled_status(self):
        payload = {"ok": True, "payload": {"message": "Job ended with status 'cancelled'"}}
        assert _is_runtime_failure(payload) is True

    def test_timeout_status(self):
        payload = {"ok": True, "payload": {"message": "Job ended with status 'timeout'"}}
        assert _is_runtime_failure(payload) is True

    def test_success_not_failure(self):
        payload = {"ok": True, "payload": {"status": "completed", "data": {"job_id": "abc"}}}
        assert _is_runtime_failure(payload) is False

    def test_schema_error_not_failure(self):
        payload = {"ok": True, "payload": {"error": {"code": "schema_validation_failed"}}}
        assert _is_runtime_failure(payload) is False


class TestExtractFailedHandles:
    def test_extracts_job_id(self):
        args = {"job_id": "20260416-abc123", "action": "result"}
        result = extract_failed_handles(arguments=args)
        assert "20260416-abc123" in result

    def test_extracts_run_id(self):
        args = {"run_id": "run-456", "action": "detail"}
        result = extract_failed_handles(arguments=args)
        assert "run-456" in result

    def test_extracts_operation_id(self):
        args = {"operation_id": "op-789"}
        result = extract_failed_handles(arguments=args)
        assert "op-789" in result

    def test_extracts_multiple_ids(self):
        args = {"job_id": "j1", "run_id": "r1"}
        result = extract_failed_handles(arguments=args)
        assert "j1" in result
        assert "r1" in result

    def test_empty_args(self):
        result = extract_failed_handles(arguments={})
        assert result == {}

    def test_none_values_ignored(self):
        args = {"job_id": None, "run_id": ""}
        result = extract_failed_handles(arguments=args)
        assert result == {}


class TestFailedHandleTracker:
    def test_update_from_runtime_failure(self):
        tracker = FailedHandleTracker()
        payload = {"ok": True, "payload": {"error": {"code": "operation_failed"}}}
        tracker.update_from_result(
            result_payload=payload,
            arguments={"job_id": "j-failed"},
        )
        assert "j-failed" in tracker.failed_ids

    def test_no_update_on_success(self):
        tracker = FailedHandleTracker()
        payload = {"ok": True, "payload": {"status": "completed"}}
        tracker.update_from_result(
            result_payload=payload,
            arguments={"job_id": "j-ok"},
        )
        assert len(tracker.failed_ids) == 0

    def test_check_arguments_blocked(self):
        tracker = FailedHandleTracker()
        tracker.tracks["j-failed"] = "runtime_operation_failed"
        msg = tracker.check_arguments({"job_id": "j-failed", "action": "result"})
        assert msg is not None
        assert "j-failed" in msg
        assert "failed operation" in msg

    def test_check_arguments_passes(self):
        tracker = FailedHandleTracker()
        msg = tracker.check_arguments({"job_id": "j-ok", "action": "result"})
        assert msg is None

    def test_check_empty_arguments(self):
        tracker = FailedHandleTracker()
        tracker.tracks["j-failed"] = "runtime_operation_failed"
        msg = tracker.check_arguments({})
        assert msg is None

    def test_full_flow_prevents_cascade(self):
        tracker = FailedHandleTracker()
        # Walkforward job fails
        wf_payload = {
            "ok": True,
            "payload": {
                "content": [{"type": "text", "text": json.dumps({
                    "error": {"code": "operation_failed"},
                })}]
            },
        }
        tracker.update_from_result(
            result_payload=wf_payload,
            arguments={"job_id": "wf-001", "action": "result"},
        )
        assert "wf-001" in tracker.failed_ids

        # Model tries to analyse the failed job — should be blocked
        msg = tracker.check_arguments({"job_id": "wf-001", "action": "status"})
        assert msg is not None
        assert "failed operation" in msg

        # Model tries to use a different, non-failed job — should pass
        msg = tracker.check_arguments({"job_id": "wf-002", "action": "result"})
        assert msg is None

    def test_multiple_failed_handles(self):
        tracker = FailedHandleTracker()
        fail_payload = {"ok": True, "payload": {"error": {"code": "operation_failed"}}}
        tracker.update_from_result(result_payload=fail_payload, arguments={"job_id": "j1"})
        tracker.update_from_result(result_payload=fail_payload, arguments={"job_id": "j2"})
        assert tracker.failed_ids == {"j1", "j2"}
