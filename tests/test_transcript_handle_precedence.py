"""Tests for _prefer_transcript_handle_facts in executor."""

from __future__ import annotations

from app.services.direct_execution.executor import _prefer_transcript_handle_facts


def test_overrides_model_transcript_ref_with_real_project_id() -> None:
    action_facts = {
        "project_id": "transcript:2:research_project",
        "research.project_id": "transcript:2:research_project",
    }
    transcript_facts = {
        "project_id": "active-signal-cycle-78c740d5",
        "research.project_id": "active-signal-cycle-78c740d5",
    }
    _prefer_transcript_handle_facts(action_facts, transcript_facts)
    assert action_facts["project_id"] == "active-signal-cycle-78c740d5"
    assert action_facts["research.project_id"] == "active-signal-cycle-78c740d5"


def test_does_not_override_when_model_has_real_value() -> None:
    action_facts = {
        "project_id": "real-project-123",
        "research.project_id": "real-project-123",
    }
    transcript_facts = {
        "project_id": "other-project-456",
    }
    _prefer_transcript_handle_facts(action_facts, transcript_facts)
    assert action_facts["project_id"] == "real-project-123"


def test_does_not_override_when_transcript_value_is_empty() -> None:
    action_facts = {
        "project_id": "transcript:2:research_project",
    }
    transcript_facts = {}
    _prefer_transcript_handle_facts(action_facts, transcript_facts)
    assert action_facts["project_id"] == "transcript:2:research_project"


def test_does_not_override_when_transcript_value_is_also_transcript_ref() -> None:
    action_facts = {
        "project_id": "transcript:2:research_project",
    }
    transcript_facts = {
        "project_id": "transcript:4:research_project",
    }
    _prefer_transcript_handle_facts(action_facts, transcript_facts)
    assert action_facts["project_id"] == "transcript:2:research_project"


def test_handles_multiple_handle_fields() -> None:
    action_facts = {
        "project_id": "transcript:2:research_project",
        "job_id": "transcript:4:experiments_run",
        "run_id": "real-run-id",
        "snapshot_id": "transcript:6:backtests_strategy",
    }
    transcript_facts = {
        "project_id": "real-project",
        "job_id": "real-job-123",
        "run_id": "other-run-id",
        "snapshot_id": "real-snapshot-abc",
    }
    _prefer_transcript_handle_facts(action_facts, transcript_facts)
    assert action_facts["project_id"] == "real-project"
    assert action_facts["research.project_id"] == "real-project"
    assert action_facts["job_id"] == "real-job-123"
    # run_id was real already — no override
    assert action_facts["run_id"] == "real-run-id"
    assert action_facts["snapshot_id"] == "real-snapshot-abc"
