"""Tests for tool_call_signature and semantic loop detection."""

from __future__ import annotations

from app.services.direct_execution.semantic_progress import tool_call_signature


# ---------- read-only actions now produce signatures ----------


def test_signature_returns_nonempty_for_readonly_list() -> None:
    sig = tool_call_signature("research_project", {"action": "list"})
    assert sig.startswith("ro:")


def test_signature_returns_nonempty_for_readonly_inspect() -> None:
    sig = tool_call_signature("features_catalog", {"action": "inspect", "scope": "available"})
    assert sig.startswith("ro:")


def test_signature_returns_nonempty_for_readonly_search() -> None:
    sig = tool_call_signature("research_memory", {"action": "search", "query": "invariants"})
    assert sig.startswith("ro:")


def test_signature_returns_nonempty_for_readonly_status() -> None:
    sig = tool_call_signature("backtests_runs", {"action": "status", "run_id": "abc"})
    assert sig.startswith("ro:")


# ---------- identical calls produce same signature ----------


def test_signature_identical_calls_give_same_signature() -> None:
    args = {"action": "list"}
    assert tool_call_signature("research_project", args) == tool_call_signature("research_project", args)


# ---------- different args produce different signatures ----------


def test_signature_different_args_give_different_signatures() -> None:
    sig1 = tool_call_signature("research_project", {"action": "list"})
    sig2 = tool_call_signature("research_project", {"action": "open", "project_id": "proj_1"})
    assert sig1 != sig2


def test_signature_different_views_give_different_signatures() -> None:
    sig1 = tool_call_signature("features_catalog", {"action": "inspect", "view": "summary"})
    sig2 = tool_call_signature("features_catalog", {"action": "inspect", "view": "detail"})
    assert sig1 != sig2


# ---------- edge cases ----------


def test_signature_empty_for_no_action_no_mutating_keys() -> None:
    """No action and no record/payload/project/coordinates keys -> empty signature."""
    assert tool_call_signature("some_tool", {}) == ""
    assert tool_call_signature("some_tool", {"unrelated_key": "value"}) == ""


def test_mutating_action_signature_unchanged() -> None:
    """Mutating actions still produce the same signature as before."""
    sig = tool_call_signature("research_memory", {"action": "create", "kind": "note", "project_id": "proj_1"})
    assert sig != ""
    assert not sig.startswith("ro:")
    assert "create" in sig
    assert "proj_1" in sig


def test_signature_includes_tool_name() -> None:
    sig = tool_call_signature("research_project", {"action": "list"})
    assert "research_project" in sig


def test_signature_includes_action() -> None:
    sig = tool_call_signature("research_project", {"action": "list"})
    assert "list" in sig
