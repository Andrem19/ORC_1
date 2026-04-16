"""Tests for is_expensive_tool_call action-aware expense classification."""

from __future__ import annotations

from app.services.mcp_catalog.classifier import is_expensive_tool, is_expensive_tool_call
from app.services.mcp_catalog.models import McpCatalogSnapshot, McpToolSpec


def _make_snapshot(*tool_defs: tuple[str, str]) -> McpCatalogSnapshot:
    tools = [
        McpToolSpec(name=name, cost_class=cost_class)
        for name, cost_class in tool_defs
    ]
    return McpCatalogSnapshot(
        server_name="test",
        endpoint_url="https://test.local",
        schema_hash="test",
        fetched_at="2026-01-01T00:00:00Z",
        tools=tools,
    )


# --- is_expensive_tool_call tests ---


def test_expensive_tool_with_read_action_is_not_expensive():
    snap = _make_snapshot(("features_custom", "expensive"))
    assert is_expensive_tool(snap, "features_custom") is True
    assert is_expensive_tool_call(snap, "features_custom", {"action": "inspect"}) is False


def test_expensive_tool_with_publish_action_is_expensive():
    snap = _make_snapshot(("features_custom", "expensive"))
    assert is_expensive_tool_call(snap, "features_custom", {"action": "publish"}) is True


def test_expensive_tool_with_validate_action_is_expensive():
    snap = _make_snapshot(("features_custom", "expensive"))
    assert is_expensive_tool_call(snap, "features_custom", {"action": "validate"}) is True


def test_expensive_tool_with_build_action_is_expensive():
    snap = _make_snapshot(("features_dataset", "expensive"))
    assert is_expensive_tool_call(snap, "features_dataset", {"action": "build"}) is True


def test_expensive_tool_with_refresh_action_is_expensive():
    snap = _make_snapshot(("features_dataset", "expensive"))
    assert is_expensive_tool_call(snap, "features_dataset", {"action": "refresh"}) is True


def test_expensive_tool_with_list_action_is_not_expensive():
    snap = _make_snapshot(("features_dataset", "expensive"))
    assert is_expensive_tool_call(snap, "features_dataset", {"action": "list"}) is False


def test_expensive_tool_with_search_action_is_not_expensive():
    snap = _make_snapshot(("research_memory", "cheap"))
    # research_memory is cheap at tool level, so action doesn't matter
    assert is_expensive_tool_call(snap, "research_memory", {"action": "search"}) is False


def test_expensive_tool_with_status_action_is_not_expensive():
    snap = _make_snapshot(("features_custom", "expensive"))
    assert is_expensive_tool_call(snap, "features_custom", {"action": "status"}) is False


def test_expensive_tool_with_compare_action_is_not_expensive():
    snap = _make_snapshot(("models_registry", "expensive"))
    assert is_expensive_tool_call(snap, "models_registry", {"action": "compare"}) is False


def test_cheap_tool_always_not_expensive():
    snap = _make_snapshot(("research_memory", "cheap"))
    assert is_expensive_tool_call(snap, "research_memory", {"action": "create"}) is False


def test_no_arguments_defaults_to_expensive():
    snap = _make_snapshot(("features_custom", "expensive"))
    assert is_expensive_tool_call(snap, "features_custom", None) is True


def test_empty_arguments_defaults_to_expensive():
    snap = _make_snapshot(("features_custom", "expensive"))
    assert is_expensive_tool_call(snap, "features_custom", {}) is True


def test_empty_action_defaults_to_expensive():
    snap = _make_snapshot(("features_custom", "expensive"))
    assert is_expensive_tool_call(snap, "features_custom", {"action": ""}) is True


def test_unknown_tool_not_expensive():
    snap = _make_snapshot(("features_custom", "expensive"))
    assert is_expensive_tool_call(snap, "nonexistent_tool", {"action": "inspect"}) is False


def test_expensive_tool_with_start_action_is_expensive():
    snap = _make_snapshot(("backtests_runs", "expensive"))
    assert is_expensive_tool_call(snap, "backtests_runs", {"action": "start"}) is True


def test_expensive_tool_with_prove_action_is_not_expensive():
    snap = _make_snapshot(("research_memory", "expensive"))
    assert is_expensive_tool_call(snap, "research_memory", {"action": "prove"}) is False


def test_action_case_insensitive():
    snap = _make_snapshot(("features_custom", "expensive"))
    assert is_expensive_tool_call(snap, "features_custom", {"action": "Inspect"}) is False
    assert is_expensive_tool_call(snap, "features_custom", {"action": "PUBLISH"}) is True
