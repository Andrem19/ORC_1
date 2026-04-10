"""Tests for config loading and validation."""

import json
import tempfile
from pathlib import Path

from app.config import OrchestratorConfig, load_config_from_dict


def test_default_config():
    cfg = OrchestratorConfig()
    assert cfg.poll_interval_seconds == 300
    assert cfg.max_empty_cycles == 12
    assert cfg.max_errors_total == 20
    assert cfg.max_plans_per_run == 1
    assert cfg.plan_source == "planner"
    assert cfg.compiled_plan_dir == "compiled_plans"
    assert len(cfg.workers) >= 1
    assert cfg.direct_execution.timeout_seconds == 1200
    assert cfg.direct_execution.max_attempts_per_slice == 2
    assert cfg.direct_execution.max_tool_calls_per_slice == 24
    assert cfg.direct_execution.max_expensive_tool_calls_per_slice == 6


def test_config_from_dict():
    data = {
        "goal": "Build a web app",
        "poll_interval_seconds": 60,
        "max_empty_cycles": 5,
        "planner_prompt_template": "Plan: $goal",
        "workers": [
            {"worker_id": "w1", "role": "executor", "system_prompt": "Do work"},
            {"worker_id": "w2", "role": "tester", "system_prompt": "Test things"},
        ],
    }
    cfg = load_config_from_dict(data)
    assert cfg.goal == "Build a web app"
    assert cfg.poll_interval_seconds == 60
    assert cfg.planner_prompt_template == "Plan: $goal"
    assert len(cfg.workers) == 2
    assert cfg.workers[0].worker_id == "w1"
def test_config_defaults_for_missing_keys():
    data = {"goal": "Minimal config"}
    cfg = load_config_from_dict(data)
    assert cfg.goal == "Minimal config"
    assert cfg.poll_interval_seconds == 300  # default


def test_config_adapter_settings():
    data = {
        "planner_adapter": {"name": "claude", "cli_path": "/usr/bin/claude", "model": "opus"},
        "worker_adapter": {"name": "qwen", "cli_path": "/usr/bin/qwen"},
    }
    cfg = load_config_from_dict(data)
    assert cfg.planner_adapter.cli_path == "/usr/bin/claude"
    assert cfg.worker_adapter.cli_path == "/usr/bin/qwen"


def test_config_state_path():
    cfg = OrchestratorConfig(state_dir="/tmp/orc_test", state_file="state.json")
    assert cfg.state_path == Path("/tmp/orc_test/state.json")


def test_config_to_dict():
    cfg = OrchestratorConfig(goal="test")
    d = cfg.to_dict()
    assert d["goal"] == "test"
    assert "workers" in d
    assert "poll_interval_seconds" in d


def test_load_config_with_lmstudio():
    data = {
        "lmstudio": {
            "base_url": "http://localhost:9999",
            "model": "qwen-test",
            "reasoning_effort": "low",
        },
    }
    cfg = load_config_from_dict(data)
    assert cfg.lmstudio.base_url == "http://localhost:9999"
    assert cfg.lmstudio.model == "qwen-test"
    assert cfg.lmstudio.reasoning_effort == "low"


def test_lmstudio_defaults_when_not_in_config():
    cfg = load_config_from_dict({"goal": "test"})
    assert cfg.lmstudio.base_url == "http://localhost:1234"
    assert cfg.lmstudio.model == ""
    assert cfg.lmstudio.reasoning_effort == "none"


def test_lmstudio_nested_translation_removed():
    """Translation settings are now in [notifications], not [lmstudio.translation]."""
    data = {
        "lmstudio": {
            "base_url": "http://localhost:8080",
            "model": "test-model",
        },
        "notifications": {
            "translation_provider": "lmstudio",
            "translation_fallback_1": "qwen_cli",
        },
    }
    cfg = load_config_from_dict(data)
    assert cfg.lmstudio.base_url == "http://localhost:8080"
    assert cfg.lmstudio.model == "test-model"
    assert cfg.notifications.translation_provider == "lmstudio"
    assert cfg.notifications.translation_fallback_1 == "qwen_cli"


def test_direct_execution_defaults_and_overrides():
    cfg = load_config_from_dict(
        {
            "worker_timeout_policy_by_tag": {"feature_contract": 180},
            "direct_execution": {
                "provider": "qwen_cli",
                "timeout_seconds": 900,
                "max_tool_calls_per_slice": 9,
                "mcp_endpoint_url": "http://127.0.0.1:8766/mcp",
            },
        }
    )

    assert cfg.worker_timeout_policy_by_tag["feature_contract"] == 180
    assert cfg.direct_execution.provider == "qwen_cli"
    assert cfg.direct_execution.timeout_seconds == 900
    assert cfg.direct_execution.max_tool_calls_per_slice == 9
    assert cfg.direct_execution.mcp_endpoint_url == "http://127.0.0.1:8766/mcp"


def test_load_config_ignores_removed_direct_legacy_settings():
    data = {
        "direct": {
            "endpoint_url": "http://127.0.0.1:8766/mcp",
        },
        "worker_adapter": {
            "name": "qwen_worker_cli",
            "cli_path": "/usr/bin/qwen",
            "allow_tool_use": False,
        },
    }

    cfg = load_config_from_dict(data)

    assert cfg.worker_adapter.allow_tool_use is False


def test_config_accepts_compiled_raw_plan_source():
    cfg = load_config_from_dict(
        {
            "plan_source": "compiled_raw",
            "raw_plan_dir": "raw_plans_custom",
            "compiled_plan_dir": "compiled_custom",
            "compiled_queue_skip_failures": False,
        }
    )

    assert cfg.plan_source == "compiled_raw"
    assert cfg.raw_plan_dir == "raw_plans_custom"
    assert cfg.compiled_plan_dir == "compiled_custom"
    assert cfg.compiled_queue_skip_failures is False


def test_config_accepts_direct_execution_section():
    cfg = load_config_from_dict(
        {
            "direct_execution": {
                "provider": "lmstudio",
                "max_attempts_per_slice": 2,
                "safe_exclude_tools": ["read_file"],
            }
        }
    )

    assert cfg.direct_execution.provider == "lmstudio"
    assert cfg.direct_execution.max_attempts_per_slice == 2
    assert cfg.direct_execution.safe_exclude_tools == ["read_file"]


def test_config_ignores_unknown_direct_execution_keys() -> None:
    cfg = load_config_from_dict(
        {
            "direct_execution": {
                "provider": "lmstudio",
                "fallback_1": "",
                "fallback_2": "",
                "max_tool_calls_per_slice": 12,
            }
        }
    )

    assert cfg.direct_execution.provider == "lmstudio"
    assert cfg.direct_execution.max_tool_calls_per_slice == 12
    assert cfg.direct_execution.fallback_1 == ""
    assert cfg.direct_execution.fallback_2 == ""


def test_config_parses_fallback_fields() -> None:
    cfg = load_config_from_dict(
        {
            "direct_execution": {
                "provider": "lmstudio",
                "fallback_1": "qwen_cli",
                "fallback_2": "claude_cli",
            }
        }
    )

    assert cfg.direct_execution.fallback_1 == "qwen_cli"
    assert cfg.direct_execution.fallback_2 == "claude_cli"


def test_config_rejects_invalid_plan_source():
    try:
        load_config_from_dict({"plan_source": "invalid"})
    except ValueError as exc:
        assert "Invalid plan_source" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid plan_source")
