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
    assert len(cfg.workers) >= 1


def test_config_from_dict():
    data = {
        "goal": "Build a web app",
        "poll_interval_seconds": 60,
        "max_empty_cycles": 5,
        "workers": [
            {"worker_id": "w1", "role": "executor", "system_prompt": "Do work"},
            {"worker_id": "w2", "role": "tester", "system_prompt": "Test things"},
        ],
    }
    cfg = load_config_from_dict(data)
    assert cfg.goal == "Build a web app"
    assert cfg.poll_interval_seconds == 60
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
            "enabled": True,
            "base_url": "http://localhost:9999",
            "model": "qwen-test",
            "assistant": {
                "analysis_interval_cycles": 25,
                "max_log_lines": 100,
            },
        },
    }
    cfg = load_config_from_dict(data)
    assert cfg.lmstudio.enabled is True
    assert cfg.lmstudio.base_url == "http://localhost:9999"
    assert cfg.lmstudio.model == "qwen-test"
    assert cfg.lmstudio.analysis_interval_cycles == 25
    assert cfg.lmstudio.max_log_lines == 100


def test_lmstudio_defaults_when_not_in_config():
    cfg = load_config_from_dict({"goal": "test"})
    assert cfg.lmstudio.enabled is False
    assert cfg.lmstudio.base_url == "http://localhost:1234"
    assert cfg.lmstudio.analysis_interval_cycles == 50


def test_lmstudio_nested_report_compressor():
    data = {
        "lmstudio": {
            "enabled": True,
            "base_url": "http://localhost:8080",
            "model": "test-model",
            "report_compressor": {"enabled": True, "max_tokens": 300},
            "translation": {"max_tokens": 2048},
        },
    }
    cfg = load_config_from_dict(data)
    assert cfg.lmstudio.report_compressor.enabled is True
    assert cfg.lmstudio.report_compressor.base_url == "http://localhost:8080"
    assert cfg.lmstudio.report_compressor.model == "test-model"
    assert cfg.lmstudio.report_compressor.max_tokens == 300
    assert cfg.lmstudio.translation.max_tokens == 2048
    # report_compressor alias on OrchestratorConfig
    assert cfg.report_compressor.enabled is True
    assert cfg.report_compressor.base_url == "http://localhost:8080"


def test_lmstudio_legacy_flat_fields():
    """Legacy: flat analysis_interval_cycles/max_log_lines route to assistant."""
    data = {
        "lmstudio": {
            "enabled": True,
            "base_url": "http://localhost:9999",
            "model": "qwen-test",
            "analysis_interval_cycles": 25,
            "max_log_lines": 100,
        },
    }
    cfg = load_config_from_dict(data)
    assert cfg.lmstudio.analysis_interval_cycles == 25
    assert cfg.lmstudio.max_log_lines == 100
