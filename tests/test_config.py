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
