"""Tests for runtime_factory sequence report adapter creation."""

import pytest

from app.config import OrchestratorConfig, SequenceReportConfig
from app.runtime_factory import create_sequence_report_adapter


def test_create_sequence_report_adapter_claude_cli():
    cfg = OrchestratorConfig(
        sequence_report=SequenceReportConfig(provider="claude_cli", model="opus"),
    )
    adapter = create_sequence_report_adapter(cfg)
    assert adapter.name() == "claude_planner_cli"


def test_create_sequence_report_adapter_lmstudio():
    from app.config import LMStudioConfig
    cfg = OrchestratorConfig(
        sequence_report=SequenceReportConfig(provider="lmstudio"),
    )
    cfg.lmstudio = LMStudioConfig(base_url="http://localhost:1234", model="test-model")
    adapter = create_sequence_report_adapter(cfg)
    assert adapter.name() == "lmstudio_worker_api"


def test_create_sequence_report_adapter_unknown_provider():
    cfg = OrchestratorConfig(
        sequence_report=SequenceReportConfig(provider="unknown_provider"),
    )
    with pytest.raises(ValueError, match="Unknown sequence_report provider"):
        create_sequence_report_adapter(cfg)
