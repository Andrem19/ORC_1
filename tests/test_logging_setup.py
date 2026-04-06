"""Tests for logging session isolation."""

from __future__ import annotations

import logging

from app.logging_setup import setup_logging


def test_setup_logging_restarts_file_and_adds_session_id(tmp_path) -> None:
    logger = setup_logging(log_level="DEBUG", log_dir=str(tmp_path), log_file="orchestrator.log", rich_console=False)
    logger.info("first run line")
    assert getattr(logger, "orchestrator_log_path", "").endswith("orchestrator.log")

    logger = setup_logging(log_level="DEBUG", log_dir=str(tmp_path), log_file="orchestrator.log", rich_console=False)
    logger.info("second run line")

    for handler in logger.handlers:
        handler.flush()

    content = (tmp_path / "orchestrator.log").read_text(encoding="utf-8")
    assert (tmp_path / "orchestrator.log").exists()
    assert "second run line" in content
    assert "first run line" not in content
    assert "[session=" in content

    logger.handlers.clear()
    logging.getLogger("orchestrator").handlers.clear()
