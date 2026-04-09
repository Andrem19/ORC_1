"""Tests for logging session isolation."""

from __future__ import annotations

import logging
import json

from app.logging_setup import _build_console_handler, setup_logging


def test_setup_logging_appends_file_and_adds_session_id(tmp_path) -> None:
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
    assert "first run line" in content
    assert "[session=" in content

    logger.handlers.clear()
    logging.getLogger("orchestrator").handlers.clear()


def test_setup_logging_uses_run_scoped_path_and_root_alias(tmp_path) -> None:
    logger = setup_logging(
        log_level="DEBUG",
        log_dir=str(tmp_path),
        log_file="orchestrator.log",
        rich_console=False,
        run_id="run-123",
    )
    logger.info("run scoped line")

    for handler in logger.handlers:
        handler.flush()

    expected = tmp_path / "runs" / "run-123" / "orchestrator.log"
    structured = tmp_path / "runs" / "run-123" / "orchestrator.events.jsonl"
    assert expected.exists()
    assert structured.exists()
    assert "run scoped line" in expected.read_text(encoding="utf-8")
    assert (tmp_path / "orchestrator.log").is_symlink()
    payload = json.loads(structured.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["message"] == "run scoped line"
    assert payload["logger"] == "orchestrator"

    logger.handlers.clear()
    logging.getLogger("orchestrator").handlers.clear()


def test_setup_logging_quarantines_corrupt_log_and_records_incident(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-123"
    run_dir.mkdir(parents=True)
    corrupt_log = run_dir / "orchestrator.log"
    corrupt_log.write_bytes(b"ok\x00bad")

    logger = setup_logging(
        log_level="DEBUG",
        log_dir=str(tmp_path),
        log_file="orchestrator.log",
        rich_console=False,
        run_id="run-123",
    )
    logger.info("healthy replacement log")
    for handler in logger.handlers:
        handler.flush()

    assert "healthy replacement log" in corrupt_log.read_text(encoding="utf-8")
    quarantined = list(run_dir.glob("orchestrator.log.corrupt.*"))
    assert quarantined
    incidents = list((run_dir / "incidents").glob("*.json"))
    assert incidents

    logger.handlers.clear()
    logging.getLogger("orchestrator").handlers.clear()


def test_setup_logging_accepts_console_log_level(tmp_path) -> None:
    logger = setup_logging(
        log_level="DEBUG",
        log_dir=str(tmp_path),
        log_file="orchestrator.log",
        rich_console=False,
        console_log_level="WARNING",
    )

    assert logger.handlers[0].level == logging.WARNING

    logger.handlers.clear()
    logging.getLogger("orchestrator").handlers.clear()


def test_build_console_handler_falls_back_to_plain(monkeypatch) -> None:
    class _BrokenRichHandler(logging.Handler):
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError("rich failed")

    monkeypatch.setattr("app.rich_handler.RichConsoleHandler", _BrokenRichHandler)

    handler = _build_console_handler(
        rich_console=True,
        log_level="INFO",
        console_log_level=None,
        truncate_length=120,
    )

    assert isinstance(handler, logging.StreamHandler)
