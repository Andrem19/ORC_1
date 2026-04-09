"""
Logging setup for the orchestrator.

Configures three handlers:
- Console: Rich-styled colored output (or plain text fallback)
- File:    Plain text, always at DEBUG level
- JSONL:   Structured trace file, always at DEBUG level
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import uuid

from app.console_logging import ConsoleNoiseFilter
from app.logging_json import JsonLinesFormatter
from app.run_context import ensure_current_run
from app.runtime_incidents import LocalIncidentStore

_SESSION_ID = uuid.uuid4().hex[:8]


class _SessionContextFilter(logging.Filter):
    """Inject a stable session id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = _SESSION_ID
        return True


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_file: str = "orchestrator.log",
    rich_console: bool = True,
    console_log_level: str | None = None,
    truncate_length: int = 300,
    run_id: str = "",
) -> logging.Logger:
    """Configure and return the orchestrator logger."""
    logger = logging.getLogger("orchestrator")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)
    logger.propagate = False

    # --- Console handler ---
    ch = _build_console_handler(
        rich_console=rich_console,
        log_level=log_level,
        console_log_level=console_log_level,
        truncate_length=truncate_length,
    )
    ch.addFilter(_SessionContextFilter())
    ch.addFilter(ConsoleNoiseFilter())

    logger.addHandler(ch)

    # --- File handlers (plain text + structured JSONL, always DEBUG) ---
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    if run_id:
        log_path = ensure_current_run(log_path, run_id)

    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [session=%(session_id)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_path = (log_path / log_file).resolve()
    _quarantine_corrupt_log(file_path, run_id=run_id)
    fh = logging.FileHandler(file_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    fh.addFilter(_SessionContextFilter())
    logger.addHandler(fh)
    json_file_path = (log_path / "orchestrator.events.jsonl").resolve()
    jh = logging.FileHandler(json_file_path, mode="a", encoding="utf-8")
    jh.setLevel(logging.DEBUG)
    jh.setFormatter(JsonLinesFormatter())
    jh.addFilter(_SessionContextFilter())
    logger.addHandler(jh)
    logger.orchestrator_log_path = str(file_path)
    logger.orchestrator_json_log_path = str(json_file_path)
    fh.flush()
    _refresh_root_log_alias(root_dir=Path(log_dir), file_name=log_file, target=file_path)

    return logger


def _build_console_handler(
    *,
    rich_console: bool,
    log_level: str,
    console_log_level: str | None,
    truncate_length: int,
) -> logging.Handler:
    handler_level = getattr(logging, (console_log_level or log_level).upper(), logging.INFO)
    if rich_console:
        try:
            from app.rich_handler import RichConsoleHandler, get_console

            console = get_console()
            handler = RichConsoleHandler(console=console, truncate_length=truncate_length)
            handler.setLevel(handler_level)
            return handler
        except Exception as exc:
            fallback = _plain_console_handler(handler_level)
            root_logger = logging.getLogger("orchestrator")
            if root_logger.handlers:
                root_logger.warning("Rich console init failed; falling back to plain console: %s", exc)
            return fallback
    return _plain_console_handler(handler_level)


def _plain_console_handler(level: int) -> logging.Handler:
    plain_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [session=%(session_id)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(plain_fmt)
    return handler


def _quarantine_corrupt_log(file_path: Path, *, run_id: str) -> None:
    if not file_path.exists():
        return
    try:
        raw = file_path.read_bytes()
    except OSError:
        return
    if b"\x00" not in raw:
        try:
            raw.decode("utf-8")
            return
        except UnicodeDecodeError:
            pass
    corrupt_path = file_path.with_suffix(file_path.suffix + ".corrupt")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    corrupt_path = corrupt_path.with_name(f"{corrupt_path.name}.{timestamp}")
    file_path.rename(corrupt_path)
    incident_root = file_path.parent
    if "runs" in file_path.parts:
        runs_index = file_path.parts.index("runs")
        incident_root = Path(*file_path.parts[:runs_index])
    LocalIncidentStore(incident_root, run_id=run_id).record(
        summary="Corrupt orchestrator log quarantined",
        metadata={"log_path": str(file_path), "corrupt_path": str(corrupt_path)},
        source="logging_setup",
        severity="high",
    )


def _refresh_root_log_alias(*, root_dir: Path, file_name: str, target: Path) -> None:
    root_dir = root_dir.resolve()
    target = target.resolve()
    alias_path = root_dir / file_name
    if alias_path.resolve() == target if alias_path.exists() else False:
        return
    try:
        if alias_path.exists() or alias_path.is_symlink():
            alias_path.unlink(missing_ok=True)
        alias_path.symlink_to(target.relative_to(root_dir), target_is_directory=False)
    except OSError:
        return
    except ValueError:
        return
