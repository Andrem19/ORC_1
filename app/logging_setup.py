"""
Logging setup for the orchestrator.

Configures two handlers:
- Console: Rich-styled colored output (or plain text fallback)
- File:    Plain text, always at DEBUG level
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import uuid

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
    truncate_length: int = 300,
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
    use_rich = rich_console and sys.stdout.isatty()

    if use_rich:
        from app.rich_handler import RichConsoleHandler

        ch = RichConsoleHandler(truncate_length=truncate_length)
        ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    else:
        plain_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] [session=%(session_id)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        ch.setFormatter(plain_fmt)
    ch.addFilter(_SessionContextFilter())

    logger.addHandler(ch)

    # --- File handler (always plain text, always DEBUG) ---
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [session=%(session_id)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_path = (log_path / log_file).resolve()
    fh = logging.FileHandler(file_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    fh.addFilter(_SessionContextFilter())
    logger.addHandler(fh)
    logger.orchestrator_log_path = str(file_path)
    fh.flush()

    return logger
