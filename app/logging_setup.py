"""
Logging setup for the orchestrator.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "logs", log_file: str = "orchestrator.log") -> logging.Logger:
    """Configure and return the orchestrator logger."""
    logger = logging.getLogger("orchestrator")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path / log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
