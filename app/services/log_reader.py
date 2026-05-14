"""Log tail reader — efficiently reads the last N lines from a log file."""

from __future__ import annotations

from collections import deque
from pathlib import Path


def read_last_n_lines(log_path: str | Path, n: int = 200) -> list[str]:
    """Read the last N lines from a log file.

    Uses a deque for memory-efficient tail reading.
    Returns an empty list if the file is missing or empty.
    """
    path = Path(log_path)
    if not path.is_file():
        return []

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            tail: deque[str] = deque(f, maxlen=n)
        return [line.rstrip("\n") for line in tail]
    except OSError:
        return []
