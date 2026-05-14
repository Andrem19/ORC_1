"""
Runtime helpers for startup log-directory lifecycle.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def clear_log_root(log_dir: str) -> None:
    """Remove all prior logs before a fresh startup, including resume mode."""
    root = Path(log_dir)
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink(missing_ok=True)
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)

