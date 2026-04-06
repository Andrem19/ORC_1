"""
PID file locking to prevent concurrent orchestrator instances.

Uses fcntl.flock(LOCK_EX | LOCK_NB) on a PID file.
If the lock is held by another process, the current process exits.
The lock is automatically released when the process terminates.
"""
from __future__ import annotations

import fcntl
import logging
import os
from pathlib import Path

logger = logging.getLogger("orchestrator.pid_lock")


class PidLock:
    """Exclusive lock via PID file."""

    def __init__(self, pid_path: str | Path) -> None:
        self.pid_path = Path(pid_path)
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Try to acquire an exclusive lock. Returns True on success."""
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self.pid_path), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as e:
            logger.error("Cannot open PID file %s: %s", self.pid_path, e)
            return False

        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            try:
                existing = os.read(self._fd, 32).decode().strip()
            except Exception:
                existing = "unknown"
            os.close(self._fd)
            self._fd = None
            logger.error(
                "Another orchestrator instance is running (PID: %s, lock: %s)",
                existing, self.pid_path,
            )
            return False

        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, str(os.getpid()).encode())
        logger.info("PID lock acquired: %s (pid=%d)", self.pid_path, os.getpid())
        return True

    def release(self) -> None:
        """Release the lock and remove the PID file."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
            try:
                self.pid_path.unlink(missing_ok=True)
            except OSError:
                pass
            logger.info("PID lock released: %s", self.pid_path)
