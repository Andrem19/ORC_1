"""
Qwen Code CLI adapter for the worker role.

Invokes `qwen-code` CLI via subprocess and returns normalized output.
Supports both synchronous invoke() and async start()/check() modes.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import subprocess
import time
from typing import Any

from app.adapters.base import AdapterResponse, BaseAdapter, ProcessHandle

logger = logging.getLogger("orchestrator.adapter.qwen")


class QwenWorkerCli(BaseAdapter):
    """Adapter that calls Qwen Code CLI as a worker."""

    def __init__(
        self,
        cli_path: str = "qwen-code",
        extra_flags: list[str] | None = None,
    ) -> None:
        self.cli_path = cli_path
        self.extra_flags = extra_flags or []

    def name(self) -> str:
        return "qwen_worker_cli"

    def is_available(self) -> bool:
        return shutil.which(self.cli_path) is not None

    def invoke(self, prompt: str, timeout: int = 300, **kwargs: Any) -> AdapterResponse:
        """Call Qwen Code CLI with a prompt and return the output (blocking)."""
        cmd = [self.cli_path, "-p", prompt]
        cmd.extend(self.extra_flags)

        logger.info("Calling Qwen CLI (timeout=%ds)", timeout)
        logger.debug("Qwen CLI command: %s (prompt: %d chars)", cmd[0], len(prompt))
        start = time.monotonic()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.monotonic() - start
            output = result.stdout.strip()

            if result.returncode != 0:
                stderr = result.stderr.strip()
                logger.warning("Qwen CLI returned exit code %d: %s", result.returncode, stderr[:200])
                return AdapterResponse(
                    success=False,
                    raw_output=output,
                    exit_code=result.returncode,
                    error=stderr[:500],
                    duration_seconds=duration,
                )

            logger.info("Qwen CLI completed in %.1fs, output length: %d", duration, len(output))
            return AdapterResponse(
                success=True,
                raw_output=output,
                exit_code=0,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            logger.warning("Qwen CLI timed out after %.1fs", duration)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=f"Timed out after {timeout}s",
                timed_out=True,
                duration_seconds=duration,
            )

        except FileNotFoundError:
            logger.error("Qwen CLI not found at: %s", self.cli_path)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=f"CLI not found: {self.cli_path}",
            )

        except Exception as e:
            duration = time.monotonic() - start
            logger.error("Qwen CLI unexpected error: %s", e)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=str(e),
                duration_seconds=duration,
            )

    # ---------------------------------------------------------------
    # Async background execution (start / check)
    # ---------------------------------------------------------------

    def start(self, prompt: str, **kwargs: Any) -> ProcessHandle:
        """Launch Qwen CLI as a background process. Returns immediately."""
        cmd = [self.cli_path, "-p", prompt]
        cmd.extend(self.extra_flags)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Set stdout to non-blocking so check() can read without blocking
        fd = process.stdout.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        logger.info("Started Qwen CLI pid=%d", process.pid)
        return ProcessHandle(
            process=process,
            task_id=kwargs.get("task_id", ""),
            worker_id=kwargs.get("worker_id", ""),
            started_at=time.monotonic(),
        )

    def check(self, handle: ProcessHandle) -> tuple[str, bool]:
        """Non-blocking check on a running worker.

        Returns (new_output_since_last_check, is_finished).
        """
        new_output = ""
        proc = handle.process
        if proc is None:
            return "", True

        # Non-blocking read from stdout
        try:
            fd = proc.stdout.fileno()
            chunk = os.read(fd, 65536)
            if chunk:
                new_output = chunk.decode("utf-8", errors="replace")
        except (BlockingIOError, OSError):
            pass  # no data available yet

        is_finished = proc.poll() is not None

        if is_finished:
            # Drain remaining stdout
            try:
                remaining = proc.stdout.read()
                if remaining:
                    new_output += remaining
            except Exception:
                pass

            # Read stderr for error reporting
            try:
                stderr_output = proc.stderr.read() or ""
            except Exception:
                stderr_output = ""

            proc.wait()

            if proc.returncode and proc.returncode != 0 and stderr_output:
                logger.warning(
                    "Worker pid=%d exited %d: %s",
                    proc.pid, proc.returncode, stderr_output[:200],
                )

        handle.partial_output += new_output
        return new_output, is_finished
