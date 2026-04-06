"""
Claude CLI adapter for the planner role.

Invokes `claude` CLI via subprocess and returns normalized output.
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

logger = logging.getLogger("orchestrator.adapter.claude")


class ClaudePlannerCli(BaseAdapter):
    """Adapter that calls Claude Code CLI as the planner."""

    def __init__(
        self,
        cli_path: str = "claude",
        model: str = "opus",
        extra_flags: list[str] | None = None,
    ) -> None:
        self.cli_path = cli_path
        self.model = model
        # Mandatory flags: disable all tools so Claude CLI acts as a pure
        # text generator instead of a full agent with tool use / thinking.
        self._mandatory_flags = ["--tools", ""]
        self.extra_flags = extra_flags or []

    def name(self) -> str:
        return "claude_planner_cli"

    def is_available(self) -> bool:
        return shutil.which(self.cli_path) is not None

    # ---------------------------------------------------------------
    # Synchronous mode (blocking, kept for backward compat)
    # ---------------------------------------------------------------

    def invoke(self, prompt: str, timeout: int = 180, **kwargs: Any) -> AdapterResponse:
        """Call Claude CLI with a prompt and return the output (blocking)."""
        cmd = [self.cli_path, "--model", self.model, "-p", prompt]
        cmd.extend(self._mandatory_flags)
        cmd.extend(self.extra_flags)

        logger.info("Calling Claude CLI (model=%s, timeout=%ds)", self.model, timeout)
        logger.debug("Claude CLI command: %s (prompt: %d chars)", cmd[:2], len(prompt))
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
                logger.warning("Claude CLI returned exit code %d: %s", result.returncode, stderr[:200])
                return AdapterResponse(
                    success=False,
                    raw_output=output,
                    exit_code=result.returncode,
                    error=stderr[:500],
                    duration_seconds=duration,
                )

            logger.info("Claude CLI completed in %.1fs, output length: %d", duration, len(output))
            return AdapterResponse(
                success=True,
                raw_output=output,
                exit_code=0,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            logger.warning("Claude CLI timed out after %.1fs", duration)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=f"Timed out after {timeout}s",
                timed_out=True,
                duration_seconds=duration,
            )

        except FileNotFoundError:
            logger.error("Claude CLI not found at: %s", self.cli_path)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=f"CLI not found: {self.cli_path}",
            )

        except Exception as e:
            duration = time.monotonic() - start
            logger.error("Claude CLI unexpected error: %s", e)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=str(e),
                duration_seconds=duration,
            )

    # ---------------------------------------------------------------
    # Async mode (non-blocking background execution)
    # ---------------------------------------------------------------

    def start(self, prompt: str, **kwargs: Any) -> ProcessHandle:
        """Launch Claude CLI as a background process. Returns immediately."""
        cmd = [self.cli_path, "--model", self.model, "-p", prompt]
        cmd.extend(self._mandatory_flags)
        cmd.extend(self.extra_flags)

        logger.info("Starting Claude CLI (model=%s) as background process", self.model)
        logger.debug("Claude CLI command: %s (prompt: %d chars)", cmd[:3], len(prompt))

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Set stdout to non-blocking
        fd = process.stdout.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        logger.info("Started Claude CLI pid=%d", process.pid)
        return ProcessHandle(
            process=process,
            task_id=kwargs.get("task_id", "planner"),
            worker_id="planner",
            started_at=time.monotonic(),
        )

    def check(self, handle: ProcessHandle) -> tuple[str, bool]:
        """Non-blocking check on a running planner process.

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
                    "Planner pid=%d exited %d: %s",
                    proc.pid, proc.returncode, stderr_output[:200],
                )

        handle.partial_output += new_output
        return new_output, is_finished
