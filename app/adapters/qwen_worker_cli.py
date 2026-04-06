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
            output = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

            if result.returncode != 0:
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

        except Exception as exc:
            duration = time.monotonic() - start
            logger.error("Qwen CLI unexpected error: %s", exc)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=str(exc),
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
            text=False,
            bufsize=0,
        )

        self._set_nonblocking(process.stdout)
        self._set_nonblocking(process.stderr)

        logger.info("Started Qwen CLI pid=%d", process.pid)
        return ProcessHandle(
            process=process,
            task_id=kwargs.get("task_id", ""),
            worker_id=kwargs.get("worker_id", ""),
            started_at=time.monotonic(),
            metadata={"raw_stdout": "", "raw_stderr": "", "output_mode": "text"},
        )

    def check(self, handle: ProcessHandle) -> tuple[str, bool]:
        """Non-blocking check on a running worker."""
        proc = handle.process
        if proc is None:
            return "", True

        stdout_fragment = self._read_available(proc.stdout)
        stderr_fragment = self._read_available(proc.stderr)

        if stdout_fragment:
            handle.partial_output += stdout_fragment
            handle.metadata["raw_stdout"] = handle.metadata.get("raw_stdout", "") + stdout_fragment
        if stderr_fragment:
            handle.partial_error_output += stderr_fragment
            handle.metadata["raw_stderr"] = handle.metadata.get("raw_stderr", "") + stderr_fragment

        is_finished = proc.poll() is not None
        if is_finished:
            final_stdout = self._read_remaining(proc.stdout)
            final_stderr = self._read_remaining(proc.stderr)
            if final_stdout:
                handle.partial_output += final_stdout
                handle.metadata["raw_stdout"] = handle.metadata.get("raw_stdout", "") + final_stdout
                stdout_fragment += final_stdout
            if final_stderr:
                handle.partial_error_output += final_stderr
                handle.metadata["raw_stderr"] = handle.metadata.get("raw_stderr", "") + final_stderr
            proc.wait()

            if proc.returncode and proc.returncode != 0:
                logger.warning(
                    "Worker pid=%d exited %d: %s",
                    proc.pid, proc.returncode, handle.partial_error_output[:200],
                )

        return stdout_fragment, is_finished

    @staticmethod
    def _set_nonblocking(pipe: Any) -> None:
        if pipe is None:
            return
        fd = pipe.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    @staticmethod
    def _read_available(pipe: Any) -> str:
        if pipe is None:
            return ""
        try:
            chunk = os.read(pipe.fileno(), 65536)
        except (BlockingIOError, OSError):
            return ""
        if not chunk:
            return ""
        return chunk.decode("utf-8", errors="replace")

    @staticmethod
    def _read_remaining(pipe: Any) -> str:
        if pipe is None:
            return ""
        fragments: list[str] = []
        while True:
            try:
                chunk = os.read(pipe.fileno(), 65536)
            except OSError:
                break
            if not chunk:
                break
            fragments.append(chunk.decode("utf-8", errors="replace"))
        return "".join(fragments)
