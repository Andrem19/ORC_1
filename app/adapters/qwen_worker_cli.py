"""
Qwen Code CLI adapter for the worker role.

Invokes `qwen-code` CLI via subprocess and returns normalized output.
Supports both synchronous invoke() and async start()/check() modes.
"""

from __future__ import annotations

import fcntl
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from typing import Any

from app.adapters.base import AdapterResponse, BaseAdapter, ProcessHandle
from app.planner_stream import consume_stream_fragment
from app.qwen_stream import render_qwen_stream_output
from app.subprocess_io import drain_pipe_text, read_available_text

logger = logging.getLogger("orchestrator.adapter.qwen")

_RAW_STDOUT_LIMIT = 600_000
_RAW_STDERR_LIMIT = 200_000
_PARTIAL_OUTPUT_LIMIT = 300_000
_LOG_PROMPT_EXCERPT = 240


class QwenWorkerCli(BaseAdapter):
    """Adapter that calls Qwen Code CLI as a worker."""

    def __init__(
        self,
        cli_path: str = "qwen-code",
        extra_flags: list[str] | None = None,
        *,
        use_stream_json: bool = True,
        allow_tool_use: bool = False,
    ) -> None:
        self.cli_path = cli_path
        self.extra_flags = extra_flags or []
        self.use_stream_json = use_stream_json
        self.allow_tool_use = allow_tool_use
        self._resolved_cli_path: str | None = None
        self._resolved_env_path_prefix = ""

    def name(self) -> str:
        return "qwen_worker_cli"

    def is_available(self) -> bool:
        return bool(self._resolve_cli_path())

    def _build_command(self, prompt: str) -> list[str]:
        cmd = [self._resolve_cli_path() or self.cli_path]
        if self.allow_tool_use:
            cmd.append("--yolo")
        else:
            cmd.extend(["--approval-mode", "plan"])
        if self.use_stream_json:
            cmd.extend(["-o", "stream-json"])
        cmd.extend(["-p", prompt])
        cmd.extend(self.extra_flags)
        return cmd

    def invoke(self, prompt: str, timeout: int = 300, **kwargs: Any) -> AdapterResponse:
        """Call Qwen Code CLI with a prompt and return the output (blocking)."""
        cmd = self._build_command(prompt)

        logger.info("Calling Qwen CLI (timeout=%ds)", timeout)
        logger.debug(
            "Qwen CLI command: %s (prompt_chars=%d excerpt=%r)",
            cmd[0],
            len(prompt),
            prompt[:_LOG_PROMPT_EXCERPT],
        )
        start = time.monotonic()

        try:
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._build_env(),
            )
            duration = time.monotonic() - start
            output = self._render_output(result.stdout or "")
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
        cmd = self._build_command(prompt)
        output_mode = "stream-json" if self.use_stream_json else "text"

        logger.info("Starting Qwen CLI (mode=%s) as background process", output_mode)
        logger.debug(
            "Qwen CLI command: %s (prompt_chars=%d excerpt=%r)",
            cmd[0],
            len(prompt),
            prompt[:_LOG_PROMPT_EXCERPT],
        )

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            env=self._build_env(),
        )

        self._set_nonblocking(process.stdout)
        self._set_nonblocking(process.stderr)

        logger.info("Started Qwen CLI pid=%d", process.pid)
        return ProcessHandle(
            process=process,
            task_id=kwargs.get("task_id", ""),
            worker_id=kwargs.get("worker_id", ""),
            started_at=time.monotonic(),
            metadata={
                "raw_stdout": "",
                "raw_stderr": "",
                "output_mode": output_mode,
                "stream_buffer": "",
                "stream_event_count": 0,
            },
        )

    def check(self, handle: ProcessHandle) -> tuple[str, bool]:
        """Non-blocking check on a running worker."""
        proc = handle.process
        if proc is None:
            return "", True

        stdout_fragment = self._read_available(proc.stdout)
        stderr_fragment = self._read_available(proc.stderr)

        rendered_fragment = self._consume_fragment(handle, stdout_fragment)
        if stderr_fragment:
            handle.partial_error_output = _append_limited(handle.partial_error_output, stderr_fragment, _RAW_STDERR_LIMIT)
            handle.metadata["raw_stderr"] = _append_limited(
                handle.metadata.get("raw_stderr", ""),
                stderr_fragment,
                _RAW_STDERR_LIMIT,
            )

        is_finished = proc.poll() is not None
        if is_finished:
            proc.wait()
            final_stdout = drain_pipe_text(proc.stdout)
            final_stderr = drain_pipe_text(proc.stderr)
            if final_stdout:
                rendered_fragment += self._consume_fragment(handle, final_stdout)
            if final_stderr:
                handle.partial_error_output = _append_limited(handle.partial_error_output, final_stderr, _RAW_STDERR_LIMIT)
                handle.metadata["raw_stderr"] = _append_limited(
                    handle.metadata.get("raw_stderr", ""),
                    final_stderr,
                    _RAW_STDERR_LIMIT,
                )

            if proc.returncode and proc.returncode != 0:
                logger.warning(
                    "Worker pid=%d exited %d: %s",
                    proc.pid, proc.returncode, handle.partial_error_output[:200],
                )

        return rendered_fragment, is_finished

    def _consume_fragment(self, handle: ProcessHandle, fragment: str) -> str:
        """Process raw stdout fragment — parse stream-json or pass through text."""
        if not fragment:
            return ""
        handle.metadata["raw_stdout"] = _append_limited(
            handle.metadata.get("raw_stdout", ""),
            fragment,
            _RAW_STDOUT_LIMIT,
        )
        output_mode = str(handle.metadata.get("output_mode", "text"))
        if output_mode == "stream-json":
            rendered, buffer, event_count = consume_stream_fragment(
                fragment,
                str(handle.metadata.get("stream_buffer", "")),
            )
            handle.metadata["stream_buffer"] = buffer
            handle.metadata["stream_event_count"] = int(handle.metadata.get("stream_event_count", 0)) + event_count
            handle.partial_output = _append_limited(handle.partial_output, rendered, _PARTIAL_OUTPUT_LIMIT)
            return rendered
        handle.partial_output = _append_limited(handle.partial_output, fragment, _PARTIAL_OUTPUT_LIMIT)
        return fragment

    @staticmethod
    def _set_nonblocking(pipe: Any) -> None:
        if pipe is None:
            return
        fd = pipe.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    @staticmethod
    def _read_available(pipe: Any) -> str:
        return read_available_text(pipe)

    @staticmethod
    def _read_remaining(pipe: Any) -> str:
        return drain_pipe_text(pipe)

    def _resolve_cli_path(self) -> str:
        if self._resolved_cli_path:
            return self._resolved_cli_path

        direct = shutil.which(self.cli_path)
        if direct:
            self._set_resolved_cli_path(direct)
            return direct

        explicit = Path(self.cli_path).expanduser()
        if explicit.exists() and explicit.is_file():
            self._set_resolved_cli_path(str(explicit))
            return str(explicit)

        shell_command = f"command -v {shlex.quote(self.cli_path)}"
        try:
            result = subprocess.run(
                ["bash", "-lic", shell_command],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            result = None
        if result and result.returncode == 0:
            resolved = (result.stdout or "").strip().splitlines()
            if resolved:
                self._set_resolved_cli_path(resolved[-1].strip())
                return self._resolved_cli_path or ""

        return ""

    def _set_resolved_cli_path(self, resolved: str) -> None:
        self._resolved_cli_path = resolved
        parent = str(Path(resolved).expanduser().parent)
        self._resolved_env_path_prefix = parent

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        resolved = self._resolve_cli_path()
        if resolved and self._resolved_env_path_prefix:
            current = env.get("PATH", "")
            env["PATH"] = f"{self._resolved_env_path_prefix}:{current}" if current else self._resolved_env_path_prefix
        return env

    def _render_output(self, stdout: str) -> str:
        raw = (stdout or "").strip()
        if not raw:
            return ""
        if not self.use_stream_json:
            return raw
        rendered = render_qwen_stream_output(raw)
        return rendered or raw


def _append_limited(existing: str, fragment: str, limit: int) -> str:
    combined = f"{existing}{fragment}"
    if len(combined) <= limit:
        return combined
    return combined[-limit:]
