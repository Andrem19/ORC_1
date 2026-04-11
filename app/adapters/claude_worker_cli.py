"""
Claude Code CLI adapter for the worker role.

Invokes `claude` CLI via subprocess and returns normalized output.
Supports --yolo for auto-approved tool use and --exclude-tools for safety.
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
from app.adapters.subprocess_groups import configure_popen_kwargs, terminate_process_handle
from app.planner_stream import consume_stream_fragment
from app.services.direct_execution.stream_tool_counter import (
    count_tool_calls_from_single_event,
    count_tool_calls_from_stream_json,
)
from app.subprocess_io import drain_pipe_text, read_available_text

logger = logging.getLogger("orchestrator.adapter.claude_worker")

_RAW_STDOUT_LIMIT = 600_000
_RAW_STDERR_LIMIT = 200_000
_PARTIAL_OUTPUT_LIMIT = 300_000
_LOG_PROMPT_EXCERPT = 240


class ClaudeWorkerCli(BaseAdapter):
    """Adapter that calls Claude Code CLI as a worker with tool access."""

    def __init__(
        self,
        cli_path: str = "claude",
        model: str = "",
        extra_flags: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        *,
        allow_tool_use: bool = True,
    ) -> None:
        self.cli_path = cli_path
        self.model = model
        self.extra_flags = extra_flags or []
        self.exclude_tools = _dedupe_preserve_order(exclude_tools or [])
        self.allow_tool_use = allow_tool_use
        self._resolved_cli_path: str | None = None
        self._resolved_env_path_prefix = ""

    def name(self) -> str:
        return "claude_worker_cli"

    def is_available(self) -> bool:
        return bool(self._resolve_cli_path())

    def _build_command(self, prompt: str, **kwargs: Any) -> list[str]:
        cmd = [self._resolve_cli_path() or self.cli_path]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.allow_tool_use:
            cmd.append("--dangerously-skip-permissions")
            runtime_exclude = _dedupe_preserve_order(kwargs.get("exclude_tools", []) or [])
            excluded = _dedupe_preserve_order([*self.exclude_tools, *runtime_exclude])
            if excluded:
                cmd.append("--disallowedTools")
                cmd.extend(excluded)
        cmd.extend(["--output-format", "stream-json"])
        cmd.extend(["-p", prompt])
        cmd.extend(self.extra_flags)
        return cmd

    def invoke(self, prompt: str, timeout: int = 300, **kwargs: Any) -> AdapterResponse:
        """Call Claude Code CLI with a prompt and return the output (blocking)."""
        cmd = self._build_command(prompt, **kwargs)

        logger.info("Calling Claude CLI worker (timeout=%ds)", timeout)
        logger.debug(
            "Claude CLI command: %s (prompt_chars=%d excerpt=%r)",
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
                logger.warning("Claude CLI returned exit code %d: %s", result.returncode, stderr[:200])
                return AdapterResponse(
                    success=False,
                    raw_output=output,
                    exit_code=result.returncode,
                    error=stderr[:500],
                    duration_seconds=duration,
                    finish_reason="process_error",
                    metadata={
                        "output_mode": "stream-json",
                        "raw_stdout_present": bool(result.stdout),
                        "raw_stderr_present": bool(result.stderr),
                    },
                )

            counted = count_tool_calls_from_stream_json(result.stdout or "", "claude_cli")
            logger.info(
                "Claude CLI completed in %.1fs, output length: %d, tool_calls: %d",
                duration, len(output), counted.tool_call_count,
            )
            return AdapterResponse(
                success=True,
                raw_output=output,
                exit_code=0,
                duration_seconds=duration,
                finish_reason="completed",
                metadata={
                    "output_mode": "stream-json",
                    "raw_stdout_present": bool(result.stdout),
                    "raw_stderr_present": bool(result.stderr),
                    "tool_call_count": counted.tool_call_count,
                    "tool_names": counted.tool_names,
                },
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
                finish_reason="timeout",
                metadata={"output_mode": "stream-json"},
            )

        except FileNotFoundError:
            logger.error("Claude CLI not found at: %s", self.cli_path)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=f"CLI not found: {self.cli_path}",
                finish_reason="adapter_missing",
            )

        except Exception as exc:
            duration = time.monotonic() - start
            logger.error("Claude CLI unexpected error: %s", exc)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=str(exc),
                duration_seconds=duration,
                finish_reason="adapter_exception",
            )

    # ---------------------------------------------------------------
    # Async background execution (start / check)
    # ---------------------------------------------------------------

    def start(self, prompt: str, **kwargs: Any) -> ProcessHandle:
        """Launch Claude CLI as a background process. Returns immediately."""
        cmd = self._build_command(prompt, **kwargs)

        logger.info("Starting Claude CLI worker as background process")
        logger.debug(
            "Claude CLI command: %s (prompt_chars=%d excerpt=%r)",
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
            **configure_popen_kwargs(),
        )

        self._set_nonblocking(process.stdout)
        self._set_nonblocking(process.stderr)

        logger.info("Started Claude CLI pid=%d", process.pid)
        return ProcessHandle(
            process=process,
            task_id=kwargs.get("task_id", ""),
            worker_id=kwargs.get("worker_id", ""),
            started_at=time.monotonic(),
            metadata={
                "raw_stdout": "",
                "raw_stderr": "",
                "output_mode": "stream-json",
                "stream_buffer": "",
                "stream_event_count": 0,
                "pgid": process.pid,
            },
        )

    def terminate(self, handle: ProcessHandle, *, force: bool = False) -> None:
        terminate_process_handle(handle, force=force)

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
        """Process raw stdout fragment — parse stream-json."""
        if not fragment:
            return ""
        handle.metadata["raw_stdout"] = _append_limited(
            handle.metadata.get("raw_stdout", ""),
            fragment,
            _RAW_STDOUT_LIMIT,
        )
        rendered, buffer, event_count = consume_stream_fragment(
            fragment,
            str(handle.metadata.get("stream_buffer", "")),
        )
        handle.metadata["stream_buffer"] = buffer
        handle.metadata["stream_event_count"] = int(handle.metadata.get("stream_event_count", 0)) + event_count
        handle.partial_output = _append_limited(handle.partial_output, rendered, _PARTIAL_OUTPUT_LIMIT)
        # Incremental tool call counting from raw stream lines
        for raw_line in fragment.splitlines():
            tool_names = count_tool_calls_from_single_event(raw_line)
            if tool_names:
                handle.metadata["tool_call_count"] = int(handle.metadata.get("tool_call_count", 0)) + len(tool_names)
                existing = handle.metadata.get("tool_names") or []
                handle.metadata["tool_names"] = existing + tool_names
        return rendered

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
        rendered, _, _ = consume_stream_fragment(raw, "")
        return rendered or raw


def _dedupe_preserve_order(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _append_limited(existing: str, fragment: str, limit: int) -> str:
    combined = f"{existing}{fragment}"
    if len(combined) <= limit:
        return combined
    return combined[-limit:]
