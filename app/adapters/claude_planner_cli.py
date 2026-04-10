"""
Claude CLI adapter for the planner role.

Invokes `claude` CLI via subprocess and returns normalized output.
Supports both synchronous invoke() and async start()/check() modes.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from app.adapters.base import AdapterResponse, BaseAdapter, ProcessHandle
from app.adapters.subprocess_groups import configure_popen_kwargs, terminate_process_handle
from app.planner_stream import consume_stream_fragment
from app.subprocess_io import drain_pipe_text, read_available_text

logger = logging.getLogger("orchestrator.adapter.claude")

_RAW_STDOUT_LIMIT = 800_000
_RAW_STDERR_LIMIT = 200_000
_PARTIAL_OUTPUT_LIMIT = 180_000
_LOG_PROMPT_EXCERPT = 240


class ClaudePlannerCli(BaseAdapter):
    """Adapter that calls Claude Code CLI as the planner."""

    def __init__(
        self,
        cli_path: str = "claude",
        model: str = "opus",
        extra_flags: list[str] | None = None,
        *,
        mode: str = "batch_json",
        use_bare: bool = False,
        no_session_persistence: bool = True,
        capture_stderr_live: bool = True,
        effort: str = "",
    ) -> None:
        self.cli_path = cli_path
        self.model = model
        self.mode = mode or "default"
        self.use_bare = use_bare
        self.no_session_persistence = no_session_persistence
        self.capture_stderr_live = capture_stderr_live
        self.effort = effort
        self._mandatory_flags = []
        self.extra_flags = extra_flags or []

    def name(self) -> str:
        return "claude_planner_cli"

    def is_available(self) -> bool:
        return shutil.which(self.cli_path) is not None

    def runtime_summary(self) -> dict[str, Any]:
        """Return sanitized runtime settings relevant to latency/debugging."""
        settings = self._load_claude_settings()
        env = settings.get("env", {}) if isinstance(settings, dict) else {}
        base_url = str(env.get("ANTHROPIC_BASE_URL", "")).strip()
        resolved_model = str(env.get("ANTHROPIC_DEFAULT_OPUS_MODEL", self.model)).strip()
        return {
            "cli_path": self.cli_path,
            "configured_model": self.model,
            "resolved_model": resolved_model or self.model,
            "mode": self.mode,
            "use_bare": self.use_bare,
            "no_session_persistence": self.no_session_persistence,
            "custom_base_url": base_url,
            "settings_path": str(self._settings_path()),
            "has_custom_backend": bool(base_url),
            "has_model_remap": resolved_model not in {"", self.model},
        }

    # ---------------------------------------------------------------
    # Synchronous mode (blocking, kept for backward compat)
    # ---------------------------------------------------------------

    def invoke(self, prompt: str, timeout: int = 180, **kwargs: Any) -> AdapterResponse:
        """Call Claude CLI with a prompt and return the output (blocking)."""
        cmd, output_mode = self._build_command(
            prompt,
            json_schema=kwargs.get("json_schema"),
        )
        logger.info("Calling Claude CLI (model=%s, timeout=%ds, mode=%s)", self.model, timeout, output_mode)
        logger.debug(
            "Claude CLI command: exec=%s mode=%s prompt_chars=%d excerpt=%r",
            cmd[0],
            output_mode,
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
            )
            duration = time.monotonic() - start
            output = self._render_output(result.stdout or "", output_mode)
            stderr = (result.stderr or "").strip()

            if result.returncode != 0:
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

        except Exception as exc:
            duration = time.monotonic() - start
            logger.error("Claude CLI unexpected error: %s", exc)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=str(exc),
                duration_seconds=duration,
            )

    # ---------------------------------------------------------------
    # Async mode (non-blocking background execution)
    # ---------------------------------------------------------------

    def start(self, prompt: str, **kwargs: Any) -> ProcessHandle:
        """Launch Claude CLI as a background process. Returns immediately."""
        cmd, output_mode = self._build_command(
            prompt,
            json_schema=kwargs.get("json_schema"),
        )
        logger.info("Starting Claude CLI (model=%s, mode=%s) as background process", self.model, output_mode)
        logger.debug(
            "Claude CLI command: exec=%s mode=%s prompt_chars=%d excerpt=%r",
            cmd[0],
            output_mode,
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
            **configure_popen_kwargs(),
        )

        self._set_nonblocking(process.stdout)
        self._set_nonblocking(process.stderr)

        logger.info("Started Claude CLI pid=%d", process.pid)
        return ProcessHandle(
            process=process,
            task_id=kwargs.get("task_id", "planner"),
            worker_id="planner",
            started_at=time.monotonic(),
            metadata={
                "output_mode": output_mode,
                "stream_buffer": "",
                "stream_event_count": 0,
                "raw_stdout": "",
                "raw_stderr": "",
                "command": cmd,
                "pgid": process.pid,
            },
        )

    def terminate(self, handle: ProcessHandle, *, force: bool = False) -> None:
        terminate_process_handle(handle, force=force)

    def check(self, handle: ProcessHandle) -> tuple[str, bool]:
        """Non-blocking check on a running planner process."""
        proc = handle.process
        if proc is None:
            return "", True

        stdout_fragment = self._read_available(proc.stdout)
        stderr_fragment = self._read_available(proc.stderr) if self.capture_stderr_live else ""

        rendered_fragment = self._consume_stdout_fragment(handle, stdout_fragment)
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
                rendered_fragment += self._consume_stdout_fragment(handle, final_stdout)
            if final_stderr:
                handle.partial_error_output = _append_limited(handle.partial_error_output, final_stderr, _RAW_STDERR_LIMIT)
                handle.metadata["raw_stderr"] = _append_limited(
                    handle.metadata.get("raw_stderr", ""),
                    final_stderr,
                    _RAW_STDERR_LIMIT,
                )
            if proc.returncode and proc.returncode != 0:
                logger.warning(
                    "Planner pid=%d exited %d: %s",
                    proc.pid, proc.returncode, handle.partial_error_output[:200],
                )

        return rendered_fragment, is_finished

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _build_command(self, prompt: str, *, json_schema: str | None = None) -> tuple[list[str], str]:
        output_mode = "text"
        cmd = [self.cli_path, "--model", self.model, "-p", prompt]

        if self.use_bare:
            cmd.append("--bare")
        if self.no_session_persistence:
            cmd.append("--no-session-persistence")
        if self.effort:
            cmd.extend(["--effort", self.effort])

        if self.mode == "batch_json":
            output_mode = "stream-json"
            cmd.extend(["--output-format", "stream-json", "--include-partial-messages"])
            # NOTE: --json-schema omitted — forces structured output which
            # non-Anthropic backends (e.g. glm-5.1 via z.ai) don't support.
            # The planner parser extracts JSON from rendered text instead.
        elif self.mode == "batch_text":
            output_mode = "text"
            cmd.extend(["--output-format", "text"])
            # Plain text output — explicit format to ensure stdout streaming

        cmd.extend(self._mandatory_flags)
        cmd.extend(self.extra_flags)
        return cmd, output_mode

    def _consume_stdout_fragment(self, handle: ProcessHandle, fragment: str) -> str:
        if not fragment:
            return ""

        handle.metadata["raw_stdout"] = _append_limited(
            handle.metadata.get("raw_stdout", ""),
            fragment,
            _RAW_STDOUT_LIMIT,
        )
        output_mode = str(handle.metadata.get("output_mode", "text"))
        rendered_fragment = fragment
        if output_mode == "stream-json":
            rendered_fragment, buffer, event_count = consume_stream_fragment(
                fragment,
                str(handle.metadata.get("stream_buffer", "")),
            )
            handle.metadata["stream_buffer"] = buffer
            handle.metadata["stream_event_count"] = int(handle.metadata.get("stream_event_count", 0)) + event_count

        handle.partial_output = _append_limited(handle.partial_output, rendered_fragment, _PARTIAL_OUTPUT_LIMIT)
        return rendered_fragment

    @staticmethod
    def _render_output(stdout: str, output_mode: str) -> str:
        if output_mode != "stream-json":
            return stdout.strip()
        rendered, buffer, _event_count = consume_stream_fragment(stdout, "")
        if buffer.strip():
            rendered += buffer
        return rendered.strip()

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

    @staticmethod
    def _settings_path() -> Path:
        return Path.home() / ".claude" / "settings.json"

    def _load_claude_settings(self) -> dict[str, Any]:
        settings_path = self._settings_path()
        if not settings_path.exists():
            return {}
        try:
            return json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Failed to read Claude settings %s: %s", settings_path, exc)
            return {}


def _append_limited(existing: str, fragment: str, limit: int) -> str:
    combined = f"{existing}{fragment}"
    if len(combined) <= limit:
        return combined
    return combined[-limit:]
