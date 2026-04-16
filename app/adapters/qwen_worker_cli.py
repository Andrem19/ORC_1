"""
Qwen Code CLI adapter for the worker role.

Invokes `qwen-code` CLI via subprocess and returns normalized output.
Supports both synchronous invoke() and async start()/check() modes.
"""

from __future__ import annotations

import fcntl
import json
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
from app.qwen_stream import render_qwen_stream_output
from app.services.direct_execution.stream_tool_counter import (
    count_tool_calls_from_single_event,
    count_tool_calls_from_stream_json,
)
from app.subprocess_io import drain_pipe_text, read_available_text

logger = logging.getLogger("orchestrator.adapter.qwen")

_RAW_STDOUT_LIMIT = 600_000
_RAW_STDERR_LIMIT = 200_000
_PARTIAL_OUTPUT_LIMIT = 300_000
_LOG_PROMPT_EXCERPT = 240
_BROKER_SAFE_EXCLUDED_TOOLS = (
    "run_shell_command",
    "read_file",
    "edit",
    "write_file",
    "grep_search",
    "glob",
    "list_directory",
    "web_fetch",
    "task",
    "skill",
)


class QwenWorkerCli(BaseAdapter):
    """Adapter that calls Qwen Code CLI as a worker."""

    def __init__(
        self,
        cli_path: str = "qwen",
        extra_flags: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        *,
        use_stream_json: bool = True,
        allow_tool_use: bool = False,
    ) -> None:
        self.cli_path = cli_path
        self.extra_flags = extra_flags or []
        self.exclude_tools = _dedupe_preserve_order(exclude_tools or [])
        self.use_stream_json = use_stream_json
        self.allow_tool_use = allow_tool_use
        self._resolved_cli_path: str | None = None
        self._resolved_env_path_prefix = ""
        self._tool_registry_cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def name(self) -> str:
        return "qwen_worker_cli"

    def is_available(self) -> bool:
        return bool(self._resolve_cli_path())

    def _build_command(self, prompt: str, **kwargs: Any) -> list[str]:
        cmd = [self._resolve_cli_path() or self.cli_path]
        allow_tool_use = bool(kwargs.get("allow_tool_use", self.allow_tool_use))
        runtime_exclude = _dedupe_preserve_order(kwargs.get("exclude_tools", []) or [])
        excluded_tools = _dedupe_preserve_order([*_BROKER_SAFE_EXCLUDED_TOOLS, *self.exclude_tools, *runtime_exclude])
        if allow_tool_use:
            cmd.append("--yolo")
            if excluded_tools:
                cmd.extend(["--exclude-tools", ",".join(excluded_tools)])
        else:
            # Non-tool worker turns must return one JSON action on stdout.
            # Disable local tools/extensions so the CLI cannot drift into
            # workspace side effects like file writes or shell execution.
            cmd.extend(["--exclude-tools", ",".join(excluded_tools), "-e", "none"])
        if self.use_stream_json:
            cmd.extend(["-o", "stream-json"])
        cmd.extend(["-p", prompt])
        cmd.extend(self.extra_flags)
        return cmd

    def preflight_tool_registry(
        self,
        *,
        required_tools: list[str],
        timeout: int | float = 60,
    ) -> dict[str, Any]:
        normalized_required = tuple(sorted({str(item).strip() for item in required_tools if str(item).strip()}))
        cached = self._tool_registry_cache.get(normalized_required)
        if cached is not None:
            return dict(cached)
        if not self._resolve_cli_path():
            result = {
                "available": False,
                "visible_tools": [],
                "exact_visible_tools": [],
                "canonical_to_visible": {},
                "missing_required_tools": list(normalized_required),
                "reason": "qwen_cli_unavailable",
            }
            self._tool_registry_cache[normalized_required] = dict(result)
            return result
        probe_timeout = max(1, int(timeout or 60))
        response = self.invoke(
            (
                "Return only JSON with shape "
                '{"visible_tools":["tool_a","tool_b"]}'
                " listing the exact currently visible tool names. "
                "Do not call any tool. Do not explain."
            ),
            timeout=probe_timeout,
            allow_tool_use=False,
        )
        if not response.success:
            return {
                "available": True,
                "visible_tools": [],
                "exact_visible_tools": [],
                "canonical_to_visible": {},
                "missing_required_tools": [],
                "reason": _preflight_probe_reason(response),
                "preflight_inconclusive": True,
            }
        exact_visible_tools, canonical_to_visible = _extract_visible_tool_registry(response.raw_output)
        visible_tools = set(canonical_to_visible)
        missing = [name for name in normalized_required if name not in visible_tools]
        result = {
            "available": not missing,
            "visible_tools": sorted(visible_tools),
            "exact_visible_tools": exact_visible_tools,
            "canonical_to_visible": canonical_to_visible,
            "missing_required_tools": missing,
            "reason": "" if not missing else "missing:" + ",".join(missing),
            "preflight_inconclusive": False,
        }
        self._tool_registry_cache[normalized_required] = dict(result)
        return result

    def invoke(self, prompt: str, timeout: int = 300, **kwargs: Any) -> AdapterResponse:
        """Call Qwen Code CLI with a prompt and return the output (blocking)."""
        cmd = self._build_command(prompt, **kwargs)

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
                    finish_reason="process_error",
                    metadata={
                        "output_mode": "stream-json" if self.use_stream_json else "text",
                        "raw_stdout_present": bool(result.stdout),
                        "raw_stderr_present": bool(result.stderr),
                    },
                )

            counted = count_tool_calls_from_stream_json(result.stdout or "", "qwen_cli")
            logger.info(
                "Qwen CLI completed in %.1fs, output length: %d, tool_calls: %d",
                duration, len(output), counted.tool_call_count,
            )
            return AdapterResponse(
                success=True,
                raw_output=output,
                exit_code=0,
                duration_seconds=duration,
                finish_reason="completed",
                metadata={
                    "output_mode": "stream-json" if self.use_stream_json else "text",
                    "raw_stdout_present": bool(result.stdout),
                    "raw_stderr_present": bool(result.stderr),
                    "tool_call_count": counted.tool_call_count,
                    "tool_names": counted.tool_names,
                },
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
                finish_reason="timeout",
                metadata={"output_mode": "stream-json" if self.use_stream_json else "text"},
            )

        except FileNotFoundError:
            logger.error("Qwen CLI not found at: %s", self.cli_path)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=f"CLI not found: {self.cli_path}",
                finish_reason="adapter_missing",
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
                finish_reason="adapter_exception",
            )

    # ---------------------------------------------------------------
    # Async background execution (start / check)
    # ---------------------------------------------------------------

    def start(self, prompt: str, **kwargs: Any) -> ProcessHandle:
        """Launch Qwen CLI as a background process. Returns immediately."""
        cmd = self._build_command(prompt, **kwargs)
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
            **configure_popen_kwargs(),
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
            # Incremental tool call counting from raw stream lines
            for raw_line in fragment.splitlines():
                tool_names = count_tool_calls_from_single_event(raw_line)
                if tool_names:
                    handle.metadata["tool_call_count"] = int(handle.metadata.get("tool_call_count", 0)) + len(tool_names)
                    existing = handle.metadata.get("tool_names") or []
                    handle.metadata["tool_names"] = existing + tool_names
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


def _extract_visible_tool_registry(raw_output: str) -> tuple[list[str], dict[str, str]]:
    text = str(raw_output or "").strip()
    if not text:
        return [], {}
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [], {}
    values = payload.get("visible_tools") if isinstance(payload, dict) else []
    if not isinstance(values, list):
        return [], {}
    ordered_exact: list[str] = []
    canonical_to_visible: dict[str, str] = {}
    for item in values:
        exact_name = str(item or "").strip()
        if not exact_name:
            continue
        if exact_name not in ordered_exact:
            ordered_exact.append(exact_name)
        canonical_name = exact_name
        if canonical_name.startswith("mcp__dev_space1__"):
            canonical_name = canonical_name.split("mcp__dev_space1__", 1)[1]
        canonical_to_visible.setdefault(canonical_name, exact_name)
    return ordered_exact, canonical_to_visible


def _preflight_probe_reason(response: AdapterResponse) -> str:
    if getattr(response, "timed_out", False):
        return "probe_timeout"
    error_text = str(getattr(response, "error", "") or "").strip()
    if error_text:
        return f"probe_failed:{error_text[:120]}"
    return "probe_inconclusive"
