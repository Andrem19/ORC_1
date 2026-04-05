"""
Claude CLI adapter for the planner role.

Invokes `claude` CLI via subprocess and returns normalized output.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from typing import Any

from app.adapters.base import AdapterResponse, BaseAdapter

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
        self.extra_flags = extra_flags or []

    def name(self) -> str:
        return "claude_planner_cli"

    def is_available(self) -> bool:
        return shutil.which(self.cli_path) is not None

    def invoke(self, prompt: str, timeout: int = 180, **kwargs: Any) -> AdapterResponse:
        """Call Claude CLI with a prompt and return the output."""
        cmd = [self.cli_path, "--model", self.model, "-p", prompt]
        cmd.extend(self.extra_flags)

        logger.info("Calling Claude CLI (model=%s, timeout=%ds)", self.model, timeout)
        start = time.monotonic()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=None,
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
