"""
LM Studio worker adapter.

Calls LM Studio's OpenAI-compatible API (http://localhost:1234)
using stdlib http.client — no external dependencies required.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.client import HTTPConnection, HTTPException
from typing import Any
from urllib.parse import urlparse

from app.adapters.base import AdapterResponse, BaseAdapter
from app.lmstudio_api import validate_lmstudio_endpoint
from app.adapters.base import ProcessHandle

logger = logging.getLogger("orchestrator.adapter.lmstudio")

# Default LM Studio server address
DEFAULT_BASE_URL = "http://localhost:1234"


class LmStudioWorkerApi(BaseAdapter):
    """Adapter that calls LM Studio's local API as a worker."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "",
        api_key: str = "lm-studio",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str = "",
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body or {}

    def name(self) -> str:
        return "lmstudio_worker_api"

    def is_available(self) -> bool:
        """Check if LM Studio server is reachable and the configured model exists."""
        try:
            ok, models = validate_lmstudio_endpoint(
                base_url=self.base_url,
                model=self.model,
                timeout=5,
            )
            if not ok and self.model and models:
                logger.warning(
                    "LM Studio worker model '%s' not found at %s (available=%s)",
                    self.model,
                    self.base_url,
                    ", ".join(models[:10]),
                )
            return ok
        except Exception:
            return False

    def invoke(self, prompt: str, timeout: int = 300, **kwargs: Any) -> AdapterResponse:
        """Send a prompt to LM Studio and return the model's response."""
        system_prompt = kwargs.get("system_prompt", (
            "You are a worker agent. Complete the assigned task and "
            "return a structured JSON result with fields: status, summary, "
            "artifacts, confidence, error."
        ))

        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.model:
            body["model"] = self.model
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        body.update(self.extra_body)

        logger.info(
            "Calling LM Studio API %s (model=%s, timeout=%ds)",
            self.base_url, self.model or "default", timeout,
        )
        start = time.monotonic()

        try:
            parsed = urlparse(self.base_url)
            conn = HTTPConnection(
                parsed.hostname,
                parsed.port or 1234,
                timeout=timeout,
            )

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            conn.request("POST", "/v1/chat/completions", json.dumps(body), headers)
            resp = conn.getresponse()
            resp_body = resp.read().decode("utf-8")
            conn.close()
            duration = time.monotonic() - start

            if resp.status != 200:
                logger.warning(
                    "LM Studio returned HTTP %d: %s",
                    resp.status, resp_body[:300],
                )
                return AdapterResponse(
                    success=False,
                    raw_output="",
                    exit_code=resp.status,
                    error=f"HTTP {resp.status}: {resp_body[:500]}",
                    duration_seconds=duration,
                )

            data = json.loads(resp_body)
            content = self._extract_content(data)
            if not content.strip():
                reasoning = self._extract_reasoning(data)
                finish_reason = self._extract_finish_reason(data)
                logger.warning(
                    "LM Studio returned empty worker content (finish_reason=%s, reasoning_chars=%d)",
                    finish_reason or "<unknown>",
                    len(reasoning),
                )
                return AdapterResponse(
                    success=False,
                    raw_output=reasoning[:1000],
                    exit_code=0,
                    error=(
                        "LM Studio returned empty assistant content for worker task. "
                        "Use reasoning_effort='none' or increase max_tokens if the model is spending "
                        "the budget on reasoning."
                    ),
                    duration_seconds=duration,
                )

            logger.info(
                "LM Studio completed in %.1fs, output length: %d",
                duration, len(content),
            )
            return AdapterResponse(
                success=True,
                raw_output=content,
                exit_code=0,
                duration_seconds=duration,
            )

        except json.JSONDecodeError as e:
            duration = time.monotonic() - start
            logger.error("LM Studio returned invalid JSON: %s", e)
            return AdapterResponse(
                success=False,
                raw_output=resp_body if "resp_body" in dir() else "",
                exit_code=-1,
                error=f"Invalid JSON from LM Studio: {e}",
                duration_seconds=duration,
            )

        except (HTTPException, ConnectionError, OSError) as e:
            duration = time.monotonic() - start
            logger.error("LM Studio connection error: %s", e)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=f"Connection error: {e}",
                timed_out=isinstance(e, TimeoutError),
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.monotonic() - start
            logger.error("LM Studio unexpected error: %s", e)
            return AdapterResponse(
                success=False,
                raw_output="",
                exit_code=-1,
                error=str(e),
                duration_seconds=duration,
            )

    def start(self, prompt: str, **kwargs: Any) -> ProcessHandle:
        """Run invoke() in a background thread for worker-service compatibility."""
        handle = ProcessHandle(
            process=None,
            task_id=kwargs.get("task_id", "worker"),
            worker_id=kwargs.get("worker_id", "unknown"),
            metadata={
                "thread_done": False,
                "response": None,
                "timeout": kwargs.get("timeout"),
                "cancelled": False,
                "result_emitted": False,
            },
        )

        def _runner() -> None:
            timeout = int(handle.metadata.get("timeout") or kwargs.get("timeout") or 300)
            invoke_kwargs: dict[str, Any] = {}
            system_prompt = kwargs.get("system_prompt")
            if system_prompt is not None:
                invoke_kwargs["system_prompt"] = system_prompt
            response = self.invoke(prompt, timeout=timeout, **invoke_kwargs)
            handle.metadata["response"] = response
            handle.metadata["thread_done"] = True

        worker = threading.Thread(target=_runner, daemon=True)
        handle.metadata["thread"] = worker
        worker.start()
        return handle

    def check(self, handle: ProcessHandle) -> tuple[str, bool]:
        """Poll background LM Studio invocation."""
        if handle.metadata.get("cancelled"):
            return "", True
        if not handle.metadata.get("thread_done"):
            return "", False
        if handle.metadata.get("result_emitted"):
            return "", True

        response = handle.metadata.get("response")
        handle.metadata["result_emitted"] = True
        if not isinstance(response, AdapterResponse):
            handle.partial_error_output += "LM Studio worker response missing"
            return "", True
        if response.success:
            handle.partial_output += response.raw_output
            return response.raw_output, True
        handle.partial_error_output += response.error
        return response.raw_output or "", True

    def terminate(self, handle: ProcessHandle) -> None:
        """Mark the async request as cancelled. HTTP request itself is not abortable."""
        handle.metadata["cancelled"] = True

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        """Extract the text content from an OpenAI-format chat completion response."""
        try:
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "")
        except (IndexError, KeyError, TypeError):
            pass
        return json.dumps(data)

    @staticmethod
    def _extract_reasoning(data: dict[str, Any]) -> str:
        try:
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return str(message.get("reasoning_content", "") or "")
        except (IndexError, KeyError, TypeError):
            pass
        return ""

    @staticmethod
    def _extract_finish_reason(data: dict[str, Any]) -> str:
        try:
            choices = data.get("choices", [])
            if choices:
                return str(choices[0].get("finish_reason", "") or "")
        except (IndexError, KeyError, TypeError):
            pass
        return ""
