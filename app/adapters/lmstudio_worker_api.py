"""
LM Studio worker adapter.

Calls LM Studio's OpenAI-compatible API (http://localhost:1234)
using stdlib http.client — no external dependencies required.
"""

from __future__ import annotations

import json
import logging
import time
from http.client import HTTPConnection, HTTPException
from typing import Any
from urllib.parse import urlparse

from app.adapters.base import AdapterResponse, BaseAdapter

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
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}

    def name(self) -> str:
        return "lmstudio_worker_api"

    def is_available(self) -> bool:
        """Check if LM Studio server is reachable by listing models."""
        try:
            parsed = urlparse(self.base_url)
            conn = HTTPConnection(parsed.hostname, parsed.port or 1234, timeout=5)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            conn.close()
            return resp.status == 200
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
