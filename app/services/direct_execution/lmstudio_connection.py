"""
LM Studio connection pool with retry logic.
"""

from __future__ import annotations

import json
import logging
from typing import Any
import urllib3
from urllib3.util.retry import Retry

logger = logging.getLogger("orchestrator.direct.lmstudio.connection")


class LMStudioConnectionPool:
    """HTTP connection pool with retry for LM Studio API."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        timeout: int = 60,
        max_connections: int = 10,
        retry_total: int = 3,
    ) -> None:
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.timeout = int(timeout or 60)
        self._pool = urllib3.PoolManager(
            num_pools=1,
            maxsize=max_connections,
            block=False,
            retries=Retry(
                total=retry_total,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["POST"],
            ),
        )

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str = "",
        reasoning_effort: str = "",
        extra_body: dict[str, Any] | None = None,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """Send chat completion request with connection pool and retry."""
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": float(temperature),
        }
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        if model:
            body["model"] = str(model)
        if reasoning_effort and reasoning_effort not in ("none", "off"):
            body["reasoning_effort"] = str(reasoning_effort)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = str(tool_choice or "auto")
        if extra_body:
            body.update(extra_body)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = self._pool.urlopen(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                body=json.dumps(body),
                headers=headers,
                timeout=urllib3.Timeout(connect=self.timeout, read=self.timeout),
                retries=False,  # Use pool-level retry instead
            )
            raw = response.data.decode("utf-8")
            if response.status != 200:
                # Fast-fail on model crash — no retries will help
                raw_lower = raw.lower()
                if response.status == 400 and (
                    "model has crashed" in raw_lower
                    or "exit code: null" in raw_lower
                    or "exit code: 1" in raw_lower
                ):
                    logger.error("LM Studio model crash detected: %s", raw[:500])
                    return {"error": f"lmstudio_model_crash: {raw[:500]}"}
                logger.warning(
                    f"LM Studio API returned {response.status}: {raw[:800]}",
                )
                return {"error": f"HTTP {response.status}: {raw[:800]}"}
            return json.loads(raw)
        except urllib3.exceptions.HTTPError as exc:
            logger.warning(f"LM Studio HTTP error: {exc}")
            return {"error": str(exc)}
        except Exception as exc:
            logger.error(f"LM Studio unexpected error: {exc}")
            return {"error": str(exc)}

    def health_check(self, timeout: int = 5) -> bool:
        """Lightweight GET /v1/models to verify LM Studio is reachable and has a model loaded."""
        try:
            resp = self._pool.urlopen(
                "GET",
                f"{self.base_url}/v1/models",
                headers={"Content-Type": "application/json"},
                timeout=urllib3.Timeout(connect=timeout, read=timeout),
                retries=False,
            )
            if resp.status != 200:
                return False
            # Verify at least one model is loaded (crash unloads the model)
            try:
                body = json.loads(resp.data.decode("utf-8"))
                models = body.get("data") or []
                if not models:
                    logger.warning("LM Studio reachable but no models loaded (possible crash)")
                    return False
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # Parse failure is OK — server is reachable
            return True
        except Exception as exc:
            logger.warning(f"LM Studio health check failed: {exc}")
            return False

    def warm_up(self, timeout: int = 5) -> bool:
        """Pre-flight request to /v1/models to establish the connection pool."""
        ok = self.health_check(timeout=timeout)
        if ok:
            logger.debug("LM Studio connection pool warmed up successfully")
        else:
            logger.warning("LM Studio connection pool warm-up failed")
        return ok

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.clear()


__all__ = ["LMStudioConnectionPool"]
