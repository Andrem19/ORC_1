"""
LM Studio connection pool with retry logic.

Supports any OpenAI-compatible provider (LM Studio, MiniMax, etc.)
over HTTP or HTTPS.  Cloud providers that lack /v1/models are probed
with a lightweight chat completion instead.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse
import urllib3
from urllib3.util.retry import Retry

logger = logging.getLogger("orchestrator.direct.lmstudio.connection")


def _is_cloud_provider(base_url: str) -> bool:
    """Return True for known cloud API hosts that lack /v1/models."""
    host = urlparse(base_url).hostname or ""
    return any(
        host.endswith(suffix)
        for suffix in ("minimax.io", "minimaxi.com", "openai.com", "anthropic.com")
    )


def _normalize_reasoning_effort(reasoning_effort: str) -> str:
    effort = str(reasoning_effort or "").strip().lower()
    if effort == "off":
        return "none"
    return effort


class LMStudioConnectionPool:
    """HTTP connection pool with retry for LM Studio API."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        model: str = "",
        timeout: int = 60,
        max_connections: int = 10,
        retry_total: int = 3,
    ) -> None:
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
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
        # Cloud providers (MiniMax) do not accept "system" role — convert to user.
        if _is_cloud_provider(self.base_url):
            messages = self._convert_system_to_user(messages)

        body: dict[str, Any] = {
            "messages": messages,
            "temperature": float(temperature),
        }
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        if model:
            body["model"] = str(model)
        normalized_effort = _normalize_reasoning_effort(reasoning_effort)
        if normalized_effort:
            body["reasoning_effort"] = normalized_effort
        if tools:
            body["tools"] = tools
            body["tool_choice"] = str(tool_choice or "auto")
        if extra_body:
            body.update(extra_body)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Cloud providers require bytes body for urllib3 (string body causes 400)
        serialized = json.dumps(body)
        post_body = serialized.encode("utf-8") if _is_cloud_provider(self.base_url) else serialized

        try:
            response = self._pool.request(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                body=post_body,
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

    @staticmethod
    def _convert_system_to_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert system-role messages to user-role for providers that don't support system.

        Merges consecutive system messages into a single user message prefixed
        with [System] to preserve semantic intent.
        """
        result: list[dict[str, Any]] = []
        system_parts: list[str] = []

        def flush_system() -> None:
            nonlocal system_parts
            if system_parts:
                result.append({"role": "user", "content": "[System] " + "\n".join(system_parts)})
                system_parts = []

        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(str(msg.get("content", "")))
            else:
                flush_system()
                result.append(msg)
        flush_system()
        return result

    def health_check(self, timeout: int = 5) -> bool:
        """Verify the API is reachable and has a model ready.

        For local LM Studio: GET /v1/models and confirm at least one model is loaded.
        For cloud providers (MiniMax, etc.): send a minimal chat completion probe since
        they do not expose /v1/models.
        """
        if _is_cloud_provider(self.base_url):
            return self._cloud_health_probe(timeout=timeout)
        return self._local_health_probe(timeout=timeout)

    def _local_health_probe(self, timeout: int = 5) -> bool:
        """GET /v1/models for local LM Studio instances."""
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
            logger.warning("LM Studio health check failed: %s", exc)
            return False

    def _cloud_health_probe(self, timeout: int = 10) -> bool:
        """Minimal chat completion probe for cloud providers without /v1/models.

        Uses stdlib http.client instead of urllib3 to avoid body-encoding
        quirks that cause 400 errors on some cloud APIs (e.g. MiniMax).
        """
        from http.client import HTTPConnection, HTTPSConnection

        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        if self.model:
            payload["model"] = self.model
        body = json.dumps(payload)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        parsed = urlparse(self.base_url.rstrip("/"))
        use_https = parsed.scheme == "https"
        port = parsed.port or (443 if use_https else 80)
        conn_cls = HTTPSConnection if use_https else HTTPConnection
        try:
            conn = conn_cls(parsed.hostname, port, timeout=timeout)
            conn.request("POST", "/v1/chat/completions", body, headers)
            resp = conn.getresponse()
            resp.read()  # drain
            conn.close()
            return resp.status == 200
        except Exception as exc:
            logger.warning("Cloud provider health probe failed: %s", exc)
            return False

    def warm_up(self, timeout: int = 5) -> bool:
        """Pre-flight health check to verify the provider is reachable."""
        ok = self.health_check(timeout=timeout)
        if ok:
            logger.debug("Connection pool warmed up successfully")
        else:
            logger.warning("Connection pool warm-up failed")
        return ok

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.clear()


__all__ = ["LMStudioConnectionPool"]
