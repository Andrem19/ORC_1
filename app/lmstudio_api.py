"""
Shared HTTP helpers for OpenAI-compatible API endpoints.

Centralizes lightweight availability and model-discovery checks.
Originally built for LM Studio, now supports any OpenAI-compatible
provider (LM Studio, MiniMax, etc.) over HTTP or HTTPS.
"""

from __future__ import annotations

import json
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlparse


def _is_cloud_provider(base_url: str) -> bool:
    """Return True for known cloud API hosts that lack /v1/models."""
    host = urlparse(base_url).hostname or ""
    return any(
        host.endswith(suffix)
        for suffix in ("minimax.io", "minimaxi.com", "openai.com", "anthropic.com")
    )


def fetch_model_ids(
    base_url: str,
    timeout: int = 5,
    api_key: str = "",
    model: str = "",
) -> tuple[bool, list[str]]:
    """Return (server_ok, model_ids) from /v1/models endpoint.

    For cloud providers that do not expose /v1/models, falls back to a
    lightweight chat completion probe that confirms reachability and
    returns the configured model name (if any).
    """
    parsed = urlparse(base_url.rstrip("/"))
    use_https = parsed.scheme == "https"
    port = parsed.port or (443 if use_https else 80)
    conn_cls = HTTPSConnection if use_https else HTTPConnection

    # Cloud providers (MiniMax, OpenAI, etc.) may not expose /v1/models.
    # Use a lightweight chat probe instead.
    if _is_cloud_provider(base_url):
        return _cloud_health_probe(base_url, api_key=api_key, timeout=timeout, model=model)

    conn = conn_cls(parsed.hostname, port, timeout=timeout)
    try:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        conn.request("GET", "/v1/models", headers=headers)
        resp = conn.getresponse()
        payload = resp.read().decode("utf-8")
        if resp.status != 200:
            return False, []
        data = json.loads(payload)
        items = data.get("data", [])
        models = [
            str(item.get("id", "")).strip()
            for item in items
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        ]
        return True, models
    finally:
        conn.close()


def _cloud_health_probe(
    base_url: str,
    *,
    api_key: str = "",
    timeout: int = 10,
    model: str = "",
) -> tuple[bool, list[str]]:
    """Probe a cloud API with a minimal chat completion request."""
    parsed = urlparse(base_url.rstrip("/"))
    use_https = parsed.scheme == "https"
    port = parsed.port or (443 if use_https else 80)
    conn_cls = HTTPSConnection if use_https else HTTPConnection
    conn = conn_cls(parsed.hostname, port, timeout=timeout)
    try:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        probe_model = model.strip() or "MiniMax-M2.7"
        body = json.dumps({
            "model": probe_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        })
        conn.request("POST", "/v1/chat/completions", body, headers)
        resp = conn.getresponse()
        resp.read()  # drain response
        return resp.status == 200, []
    except Exception:
        return False, []
    finally:
        conn.close()


def validate_lmstudio_endpoint(
    *,
    base_url: str,
    model: str = "",
    timeout: int = 5,
    api_key: str = "",
) -> tuple[bool, list[str]]:
    """Return (ok, discovered_models) and optionally validate one configured model.

    When the provider does not expose a model list (cloud providers via
    ``_cloud_health_probe``), the model validation step is skipped — a
    successful health probe is sufficient.
    """
    ok, models = fetch_model_ids(base_url, timeout=timeout, api_key=api_key, model=model)
    if not ok:
        return False, models
    # Cloud probes return ok=True but no model list — skip model validation.
    if not models:
        return True, models
    configured_model = model.strip()
    if configured_model and configured_model not in models:
        return False, models
    return True, models
