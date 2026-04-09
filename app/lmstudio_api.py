"""
Shared LM Studio HTTP helpers.

Centralizes lightweight availability and model-discovery checks so all
LM Studio callers validate the same way.
"""

from __future__ import annotations

import json
from http.client import HTTPConnection
from urllib.parse import urlparse


def fetch_model_ids(base_url: str, timeout: int = 5) -> tuple[bool, list[str]]:
    """Return (server_ok, model_ids) from LM Studio /v1/models."""
    parsed = urlparse(base_url.rstrip("/"))
    conn = HTTPConnection(parsed.hostname, parsed.port or 1234, timeout=timeout)
    try:
        conn.request("GET", "/v1/models")
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


def validate_lmstudio_endpoint(
    *,
    base_url: str,
    model: str = "",
    timeout: int = 5,
) -> tuple[bool, list[str]]:
    """Return (ok, discovered_models) and optionally validate one configured model."""
    ok, models = fetch_model_ids(base_url, timeout=timeout)
    if not ok:
        return False, models
    configured_model = model.strip()
    if configured_model and configured_model not in models:
        return False, models
    return True, models
