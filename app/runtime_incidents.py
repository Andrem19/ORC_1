"""
Local incident persistence for runtime anomalies.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.run_context import ensure_current_run, resolve_run_dir


class LocalIncidentStore:
    """Write incident artifacts even when MCP incident capture is unavailable."""

    def __init__(self, root_dir: str | Path, *, run_id: str = "") -> None:
        self.root_dir = Path(root_dir)
        self.run_id = run_id

    def record(
        self,
        *,
        summary: str,
        metadata: dict[str, Any] | None = None,
        source: str = "runtime",
        severity: str = "medium",
    ) -> Path:
        run_dir = resolve_run_dir(self.root_dir, self.run_id)
        if run_dir == self.root_dir and self.run_id:
            run_dir = ensure_current_run(self.root_dir, self.run_id)
        incidents_dir = run_dir / "incidents"
        incidents_dir.mkdir(parents=True, exist_ok=True)
        incident_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        path = incidents_dir / f"{incident_id}.json"
        payload = {
            "incident_id": incident_id,
            "source": source,
            "severity": severity,
            "summary": summary,
            "metadata": _sanitize(metadata or {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _sanitize(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return _shorten(value)
    if isinstance(value, dict):
        return {str(k): _sanitize(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, depth=depth + 1) for item in value[:50]]
    return _shorten(value)


def _shorten(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if len(text) <= 2000:
        return text
    return f"{text[:1200]} ... {text[-400:]}"
