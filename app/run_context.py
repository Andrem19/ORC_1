"""
Helpers for run-scoped artifact layout.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CURRENT_RUN_FILENAME = "current_run.json"
CURRENT_LINK_NAME = "current"
RUNS_DIRNAME = "runs"


def generate_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def read_current_run_id(root: str | Path) -> str:
    payload = read_current_run_payload(root)
    return str(payload.get("run_id", "") or "")


def read_current_run_payload(root: str | Path) -> dict[str, Any]:
    path = Path(root) / CURRENT_RUN_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def ensure_current_run(root: str | Path, run_id: str) -> Path:
    root_path = Path(root)
    run_path = root_path / RUNS_DIRNAME / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "active_path": str(run_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (root_path / CURRENT_RUN_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _refresh_current_symlink(root_path=root_path, target=run_path)
    return run_path


def resolve_run_dir(root: str | Path, run_id: str = "") -> Path:
    root_path = Path(root)
    resolved_run_id = run_id or read_current_run_id(root_path)
    if not resolved_run_id:
        return root_path
    return root_path / RUNS_DIRNAME / resolved_run_id


def build_state_pointer(*, run_id: str, state_path: Path) -> dict[str, Any]:
    return {
        "pointer_type": "run_state",
        "run_id": run_id,
        "state_path": str(state_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_pointed_path(data: dict[str, Any], *, pointer_type: str) -> Path | None:
    if str(data.get("pointer_type", "")) != pointer_type:
        return None
    raw = str(data.get("state_path", "") or data.get("path", "") or "")
    return Path(raw) if raw else None


def _refresh_current_symlink(*, root_path: Path, target: Path) -> None:
    link_path = root_path / CURRENT_LINK_NAME
    try:
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_dir() and not link_path.is_symlink():
                return
            link_path.unlink(missing_ok=True)
        relative_target = os.path.relpath(target, start=root_path)
        link_path.symlink_to(relative_target, target_is_directory=True)
    except OSError:
        # Symlink support is best-effort only.
        return
