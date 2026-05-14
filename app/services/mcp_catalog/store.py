"""
Persistence for live MCP catalog snapshots and diffs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.run_context import ensure_current_run, resolve_run_dir
from app.services.mcp_catalog.models import McpCatalogDiff, McpCatalogSnapshot
from app.services.mcp_catalog.normalizer import snapshot_from_dict, snapshot_to_dict


class McpCatalogStore:
    def __init__(self, root_dir: str | Path, *, run_id: str = "") -> None:
        self.root_dir = Path(root_dir)
        self.run_id = str(run_id or "").strip()

    @property
    def catalog_root(self) -> Path:
        return self.root_dir / "mcp_catalog"

    @property
    def latest_path(self) -> Path:
        return self.catalog_root / "latest.json"

    @property
    def history_dir(self) -> Path:
        return self.catalog_root / "history"

    @property
    def diffs_dir(self) -> Path:
        return self.catalog_root / "diffs"

    def load_latest(self) -> McpCatalogSnapshot | None:
        if not self.latest_path.exists():
            return None
        payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
        return snapshot_from_dict(payload)

    def save_snapshot(
        self,
        snapshot: McpCatalogSnapshot,
        *,
        diff: McpCatalogDiff | None = None,
        artifact_root: str | Path | None = None,
    ) -> dict[str, Path]:
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.diffs_dir.mkdir(parents=True, exist_ok=True)
        latest_payload = snapshot_to_dict(snapshot)
        latest_path = self.latest_path
        latest_path.write_text(json.dumps(latest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        history_path = self.history_dir / f"{snapshot.schema_hash}.json"
        if not history_path.exists():
            history_path.write_text(json.dumps(latest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result: dict[str, Path] = {
            "latest": latest_path,
            "history": history_path,
        }
        if diff is not None:
            diff_path = self.diffs_dir / f"{snapshot.fetched_at.replace(':', '').replace('.', '')}_{snapshot.schema_hash[:16]}.json"
            diff_path.write_text(json.dumps(asdict(diff), ensure_ascii=False, indent=2), encoding="utf-8")
            result["diff"] = diff_path
        run_path = self._save_run_copy(snapshot)
        if run_path is not None:
            result["run_state"] = run_path
        artifact_path = self._save_artifact_copy(snapshot, artifact_root=artifact_root)
        if artifact_path is not None:
            result["run_artifact"] = artifact_path
        return result

    def _save_run_copy(self, snapshot: McpCatalogSnapshot) -> Path | None:
        if not self.run_id:
            return None
        run_root = resolve_run_dir(self.root_dir, self.run_id)
        if run_root == self.root_dir:
            run_root = ensure_current_run(self.root_dir, self.run_id)
        path = run_root / "mcp_catalog" / "snapshot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot_to_dict(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _save_artifact_copy(self, snapshot: McpCatalogSnapshot, *, artifact_root: str | Path | None) -> Path | None:
        if artifact_root is None:
            return None
        path = Path(artifact_root) / "mcp_catalog" / "snapshot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot_to_dict(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


__all__ = ["McpCatalogStore"]
