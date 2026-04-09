"""
Persistence helpers for canonical report artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.run_context import resolve_run_dir


class ReportStore:
    def __init__(self, *, plan_root: str | Path, run_id: str) -> None:
        self.plan_root = Path(plan_root)
        self.run_id = run_id

    @property
    def active_root(self) -> Path:
        return resolve_run_dir(self.plan_root, self.run_id)

    def save_plan_report(self, *, plan_id: str, payload: Any, markdown: str) -> tuple[Path, Path]:
        return self._save_pair(self.active_root / "plan_reports" / plan_id, payload, markdown)

    def save_sequence_report(self, *, sequence_id: str, payload: Any, markdown: str) -> tuple[Path, Path]:
        return self._save_pair(self.active_root / "sequence_reports" / sequence_id, payload, markdown)

    def save_run_report(self, *, payload: Any, markdown: str) -> tuple[Path, Path]:
        root = self.active_root / "run_report"
        return self._save_pair(root, payload, markdown)

    def _save_pair(self, root: Path, payload: Any, markdown: str) -> tuple[Path, Path]:
        json_path = root.with_suffix(".json")
        md_path = root.with_suffix(".md")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
        return json_path, md_path
