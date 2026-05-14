"""
Reset manager — handles archive + clear logic for startup_mode.

Supports three modes:
  - resume: no action (default)
  - reset: clear state + plans, preserve research_context
  - reset_all: clear everything including research_context

Always archives files to state/archive/<timestamp>/ before deletion.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from app.state_store import StateStore

logger = logging.getLogger("orchestrator.reset")


class _PlanStoreLike(Protocol):
    plans_dir: Path

    def clear_all(self) -> None:
        ...


class ResetManager:
    def __init__(
        self,
        state_dir: str,
        state_store: StateStore,
        plan_store: _PlanStoreLike | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_store = state_store
        self.plan_store = plan_store

    def perform_reset(self, mode: str) -> None:
        """Execute reset: archive current files, then clear based on mode."""
        if mode not in ("reset", "reset_all"):
            raise ValueError(f"Invalid reset mode: {mode}")

        # 1. Archive current files before any deletion
        archive_dir = self._create_archive()
        if archive_dir:
            logger.info("Archived current state to %s", archive_dir)
        else:
            logger.warning("Nothing to archive — no existing state files found")

        # 2. Clear orchestrator state
        self.state_store.clear()

        # 3. Clear plans and reports
        if self.plan_store:
            self.plan_store.clear_all()

        # 4. Conditionally clear research context
        if mode == "reset_all":
            self._clear_research_context()

        logger.info(
            "Reset complete (mode=%s): state cleared, plans cleared, research_context %s",
            mode,
            "cleared" if mode == "reset_all" else "preserved",
        )

    def _create_archive(self) -> Path | None:
        """Create a timestamped archive with copies of all current files."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        archive_dir = self.state_dir / "archive" / ts

        has_files = False

        # Archive state file
        if self.state_store.archive_to(archive_dir):
            has_files = True

        # Archive plans
        if self.plan_store and getattr(self.plan_store, "plans_dir", None):
            plans_dir = Path(self.plan_store.plans_dir)
            if plans_dir.exists():
                archive_plans_dir = archive_dir / "plans"
                shutil.copytree(plans_dir, archive_plans_dir, dirs_exist_ok=True)
                if any(archive_plans_dir.rglob("*")):
                    has_files = True

        # Archive research context if it exists
        rc_path = self.state_dir / "research_context.json"
        if rc_path.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rc_path, archive_dir / "research_context.json")
            has_files = True

        return archive_dir if has_files else None

    def _clear_research_context(self) -> None:
        """Delete research_context.json."""
        rc_path = self.state_dir / "research_context.json"
        if rc_path.exists():
            rc_path.unlink()
            logger.info("Removed research context: %s", rc_path)
