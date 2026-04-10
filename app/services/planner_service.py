"""Thin planner adapter holder for the direct runtime."""

from __future__ import annotations
from app.adapters.base import BaseAdapter


class PlannerService:
    """Container for the planner adapter used by the direct runtime."""

    def __init__(
        self,
        adapter: BaseAdapter,
        timeout: int = 180,
        operator_directives: str = "",
    ) -> None:
        self.adapter = adapter
        self.timeout = timeout
        self.operator_directives = operator_directives
