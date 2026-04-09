"""
Hybrid semantic extraction service for raw markdown plans.
"""

from __future__ import annotations

from app.adapters.base import BaseAdapter
from app.execution_parsing import StructuredOutputError
from app.raw_plan_models import RawPlanDocument, SemanticRawPlan
from app.raw_plan_parsing import parse_semantic_raw_plan_output
from app.raw_plan_prompts import build_raw_plan_semantic_prompt
from app.services.brokered_execution.invocation import AdapterInvocationError, invoke_adapter_with_retries


class RawPlanSemanticError(RuntimeError):
    """Raised when semantic extraction cannot produce a valid structured result."""


class RawPlanSemanticService:
    def __init__(
        self,
        *,
        adapter: BaseAdapter,
        timeout_seconds: int,
        retry_attempts: int,
        retry_backoff_seconds: float,
    ) -> None:
        self.adapter = adapter
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

    async def extract(self, document: RawPlanDocument) -> SemanticRawPlan:
        prompt = build_raw_plan_semantic_prompt(document)
        try:
            response = await invoke_adapter_with_retries(
                adapter=self.adapter,
                prompt=prompt,
                timeout_seconds=self.timeout_seconds,
                max_attempts=self.retry_attempts,
                base_backoff_seconds=self.retry_backoff_seconds,
            )
        except AdapterInvocationError as exc:
            raise RawPlanSemanticError(str(exc)) from exc
        if not response.success:
            raise RawPlanSemanticError(response.error or "raw_plan_semantic_invoke_failed")
        try:
            return parse_semantic_raw_plan_output(response.raw_output, document=document)
        except StructuredOutputError as exc:
            raise RawPlanSemanticError(str(exc)) from exc

