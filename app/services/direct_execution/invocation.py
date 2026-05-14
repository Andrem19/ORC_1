"""
Async helpers for planner/worker adapter invocation with bounded retries.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

from app.adapters.base import AdapterResponse, BaseAdapter

logger = logging.getLogger("orchestrator.adapter.runtime")


class AdapterInvocationError(RuntimeError):
    """Raised when an adapter cannot produce a usable response."""


class AdapterInvocationCancelled(AdapterInvocationError):
    """Raised when an in-flight adapter process is cancelled by runtime shutdown."""


async def invoke_adapter_with_retries(
    *,
    adapter: BaseAdapter,
    prompt: str,
    timeout_seconds: int,
    max_attempts: int,
    base_backoff_seconds: float,
    on_attempt_start: Any | None = None,
    on_attempt_retry: Any | None = None,
    on_attempt_finish: Any | None = None,
    on_partial_response: Any | None = None,
    **kwargs: Any,
) -> AdapterResponse:
    del on_partial_response
    last_response: AdapterResponse | None = None
    last_exception: Exception | None = None
    attempts = max(1, int(max_attempts or 1))
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    prompt_chars = len(prompt)
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        if callable(on_attempt_start):
            maybe = on_attempt_start(attempt=attempt, max_attempts=attempts)
            if asyncio.iscoroutine(maybe):
                await maybe
        logger.info(
            "adapter_invoke_started",
            extra={
                "event_kind": "adapter_invoke_started",
                "adapter_name": adapter.name(),
                "attempt": attempt,
                "max_attempts": attempts,
                "timeout_seconds": timeout_seconds,
                "prompt_chars": prompt_chars,
                "prompt_hash": prompt_hash,
            },
        )
        try:
            response = adapter.invoke(
                prompt,
                timeout=timeout_seconds,
                **kwargs,
            )
        except Exception as exc:
            last_exception = exc
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if attempt >= attempts or not _is_transient_exception(exc):
                logger.warning(
                    "adapter_invoke_exception",
                    extra={
                        "event_kind": "adapter_invoke_exception",
                        "adapter_name": adapter.name(),
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "elapsed_ms": elapsed_ms,
                        "error": str(exc),
                        "prompt_hash": prompt_hash,
                    },
                )
                if callable(on_attempt_finish):
                    maybe = on_attempt_finish(success=False, error=str(exc), attempt=attempt, max_attempts=attempts)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                raise AdapterInvocationError(str(exc)) from exc
            logger.warning(
                "adapter_invoke_retry_exception",
                extra={
                    "event_kind": "adapter_invoke_retry",
                    "adapter_name": adapter.name(),
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "elapsed_ms": elapsed_ms,
                    "error": str(exc),
                    "prompt_hash": prompt_hash,
                },
            )
            if callable(on_attempt_retry):
                maybe = on_attempt_retry(error=str(exc), attempt=attempt, max_attempts=attempts)
                if asyncio.iscoroutine(maybe):
                    await maybe
            await asyncio.sleep(_backoff_seconds(base_backoff_seconds, attempt))
            continue
        last_response = response
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "adapter_invoke_finished",
            extra={
                "event_kind": "adapter_invoke_finished",
                "adapter_name": adapter.name(),
                "attempt": attempt,
                "max_attempts": attempts,
                "elapsed_ms": elapsed_ms,
                "success": response.success,
                "timed_out": response.timed_out,
                "exit_code": response.exit_code,
                "error": response.error or "",
                "output_chars": len(response.raw_output or ""),
                "prompt_hash": prompt_hash,
            },
        )
        if response.success:
            if callable(on_attempt_finish):
                maybe = on_attempt_finish(success=True, error="", attempt=attempt, max_attempts=attempts)
                if asyncio.iscoroutine(maybe):
                    await maybe
            return response
        if attempt >= attempts or not _is_transient_response(response):
            if callable(on_attempt_finish):
                maybe = on_attempt_finish(success=False, error=response.error or "adapter_failed", attempt=attempt, max_attempts=attempts)
                if asyncio.iscoroutine(maybe):
                    await maybe
            return response
        if callable(on_attempt_retry):
            maybe = on_attempt_retry(error=response.error or "transient_response", attempt=attempt, max_attempts=attempts)
            if asyncio.iscoroutine(maybe):
                await maybe
        await asyncio.sleep(_backoff_seconds(base_backoff_seconds, attempt))
    if last_response is not None:
        return last_response
    assert last_exception is not None
    raise AdapterInvocationError(str(last_exception)) from last_exception


def _is_transient_response(response: AdapterResponse) -> bool:
    if response.timed_out:
        return True
    message = f"{response.error} {response.raw_output}".lower()
    return any(token in message for token in _TRANSIENT_TOKENS)


def _is_transient_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in _TRANSIENT_TOKENS)


def _backoff_seconds(base_backoff_seconds: float, attempt: int) -> float:
    base = max(0.05, float(base_backoff_seconds or 0.25))
    return min(base * (2 ** max(0, attempt - 1)), 2.0)


_TRANSIENT_TOKENS = (
    "timeout",
    "timed out",
    "connection",
    "transport",
    "temporarily",
    "temporary",
    "broken pipe",
    "subprocess",
)
