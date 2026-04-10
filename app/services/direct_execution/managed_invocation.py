"""
Managed subprocess invocation for planner/worker CLI adapters.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

from app.adapters.base import AdapterResponse, BaseAdapter, ProcessHandle
from app.services.direct_execution.invocation import (
    AdapterInvocationCancelled,
    AdapterInvocationError,
    _backoff_seconds,
    _is_transient_exception,
    _is_transient_response,
)
from app.services.direct_execution.process_registry import ProcessRegistry

logger = logging.getLogger("orchestrator.adapter.runtime")


class ManagedAdapterInvoker:
    def __init__(
        self,
        *,
        process_registry: ProcessRegistry,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        self.process_registry = process_registry
        self.poll_interval_seconds = max(0.02, float(poll_interval_seconds or 0.1))

    async def invoke_with_retries(
        self,
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
        process_owner: str = "",
        plan_id: str = "",
        slice_id: str = "",
        **kwargs: Any,
    ) -> AdapterResponse:
        if not _supports_managed_process(adapter):
            from app.services.direct_execution.invocation import invoke_adapter_with_retries

            return await invoke_adapter_with_retries(
                adapter=adapter,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                base_backoff_seconds=base_backoff_seconds,
                on_attempt_start=on_attempt_start,
                on_attempt_retry=on_attempt_retry,
                on_attempt_finish=on_attempt_finish,
                on_partial_response=on_partial_response,
                **kwargs,
            )

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
                response = await self._invoke_once(
                    adapter=adapter,
                    prompt=prompt,
                    timeout_seconds=timeout_seconds,
                    on_partial_response=on_partial_response,
                    process_owner=process_owner or adapter.name(),
                    plan_id=plan_id,
                    slice_id=slice_id,
                    **kwargs,
                )
            except AdapterInvocationCancelled:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                logger.warning(
                    "adapter_invoke_cancelled",
                    extra={
                        "event_kind": "adapter_invoke_cancelled",
                        "adapter_name": adapter.name(),
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "elapsed_ms": elapsed_ms,
                        "prompt_hash": prompt_hash,
                    },
                )
                if callable(on_attempt_finish):
                    maybe = on_attempt_finish(success=False, error="operation_cancelled", attempt=attempt, max_attempts=attempts)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                raise
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

    async def _invoke_once(
        self,
        *,
        adapter: BaseAdapter,
        prompt: str,
        timeout_seconds: int,
        on_partial_response: Any | None,
        process_owner: str,
        plan_id: str,
        slice_id: str,
        **kwargs: Any,
    ) -> AdapterResponse:
        handle = adapter.start(prompt, plan_id=plan_id, slice_id=slice_id, **kwargs)
        token = self.process_registry.register(
            adapter=adapter,
            handle=handle,
            owner=process_owner,
            plan_id=plan_id,
            slice_id=slice_id,
        )
        started = time.monotonic()
        try:
            while True:
                if self.process_registry.is_cancelled(token):
                    adapter.terminate(handle, force=False)
                _fragment, is_finished = adapter.check(handle)
                accepted = await _accept_partial_response(
                    on_partial_response=on_partial_response,
                    handle=handle,
                    adapter=adapter,
                    timeout_seconds=timeout_seconds,
                    started=started,
                )
                if accepted is not None:
                    if not is_finished:
                        adapter.terminate(handle, force=False)
                        await asyncio.sleep(min(0.2, self.poll_interval_seconds))
                        _fragment, is_finished = adapter.check(handle)
                        if not is_finished:
                            adapter.terminate(handle, force=True)
                            await asyncio.sleep(min(0.2, self.poll_interval_seconds))
                            adapter.check(handle)
                    return accepted
                if is_finished:
                    if self.process_registry.is_cancelled(token):
                        raise AdapterInvocationCancelled("shutdown_cancelled_inflight")
                    return _response_from_handle(handle)
                if time.monotonic() - started >= timeout_seconds:
                    adapter.terminate(handle, force=False)
                    await asyncio.sleep(min(0.2, self.poll_interval_seconds))
                    _fragment, is_finished = adapter.check(handle)
                    if not is_finished:
                        adapter.terminate(handle, force=True)
                        await asyncio.sleep(min(0.2, self.poll_interval_seconds))
                        _fragment, _is_finished = adapter.check(handle)
                    return AdapterResponse(
                        success=False,
                        raw_output=handle.partial_output,
                        exit_code=-1,
                        error=f"Timed out after {timeout_seconds}s",
                        timed_out=True,
                        duration_seconds=time.monotonic() - started,
                        finish_reason="timeout",
                        metadata={
                            "partial_output_chars": len(handle.partial_output or ""),
                            "partial_error_chars": len(handle.partial_error_output or ""),
                            "partial_output_present": bool(handle.partial_output),
                            "partial_error_present": bool(handle.partial_error_output),
                            **_json_safe_handle_metadata(handle),
                        },
                    )
                await asyncio.sleep(self.poll_interval_seconds)
        finally:
            self.process_registry.unregister(token)


def _supports_managed_process(adapter: BaseAdapter) -> bool:
    return (
        adapter.__class__.start is not BaseAdapter.start
        and adapter.__class__.check is not BaseAdapter.check
    )


def _response_from_handle(handle: ProcessHandle) -> AdapterResponse:
    process = handle.process
    exit_code = 0 if process is None or process.returncode is None else int(process.returncode)
    error = (handle.partial_error_output or "").strip()
    if exit_code != 0 and not error:
        error = f"Process exited with code {exit_code}"
    return AdapterResponse(
        success=exit_code == 0,
        raw_output=handle.partial_output,
        exit_code=exit_code,
        error=error[:500],
        duration_seconds=max(0.0, time.monotonic() - handle.started_at),
        finish_reason="completed" if exit_code == 0 else "process_error",
        metadata={
            "partial_output_chars": len(handle.partial_output or ""),
            "partial_error_chars": len(handle.partial_error_output or ""),
            **_json_safe_handle_metadata(handle),
        },
    )


def _json_safe_handle_metadata(handle: ProcessHandle) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in dict(handle.metadata or {}).items():
        if key == "thread":
            metadata[key] = "background_thread"
            continue
        if isinstance(value, AdapterResponse):
            metadata[key] = {
                "success": value.success,
                "exit_code": value.exit_code,
                "error": value.error,
                "timed_out": value.timed_out,
                "duration_seconds": value.duration_seconds,
                "finish_reason": value.finish_reason,
                "raw_output_chars": len(value.raw_output or ""),
            }
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
            continue
        if isinstance(value, (list, tuple)):
            metadata[key] = [item if isinstance(item, (str, int, float, bool)) or item is None else repr(item) for item in value]
            continue
        if isinstance(value, dict):
            metadata[key] = {
                str(sub_key): sub_value if isinstance(sub_value, (str, int, float, bool)) or sub_value is None else repr(sub_value)
                for sub_key, sub_value in value.items()
            }
            continue
        metadata[key] = repr(value)
    return metadata


async def _accept_partial_response(
    *,
    on_partial_response: Any | None,
    handle: ProcessHandle,
    adapter: BaseAdapter,
    timeout_seconds: int,
    started: float,
) -> AdapterResponse | None:
    if not callable(on_partial_response):
        return None
    maybe = on_partial_response(
        partial_output=handle.partial_output,
        partial_error_output=handle.partial_error_output,
        adapter_name=adapter.name(),
        timeout_seconds=timeout_seconds,
        elapsed_seconds=max(0.0, time.monotonic() - started),
    )
    if asyncio.iscoroutine(maybe):
        maybe = await maybe
    if not isinstance(maybe, AdapterResponse):
        return None
    metadata = dict(maybe.metadata or {})
    metadata.setdefault("partial_output_chars", len(handle.partial_output or ""))
    metadata.setdefault("partial_error_chars", len(handle.partial_error_output or ""))
    maybe.metadata = metadata
    if not maybe.finish_reason:
        maybe.finish_reason = "early_success"
    return maybe
