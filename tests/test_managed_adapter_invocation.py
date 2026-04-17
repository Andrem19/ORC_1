from __future__ import annotations

import asyncio
import json
import sys

from app.adapters.base import AdapterResponse, BaseAdapter, ProcessHandle
from app.adapters.subprocess_groups import configure_popen_kwargs, terminate_process_handle
from app.services.direct_execution.managed_invocation import ManagedAdapterInvoker
from app.services.direct_execution.process_registry import ProcessRegistry


class _SleepyAdapter(BaseAdapter):
    def name(self) -> str:
        return "sleepy"

    def is_available(self) -> bool:
        return True

    def invoke(self, prompt: str, timeout: int = 120, **kwargs):
        raise NotImplementedError

    def start(self, prompt: str, **kwargs) -> ProcessHandle:
        import subprocess

        script = (
            "import sys,time; "
            "sys.stdout.write('started\\n'); sys.stdout.flush(); "
            f"time.sleep({float(prompt)}); "
            "sys.stdout.write('done\\n'); sys.stdout.flush()"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            **configure_popen_kwargs(),
        )
        return ProcessHandle(
            process=process,
            task_id=kwargs.get("task_id", ""),
            worker_id=kwargs.get("worker_id", ""),
            metadata={"pgid": process.pid},
        )

    def check(self, handle: ProcessHandle):
        from app.subprocess_io import drain_pipe_text, read_available_text

        proc = handle.process
        assert proc is not None
        fragment = read_available_text(proc.stdout)
        if fragment:
            handle.partial_output += fragment
        err = read_available_text(proc.stderr)
        if err:
            handle.partial_error_output += err
        finished = proc.poll() is not None
        if finished:
            tail = drain_pipe_text(proc.stdout)
            if tail:
                handle.partial_output += tail
            tail_err = drain_pipe_text(proc.stderr)
            if tail_err:
                handle.partial_error_output += tail_err
            proc.wait()
        return fragment, finished

    def terminate(self, handle: ProcessHandle, *, force: bool = False) -> None:
        terminate_process_handle(handle, force=force)



class _NoProgressAdapter(BaseAdapter):
    def name(self) -> str:
        return "no_progress"

    def is_available(self) -> bool:
        return True

    def invoke(self, prompt: str, timeout: int = 120, **kwargs):
        raise NotImplementedError

    def start(self, prompt: str, **kwargs) -> ProcessHandle:
        del prompt
        return ProcessHandle(
            process=None,
            task_id=kwargs.get("task_id", ""),
            worker_id=kwargs.get("worker_id", ""),
            metadata={"terminated": False},
        )

    def check(self, handle: ProcessHandle):
        if handle.metadata.get("terminated"):
            return "", True
        return "", False

    def terminate(self, handle: ProcessHandle, *, force: bool = False) -> None:
        del force
        handle.metadata["terminated"] = True


class _ProgressThenStallAdapter(BaseAdapter):
    def name(self) -> str:
        return "progress_then_stall"

    def is_available(self) -> bool:
        return True

    def invoke(self, prompt: str, timeout: int = 120, **kwargs):
        raise NotImplementedError

    def start(self, prompt: str, **kwargs) -> ProcessHandle:
        del prompt
        return ProcessHandle(
            process=None,
            task_id=kwargs.get("task_id", ""),
            worker_id=kwargs.get("worker_id", ""),
            metadata={"phase": "initial", "terminated": False},
        )

    def check(self, handle: ProcessHandle):
        if handle.metadata.get("terminated"):
            return "", True
        if handle.metadata.get("phase") == "initial":
            handle.metadata["phase"] = "stalled"
            handle.partial_output += "started\n"
            return "started\n", False
        return "", False

    def terminate(self, handle: ProcessHandle, *, force: bool = False) -> None:
        del force
        handle.metadata["terminated"] = True



class _StreamJsonNoToolProgressAdapter(BaseAdapter):
    def name(self) -> str:
        return "stream_json_no_tool_progress"

    def is_available(self) -> bool:
        return True

    def invoke(self, prompt: str, timeout: int = 120, **kwargs):
        raise NotImplementedError

    def start(self, prompt: str, **kwargs) -> ProcessHandle:
        del prompt
        return ProcessHandle(
            process=None,
            task_id=kwargs.get("task_id", ""),
            worker_id=kwargs.get("worker_id", ""),
            metadata={
                "phase": "initial",
                "terminated": False,
                "output_mode": "stream-json",
                "tool_call_count": 0,
            },
        )

    def check(self, handle: ProcessHandle):
        if handle.metadata.get("terminated"):
            return "", True
        handle.partial_output += "thinking\n"
        return "thinking\n", False

    def terminate(self, handle: ProcessHandle, *, force: bool = False) -> None:
        del force
        handle.metadata["terminated"] = True


class _ChunkedJsonAdapter(BaseAdapter):
    def __init__(self) -> None:
        self.fragments = [
            '{"type":"tool_call","tool":"features_catalog",',
            '"arguments":{"scope":"available"},"reason":"inspect"}',
        ]
        self.calls = 0

    def name(self) -> str:
        return "chunked_json"

    def is_available(self) -> bool:
        return True

    def invoke(self, prompt: str, timeout: int = 120, **kwargs):
        raise NotImplementedError

    def start(self, prompt: str, **kwargs) -> ProcessHandle:
        del prompt, kwargs
        self.calls += 1
        return ProcessHandle(process=None, task_id="", worker_id="", metadata={"fragments": list(self.fragments)})

    def check(self, handle: ProcessHandle):
        fragments = handle.metadata.get("fragments", [])
        if fragments:
            fragment = fragments.pop(0)
            handle.partial_output += fragment
            return fragment, False
        return "", False

    def terminate(self, handle: ProcessHandle, *, force: bool = False) -> None:
        del force
        handle.metadata["terminated"] = True


def test_managed_adapter_invoker_returns_success_response() -> None:
    invoker = ManagedAdapterInvoker(process_registry=ProcessRegistry(), poll_interval_seconds=0.02)
    response = asyncio.run(
        invoker.invoke_with_retries(
            adapter=_SleepyAdapter(),
            prompt="0.05",
            timeout_seconds=3,
            max_attempts=1,
            base_backoff_seconds=0.01,
        )
    )
    assert response.success is True
    assert "done" in response.raw_output


def test_process_registry_terminates_registered_adapter_process() -> None:
    registry = ProcessRegistry()
    adapter = _SleepyAdapter()
    handle = adapter.start("10")
    registry.register(adapter=adapter, handle=handle, owner="worker")

    assert registry.has_live_processes() is True
    registry.terminate_all(grace_seconds=0.05, force_after=0.1)
    assert registry.has_live_processes() is False


def test_managed_adapter_invoker_accepts_partial_response_early() -> None:
    invoker = ManagedAdapterInvoker(process_registry=ProcessRegistry(), poll_interval_seconds=0.01)
    adapter = _ChunkedJsonAdapter()

    async def _acceptor(**kwargs):
        partial = kwargs["partial_output"]
        if partial.endswith('"}'):
            from app.adapters.base import AdapterResponse

            return AdapterResponse(success=True, raw_output=partial, finish_reason="early_success")
        return None

    response = asyncio.run(
        invoker.invoke_with_retries(
            adapter=adapter,
            prompt="ignored",
            timeout_seconds=1,
            max_attempts=1,
            base_backoff_seconds=0.01,
            on_partial_response=_acceptor,
        )
    )

    assert response.success is True
    assert response.finish_reason == "early_success"
    assert '"tool":"features_catalog"' in response.raw_output


class _ThreadMetadataAdapter(BaseAdapter):
    def name(self) -> str:
        return "thread_metadata"

    def is_available(self) -> bool:
        return True

    def invoke(self, prompt: str, timeout: int = 120, **kwargs):
        raise NotImplementedError

    def start(self, prompt: str, **kwargs) -> ProcessHandle:
        del prompt, kwargs
        return ProcessHandle(
            process=None,
            task_id="task",
            worker_id="worker",
            metadata={
                "thread_done": True,
                "response": AdapterResponse(success=True, raw_output='{"type":"final_report","summary":"ok"}'),
                "thread": object(),
            },
        )

    def check(self, handle: ProcessHandle):
        response = handle.metadata["response"]
        handle.partial_output = response.raw_output
        return response.raw_output, True

    def terminate(self, handle: ProcessHandle, *, force: bool = False) -> None:
        del force
        handle.metadata["cancelled"] = True


def test_managed_adapter_invoker_returns_json_safe_metadata_for_thread_adapters() -> None:
    invoker = ManagedAdapterInvoker(process_registry=ProcessRegistry(), poll_interval_seconds=0.01)

    response = asyncio.run(
        invoker.invoke_with_retries(
            adapter=_ThreadMetadataAdapter(),
            prompt="ignored",
            timeout_seconds=1,
            max_attempts=1,
            base_backoff_seconds=0.01,
        )
    )

    assert response.success is True
    assert response.metadata["thread"] == "background_thread"
    assert response.metadata["response"]["success"] is True
    json.dumps(response.metadata)


def test_managed_adapter_invoker_stalls_before_first_output() -> None:
    invoker = ManagedAdapterInvoker(process_registry=ProcessRegistry(), poll_interval_seconds=0.01)

    response = asyncio.run(
        invoker.invoke_with_retries(
            adapter=_NoProgressAdapter(),
            prompt="1.0",
            timeout_seconds=5,
            first_action_timeout_seconds=0.05,
            stalled_action_timeout_seconds=0.05,
            max_attempts=1,
            base_backoff_seconds=0.01,
        )
    )

    assert response.success is False
    assert response.timed_out is True
    assert response.finish_reason == "stalled_before_first_output"
    assert "Stalled after" in response.error



def test_managed_adapter_invoker_stalls_before_first_action_on_stream_json_no_tool_calls() -> None:
    invoker = ManagedAdapterInvoker(process_registry=ProcessRegistry(), poll_interval_seconds=0.01)

    response = asyncio.run(
        invoker.invoke_with_retries(
            adapter=_StreamJsonNoToolProgressAdapter(),
            prompt="ignored",
            timeout_seconds=5,
            first_action_timeout_seconds=0.05,
            stalled_action_timeout_seconds=0.5,
            max_attempts=1,
            base_backoff_seconds=0.01,
        )
    )

    assert response.success is False
    assert response.timed_out is True
    assert response.finish_reason == "stalled_before_first_action"
    assert response.metadata["stall_before_first_action"] is True


def test_managed_adapter_invoker_stalls_between_outputs() -> None:
    invoker = ManagedAdapterInvoker(process_registry=ProcessRegistry(), poll_interval_seconds=0.01)

    response = asyncio.run(
        invoker.invoke_with_retries(
            adapter=_ProgressThenStallAdapter(),
            prompt="ignored",
            timeout_seconds=5,
            first_action_timeout_seconds=0.5,
            stalled_action_timeout_seconds=0.05,
            max_attempts=1,
            base_backoff_seconds=0.01,
        )
    )

    assert response.success is False
    assert response.timed_out is True
    assert response.finish_reason == "stalled_between_outputs"
    assert "started" in response.raw_output
