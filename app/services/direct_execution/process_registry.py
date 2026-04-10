"""
Thread-safe registry for runtime-owned external adapter subprocesses.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.adapters.base import BaseAdapter, ProcessHandle


@dataclass
class RegisteredProcess:
    token: str
    adapter: BaseAdapter
    handle: ProcessHandle
    owner: str
    plan_id: str = ""
    slice_id: str = ""
    cancelled: bool = False


class ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, RegisteredProcess] = {}
        self._counter = 0

    def register(
        self,
        *,
        adapter: BaseAdapter,
        handle: ProcessHandle,
        owner: str,
        plan_id: str = "",
        slice_id: str = "",
    ) -> str:
        with self._lock:
            self._counter += 1
            token = f"proc_{self._counter}"
            self._items[token] = RegisteredProcess(
                token=token,
                adapter=adapter,
                handle=handle,
                owner=owner,
                plan_id=plan_id,
                slice_id=slice_id,
            )
            return token

    def unregister(self, token: str) -> None:
        with self._lock:
            self._items.pop(token, None)

    def is_cancelled(self, token: str) -> bool:
        with self._lock:
            item = self._items.get(token)
            return bool(item and item.cancelled)

    def has_live_processes(self) -> bool:
        with self._lock:
            items = list(self._items.values())
        return any(_is_alive(item.handle) for item in items)

    def terminate_all(self, *, grace_seconds: float = 0.5, force_after: float = 1.5) -> int:
        with self._lock:
            items = list(self._items.values())
            for item in items:
                item.cancelled = True
        if not items:
            return 0
        for item in items:
            item.adapter.terminate(item.handle, force=False)
        self._wait_until_dead(items, deadline=time.monotonic() + max(0.0, grace_seconds))
        alive = [item for item in items if _is_alive(item.handle)]
        for item in alive:
            item.adapter.terminate(item.handle, force=True)
        self._wait_until_dead(alive, deadline=time.monotonic() + max(0.0, force_after))
        return len(items)

    @staticmethod
    def _wait_until_dead(items: list[RegisteredProcess], *, deadline: float) -> None:
        while time.monotonic() < deadline:
            if not any(_is_alive(item.handle) for item in items):
                return
            time.sleep(0.05)


def _is_alive(handle: ProcessHandle) -> bool:
    process = handle.process
    if process is None:
        return False
    return process.poll() is None
