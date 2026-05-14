"""
Shared nonblocking subprocess pipe helpers.
"""

from __future__ import annotations

import errno
import os
import time
from typing import Any


_RETRY_ERRNOS = {errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR}


def read_available_text(pipe: Any) -> str:
    """Read one nonblocking chunk from pipe, returning decoded text."""
    if pipe is None:
        return ""
    try:
        chunk = os.read(pipe.fileno(), 65536)
    except BlockingIOError:
        return ""
    except OSError as exc:
        if exc.errno in _RETRY_ERRNOS:
            return ""
        return ""
    if not chunk:
        return ""
    return chunk.decode("utf-8", errors="replace")


def drain_pipe_text(
    pipe: Any,
    *,
    drain_timeout_seconds: float = 2.0,
    retry_sleep_seconds: float = 0.01,
) -> str:
    """Drain pipe until EOF with bounded retries for transient nonblocking reads."""
    if pipe is None:
        return ""

    fragments: list[str] = []
    deadline = time.monotonic() + max(0.0, drain_timeout_seconds)

    while True:
        try:
            chunk = os.read(pipe.fileno(), 65536)
        except BlockingIOError:
            if time.monotonic() >= deadline:
                break
            time.sleep(retry_sleep_seconds)
            continue
        except OSError as exc:
            if exc.errno in _RETRY_ERRNOS and time.monotonic() < deadline:
                time.sleep(retry_sleep_seconds)
                continue
            break

        if not chunk:
            break
        fragments.append(chunk.decode("utf-8", errors="replace"))

    return "".join(fragments)
