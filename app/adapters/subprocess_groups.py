"""
Helpers for process-group aware adapter subprocess termination on POSIX.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any

from app.adapters.base import ProcessHandle


def configure_popen_kwargs() -> dict[str, Any]:
    return {"start_new_session": True}


def terminate_process_handle(handle: ProcessHandle, *, force: bool = False) -> None:
    process = handle.process
    if process is None:
        return
    if process.poll() is not None:
        return
    pgid = int(handle.metadata.get("pgid", 0) or 0)
    try:
        if pgid > 0:
            os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
        else:
            if force:
                process.kill()
            else:
                process.terminate()
    except ProcessLookupError:
        return
    except OSError:
        if force:
            process.kill()
        else:
            process.terminate()
    try:
        process.wait(timeout=0.5 if force else 1.0)
    except subprocess.TimeoutExpired:
        if not force:
            terminate_process_handle(handle, force=True)
