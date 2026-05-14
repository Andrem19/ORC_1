from __future__ import annotations

import subprocess
import sys

from app.adapters.base import ProcessHandle
from app.adapters.subprocess_groups import configure_popen_kwargs, terminate_process_handle


def test_terminate_process_handle_kills_process_group() -> None:
    script = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
        **configure_popen_kwargs(),
    )
    handle = ProcessHandle(process=process, task_id="test", worker_id="test", metadata={"pgid": process.pid})
    terminate_process_handle(handle, force=False)
    assert process.poll() is not None
