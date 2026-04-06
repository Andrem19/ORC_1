"""Tests for shared subprocess pipe draining helpers."""

from __future__ import annotations

import errno
from unittest.mock import patch

from app.subprocess_io import drain_pipe_text


class _FakePipe:
    def fileno(self) -> int:
        return 123


def test_drain_pipe_text_retries_eagain_until_data_and_eof() -> None:
    pipe = _FakePipe()
    reads = [
        BlockingIOError(errno.EAGAIN, "try again"),
        b"tail",
        b"",
    ]

    def _fake_read(_fd: int, _size: int) -> bytes:
        value = reads.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    with patch("app.subprocess_io.os.read", side_effect=_fake_read), patch("app.subprocess_io.time.sleep"):
        drained = drain_pipe_text(pipe, drain_timeout_seconds=0.1, retry_sleep_seconds=0.0)

    assert drained == "tail"


def test_drain_pipe_text_stops_on_non_retryable_oserror() -> None:
    pipe = _FakePipe()

    def _fake_read(_fd: int, _size: int) -> bytes:
        raise OSError(errno.EBADF, "bad fd")

    with patch("app.subprocess_io.os.read", side_effect=_fake_read):
        drained = drain_pipe_text(pipe, drain_timeout_seconds=0.1, retry_sleep_seconds=0.0)

    assert drained == ""
