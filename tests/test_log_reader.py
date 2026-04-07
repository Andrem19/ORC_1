"""Tests for log_reader utility."""

from __future__ import annotations

import textwrap
from pathlib import Path

from app.services.log_reader import read_last_n_lines


class TestReadLastNLines:
    def test_returns_tail_of_file(self, tmp_path: Path) -> None:
        log = tmp_path / "test.log"
        lines = [f"line {i:03d}" for i in range(300)]
        log.write_text("\n".join(lines))

        result = read_last_n_lines(log, n=200)
        assert len(result) == 200
        assert result[0] == "line 100"
        assert result[-1] == "line 299"

    def test_returns_all_when_fewer_than_n(self, tmp_path: Path) -> None:
        log = tmp_path / "test.log"
        log.write_text("line 1\nline 2\nline 3")

        result = read_last_n_lines(log, n=200)
        assert len(result) == 3
        assert result[0] == "line 1"

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        log = tmp_path / "nonexistent.log"
        result = read_last_n_lines(log, n=200)
        assert result == []

    def test_returns_empty_for_empty_file(self, tmp_path: Path) -> None:
        log = tmp_path / "empty.log"
        log.write_text("")

        result = read_last_n_lines(log, n=200)
        assert result == []

    def test_strips_trailing_newlines(self, tmp_path: Path) -> None:
        log = tmp_path / "test.log"
        log.write_text("line 1\nline 2\n")

        result = read_last_n_lines(log, n=10)
        assert result == ["line 1", "line 2"]

    def test_handles_string_path(self, tmp_path: Path) -> None:
        log = tmp_path / "test.log"
        log.write_text("hello\nworld")

        result = read_last_n_lines(str(log), n=10)
        assert result == ["hello", "world"]
