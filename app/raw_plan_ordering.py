"""
Stable natural ordering for raw-plan file names and compiled manifests.
"""

from __future__ import annotations

import re
from pathlib import Path

_NUMERIC_CHUNK_RE = re.compile(r"(\d+)")


def raw_plan_sort_key(value: str | Path) -> tuple[object, ...]:
    name = Path(value).name.lower()
    parts = _NUMERIC_CHUNK_RE.split(name)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)
