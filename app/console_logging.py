"""
Console logging suppression helpers shared across Rich and plain handlers.
"""

from __future__ import annotations

import logging

from app.models import OrchestratorEvent
from app.rich_formatter import _extract_event_tag

_SUPPRESSED_EVENTS: set[OrchestratorEvent] = {
    OrchestratorEvent.STATE_SAVED,
}

_SUPPRESSED_LOGGER_PREFIXES: tuple[str, ...] = (
    "orchestrator.state",
    "orchestrator.direct.mcp",
)


def should_suppress_console_record(record: logging.LogRecord) -> bool:
    event_kind = str(getattr(record, "event_kind", "") or "")
    if event_kind in {
        "direct_tool_call",
        "adapter_invoke_started",
        "adapter_invoke_finished",
        "adapter_invoke_retry",
        "adapter_invoke_exception",
    }:
        return True
    event, _ = _extract_event_tag(record.getMessage())
    if event in _SUPPRESSED_EVENTS:
        return True
    for prefix in _SUPPRESSED_LOGGER_PREFIXES:
        if record.name == prefix or record.name.startswith(prefix + "."):
            return True
    return False


class ConsoleNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not should_suppress_console_record(record)
