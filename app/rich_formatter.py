"""
Rich formatter — maps orchestrator events, loggers, and levels to styled output.

Produces rich.text.Text objects with colors, bold, and italic based on:
- OrchestratorEvent enum values extracted from [bracket_tags] in messages
- Logger name prefixes (visual grouping)
- Log level (color weight)
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from rich.style import Style
from rich.text import Text

from app.models import OrchestratorEvent

# ---------------------------------------------------------------------------
# Event → style + display tag
# ---------------------------------------------------------------------------

EVENT_STYLES: dict[OrchestratorEvent, tuple[Style, str]] = {
    OrchestratorEvent.STARTED:         (Style(color="green", bold=True),          "STARTED"),
    OrchestratorEvent.CONFIG_LOADED:   (Style(color="cyan", dim=True),            "CONFIG"),
    OrchestratorEvent.STATE_RESTORED:  (Style(color="cyan"),                      "RESTORED"),
    OrchestratorEvent.STATE_SAVED:     (Style(dim=True),                          "SAVED"),
    OrchestratorEvent.PLANNER_CALLED:  (Style(color="magenta", bold=True),        "PLANNER"),
    OrchestratorEvent.PLANNER_RESULT:  (Style(color="yellow", bold=True),         "RESULT"),
    OrchestratorEvent.PLANNER_ERROR:   (Style(color="red", bold=True),            "PLAN_ERR"),
    OrchestratorEvent.WORKER_LAUNCHED: (Style(color="blue", bold=True),           "LAUNCHED"),
    OrchestratorEvent.WORKER_COMPLETED:(Style(color="green", bold=True),          "COMPLETED"),
    OrchestratorEvent.WORKER_FAILED:   (Style(color="red"),                       "FAILED"),
    OrchestratorEvent.WORKER_TIMED_OUT:(Style(color="red"),                       "TIMEOUT"),
    OrchestratorEvent.WORKER_STOPPED:  (Style(color="yellow"),                    "STOPPED"),
    OrchestratorEvent.NEW_RESULT:      (Style(color="cyan"),                      "RESULT"),
    OrchestratorEvent.NO_CHANGE:       (Style(dim=True),                          "IDLE"),
    OrchestratorEvent.SLEEPING:        (Style(color="blue", dim=True),            "SLEEP"),
    OrchestratorEvent.FINISHED:        (Style(color="green", bold=True),          "FINISHED"),
    OrchestratorEvent.ERROR:           (Style(color="red", bold=True),            "ERROR"),
    OrchestratorEvent.RESTART_RECOVERY:(Style(color="yellow"),                    "RECOVERY"),
}

# Reverse lookup: bracket tag string → OrchestratorEvent
_TAG_TO_EVENT: dict[str, OrchestratorEvent] = {e.value: e for e in OrchestratorEvent}

# ---------------------------------------------------------------------------
# Logger name → display tag + style
# ---------------------------------------------------------------------------

_LOGGER_GROUP: list[tuple[str, str, Style]] = [
    # (prefix_match, display_tag, style)  — first match wins
    ("orchestrator.plan",                "PLAN",     Style(color="magenta")),
    ("orchestrator.planner_service",     "PLAN_SVC", Style(color="magenta")),
    ("orchestrator.adapter.claude",      "CLAUDE",   Style(color="yellow")),
    ("orchestrator.adapter.qwen",        "QWEN",     Style(color="blue")),
    ("orchestrator.adapter.lmstudio",    "LMSTUD",   Style(color="blue")),
    ("orchestrator.adapter.fake_worker", "FAKE_W",   Style(color="blue")),
    ("orchestrator.adapter.fake_planner","FAKE_P",   Style(color="blue")),
    ("orchestrator.worker_service",      "WORKER",   Style(color="cyan")),
    ("orchestrator.task_supervisor",     "SUPER",    Style(color="cyan")),
    ("orchestrator.scheduler",           "SCHED",    Style(color="blue", dim=True)),
    ("orchestrator.notifications",       "TELEGRAM", Style(color="green")),
    ("orchestrator.translation",         "TRANSL",   Style(dim=True)),
    ("orchestrator.state",               "STATE",    Style(dim=True)),
    ("orchestrator.plan_store",          "PLAN_IO",  Style(color="magenta", dim=True)),
    ("orchestrator.mcp_problem_scanner", "MCP_SCAN", Style(color="red")),
    ("orchestrator.mcp_problem_store",   "MCP_IO",   Style(color="red", dim=True)),
    ("orchestrator.research_context",    "RESEARCH", Style(color="cyan", dim=True)),
    ("orchestrator.pid_lock",            "PID",      Style(dim=True)),
    ("orchestrator.reset",               "RESET",    Style(color="yellow", dim=True)),
    ("orchestrator.memory_service",      "MEMORY",   Style(dim=True)),
]


def _get_logger_tag(name: str) -> tuple[str, Style]:
    """Return (display_tag, style) for a logger name."""
    for prefix, tag, style in _LOGGER_GROUP:
        if name == prefix or name.startswith(prefix + "."):
            return tag, style
    return "", Style()


# ---------------------------------------------------------------------------
# Level → style
# ---------------------------------------------------------------------------

LEVEL_STYLES: dict[int, Style] = {
    logging.DEBUG:    Style(dim=True),
    logging.INFO:     Style(),
    logging.WARNING:  Style(color="yellow", italic=True),
    logging.ERROR:    Style(color="red", bold=True),
    logging.CRITICAL: Style(color="white", bold=True, bgcolor="red"),
}

# ---------------------------------------------------------------------------
# Tag detection regex
# ---------------------------------------------------------------------------

_BRACKET_RE = re.compile(r"^\[([a-z_]+)\]\s*")


def _extract_event_tag(message: str) -> tuple[OrchestratorEvent | None, str]:
    """Extract an [event_tag] from the start of a message.

    Returns (matched_event_or_None, remaining_message).
    """
    m = _BRACKET_RE.match(message)
    if not m:
        return None, message
    tag = m.group(1)
    event = _TAG_TO_EVENT.get(tag)
    if event is None:
        return None, message
    return event, message[m.end():]


# ---------------------------------------------------------------------------
# RichFormatter
# ---------------------------------------------------------------------------

class RichFormatter:
    """Formats a log record into a styled rich.text.Text for the console."""

    def __init__(self, truncate_length: int = 300) -> None:
        self.truncate_length = truncate_length

    def format(self, record: logging.LogRecord) -> Text:
        """Produce a styled Text from a log record."""
        message = record.getMessage()
        logger_name = record.name

        # Strip the orchestrator prefix for cleaner display
        if logger_name.startswith("orchestrator."):
            short_name = logger_name[len("orchestrator."):]
        elif logger_name == "orchestrator":
            short_name = ""
        else:
            short_name = logger_name

        # Detect [event_tag] at the start of the message
        event, rest = _extract_event_tag(message)

        text = Text()

        # -- event tag (colored badge) --
        if event is not None and event in EVENT_STYLES:
            style, label = EVENT_STYLES[event]
            text.append(f"[{label}]", style)
            text.append(" ")
        else:
            # -- logger group tag (fallback badge) --
            tag, tag_style = _get_logger_tag(logger_name)
            if tag:
                text.append(f"[{tag}]", tag_style)
                text.append(" ")

        # -- message body --
        body = rest if event is not None else message

        # Truncate long messages on the console
        if len(body) > self.truncate_length:
            body = body[: self.truncate_length] + "..."

        # Apply level-based style as base
        level_style = LEVEL_STYLES.get(record.levelno, Style())

        # For event-tagged messages, combine event style context with level style
        # Event tag already carries the semantic meaning; body gets level style
        text.append(body, level_style)

        return text
