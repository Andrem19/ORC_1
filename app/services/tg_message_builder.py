"""
Structured message builder for Telegram notifications.

Produces a list of ``Section`` objects that can be rendered to either
plain text (for translation) or HTML (for Telegram Bot API).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.tg_html import (
    bold,
    code,
    escape_html,
    field as html_field,
    pre_block,
    separator as html_separator,
)


@dataclass
class Section:
    """One visual section of a notification message."""

    kind: str  # "header", "field", "body", "code", "separator"
    label: str = ""
    value: str = ""
    icon: str = ""


class TelegramMessageBuilder:
    """Fluent builder for structured Telegram notification messages."""

    def __init__(self) -> None:
        self._sections: list[Section] = []

    def add_header(self, icon: str, text: str) -> "TelegramMessageBuilder":
        self._sections.append(Section(kind="header", value=text, icon=icon))
        return self

    def add_field(self, label: str, value: str) -> "TelegramMessageBuilder":
        self._sections.append(Section(kind="field", label=label, value=value))
        return self

    def add_body(self, text: str) -> "TelegramMessageBuilder":
        self._sections.append(Section(kind="body", value=text))
        return self

    def add_code_block(self, code_text: str) -> "TelegramMessageBuilder":
        self._sections.append(Section(kind="code", value=code_text))
        return self

    def add_separator(self) -> "TelegramMessageBuilder":
        self._sections.append(Section(kind="separator"))
        return self

    def build(self) -> list[Section]:
        """Return a copy of the accumulated sections."""
        return list(self._sections)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_plain(sections: list[Section]) -> str:
    """Render sections to plain text (for translation input).

    Labels are NOT included — they are kept in English and added only
    during HTML rendering.  This means each non-separator section
    produces exactly one line containing only its value.
    """
    lines: list[str] = []
    for sec in sections:
        if sec.kind == "header":
            lines.append(f"{sec.icon} {sec.value}" if sec.icon else sec.value)
        elif sec.kind == "field":
            lines.append(sec.value)
        elif sec.kind == "body":
            lines.append(sec.value)
        elif sec.kind == "code":
            lines.append(sec.value)
        elif sec.kind == "separator":
            pass  # skip separators in plain text
    return "\n".join(lines)


def render_html(sections: list[Section]) -> str:
    """Render sections to Telegram-compatible HTML."""
    parts: list[str] = []
    for sec in sections:
        if sec.kind == "header":
            header = f"{sec.icon} {sec.value}" if sec.icon else sec.value
            parts.append(bold(escape_html(header)))
        elif sec.kind == "field":
            parts.append(html_field(sec.label, sec.value))
        elif sec.kind == "body":
            parts.append(escape_html(sec.value))
        elif sec.kind == "code":
            parts.append(pre_block(escape_html(sec.value)))
        elif sec.kind == "separator":
            parts.append(html_separator())
    return "\n".join(parts)


def apply_translated_text(
    sections: list[Section], translated_plain: str
) -> list[Section]:
    """Replace section values with translated text, matched line-by-line.

    ``translated_plain`` must have been produced by ``render_plain(sections)``
    and then passed through the translator.  The translator preserves ``\\n``
    boundaries so we can split and re-associate.
    """
    lines = translated_plain.split("\n")
    result: list[Section] = []
    line_idx = 0

    for sec in sections:
        new_sec = Section(kind=sec.kind, label=sec.label, icon=sec.icon)
        if sec.kind == "separator":
            result.append(new_sec)
            continue
        if line_idx < len(lines):
            line = lines[line_idx]
            if sec.kind == "header":
                # Strip icon prefix if translator kept it
                if sec.icon and line.startswith(sec.icon):
                    new_sec.value = line[len(sec.icon):].lstrip()
                else:
                    new_sec.value = line
            else:
                new_sec.value = line
            line_idx += 1
        else:
            new_sec.value = sec.value
        result.append(new_sec)

    return result
