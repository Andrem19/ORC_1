"""
HTML formatting utilities for Telegram Bot API messages.

Telegram supports a subset of HTML tags:
  <b>bold</b>, <i>italic</i>, <u>underline</u>, <s>strikethrough</s>,
  <code>inline code</code>, <pre>preformatted</pre>,
  <a href="...">link</a>.

All user-provided text MUST be escaped via escape_html() before
being inserted into any HTML context.
"""

from __future__ import annotations

import re


def escape_html(text: str) -> str:
    """Escape ``<``, ``>``, ``&`` for Telegram HTML mode."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def bold(text: str) -> str:
    """Wrap text in ``<b>`` tags."""
    return f"<b>{text}</b>"


def italic(text: str) -> str:
    """Wrap text in ``<i>`` tags."""
    return f"<i>{text}</i>"


def code(text: str) -> str:
    """Wrap text in ``<code>`` tags (inline monospace)."""
    return f"<code>{text}</code>"


def pre_block(text: str) -> str:
    """Wrap text in ``<pre>`` tags (block monospace)."""
    return f"<pre>{text}</pre>"


def field(label: str, value: str) -> str:
    """Render a labeled field: ``<b>Label:</b> value``.

    Both *label* and *value* are escaped automatically.
    """
    return f"{bold(escape_html(label) + ':')} {escape_html(value)}"


# Visual separator using Telegram-supported Unicode characters.
_SEPARATOR = "\u2501" * 18


def separator() -> str:
    """Return a visual separator line (horizontal bar)."""
    return _SEPARATOR


# Regex to detect HTML tags so we can skip them when counting visible chars.
_TAG_RE = re.compile(r"<[^>]*>")


def truncate_html(html: str, max_visible: int = 4096) -> str:
    """Truncate HTML to *max_visible* visible (non-tag) characters.

    Never cuts inside a tag.  If the limit falls mid-tag the entire
    tag (and everything after it) is dropped.
    """
    visible = 0
    i = 0
    length = len(html)

    while i < length:
        if html[i] == "<":
            # Skip the whole tag.
            close = html.find(">", i)
            if close == -1:
                # Malformed — treat rest as text.
                remaining = length - i
                if visible + remaining <= max_visible:
                    return html
                return html[: i + max_visible - visible]
            i = close + 1
            continue

        visible += 1
        if visible > max_visible:
            return html[:i]
        i += 1

    return html
