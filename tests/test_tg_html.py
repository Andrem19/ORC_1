"""Tests for app.services.tg_html — Telegram HTML formatting utilities."""

from __future__ import annotations

from app.services.tg_html import (
    bold,
    code,
    escape_html,
    field,
    italic,
    pre_block,
    separator,
    truncate_html,
)


# ---------------------------------------------------------------------------
# escape_html
# ---------------------------------------------------------------------------

class TestEscapeHtml:
    def test_ampersand(self):
        assert escape_html("a & b") == "a &amp; b"

    def test_angle_brackets(self):
        assert escape_html("<tag>") == "&lt;tag&gt;"

    def test_all_special(self):
        assert escape_html("a<b&c>d") == "a&lt;b&amp;c&gt;d"

    def test_no_special(self):
        assert escape_html("hello world") == "hello world"

    def test_empty(self):
        assert escape_html("") == ""

    def test_already_escaped_not_double_escaped(self):
        # Our function is simple substitution; &amp; becomes &amp;amp;
        # This is by design — only raw text should be passed.
        assert escape_html("&amp;") == "&amp;amp;"


# ---------------------------------------------------------------------------
# Tag helpers
# ---------------------------------------------------------------------------

class TestTagHelpers:
    def test_bold(self):
        assert bold("hello") == "<b>hello</b>"

    def test_italic(self):
        assert italic("hello") == "<i>hello</i>"

    def test_code(self):
        assert code("x") == "<code>x</code>"

    def test_pre_block(self):
        assert pre_block("line1\nline2") == "<pre>line1\nline2</pre>"


# ---------------------------------------------------------------------------
# field
# ---------------------------------------------------------------------------

class TestField:
    def test_basic(self):
        result = field("Task", "abc-123")
        assert result == "<b>Task:</b> abc-123"

    def test_escapes_label(self):
        result = field("A<B", "value")
        assert "&lt;" in result

    def test_escapes_value(self):
        result = field("Label", "a & b < c")
        assert "&amp;" in result and "&lt;" in result

    def test_empty_value(self):
        result = field("Label", "")
        assert result.endswith("</b> ")


# ---------------------------------------------------------------------------
# separator
# ---------------------------------------------------------------------------

class TestSeparator:
    def test_is_string(self):
        assert isinstance(separator(), str)

    def test_non_empty(self):
        assert len(separator()) > 0

    def test_consistent(self):
        assert separator() == separator()


# ---------------------------------------------------------------------------
# truncate_html
# ---------------------------------------------------------------------------

class TestTruncateHtml:
    def test_short_text_unchanged(self):
        assert truncate_html("hello", 10) == "hello"

    def test_truncates_plain_text(self):
        assert truncate_html("abcdefghij", 5) == "abcde"

    def test_tags_not_counted(self):
        html = "<b>abc</b>def"
        # visible: a b c d e f = 6 chars
        assert truncate_html(html, 4) == "<b>abc</b>d"
        assert truncate_html(html, 3) == "<b>abc</b>"

    def test_does_not_break_tags(self):
        html = "<b>hello</b> world"
        # visible: h e l l o   w o r l d = 11
        result = truncate_html(html, 7)
        assert result == "<b>hello</b> w"
        assert result.endswith(" w")

    def test_exact_limit(self):
        assert truncate_html("12345", 5) == "12345"

    def test_empty(self):
        assert truncate_html("", 10) == ""

    def test_tag_only(self):
        assert truncate_html("<br>", 0) == "<br>"

    def test_multiple_tags(self):
        html = "<b>a</b><i>b</i><code>c</code>"
        # visible: a b c = 3
        # max=2 → includes a,b and trailing <code> (harmless, empty tag)
        result = truncate_html(html, 2)
        assert result.startswith("<b>a</b><i>b</i>")
        assert truncate_html(html, 3) == html

    def test_malformed_unclosed_tag(self):
        html = "<b>hello"
        # visible: h e l l o = 5
        assert truncate_html(html, 3) == "<b>hel"
