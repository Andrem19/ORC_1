"""Tests for app.services.tg_message_builder — structured message builder."""

from __future__ import annotations

from app.services.tg_message_builder import (
    Section,
    TelegramMessageBuilder,
    apply_translated_text,
    render_html,
    render_plain,
)


# ---------------------------------------------------------------------------
# Builder basics
# ---------------------------------------------------------------------------

class TestBuilder:
    def test_empty(self):
        assert TelegramMessageBuilder().build() == []

    def test_header(self):
        sections = TelegramMessageBuilder().add_header("✅", "Hello").build()
        assert len(sections) == 1
        assert sections[0].kind == "header"
        assert sections[0].icon == "✅"
        assert sections[0].value == "Hello"

    def test_field(self):
        sections = (
            TelegramMessageBuilder()
            .add_field("Task", "abc-123")
            .build()
        )
        assert sections[0].kind == "field"
        assert sections[0].label == "Task"
        assert sections[0].value == "abc-123"

    def test_body(self):
        sections = TelegramMessageBuilder().add_body("some text").build()
        assert sections[0].kind == "body"
        assert sections[0].value == "some text"

    def test_code_block(self):
        sections = TelegramMessageBuilder().add_code_block("x=1").build()
        assert sections[0].kind == "code"

    def test_separator(self):
        sections = TelegramMessageBuilder().add_separator().build()
        assert sections[0].kind == "separator"

    def test_chaining(self):
        sections = (
            TelegramMessageBuilder()
            .add_header("✅", "Title")
            .add_separator()
            .add_field("Key", "Val")
            .add_body("desc")
            .build()
        )
        assert len(sections) == 4
        assert [s.kind for s in sections] == [
            "header", "separator", "field", "body"
        ]

    def test_build_returns_copy(self):
        builder = TelegramMessageBuilder()
        builder.add_header("!", "A")
        s1 = builder.build()
        builder.add_body("B")
        s2 = builder.build()
        assert len(s1) == 1
        assert len(s2) == 2


# ---------------------------------------------------------------------------
# render_plain
# ---------------------------------------------------------------------------

class TestRenderPlain:
    def test_header_with_icon(self):
        sections = [Section(kind="header", value="Title", icon="✅")]
        assert render_plain(sections) == "✅ Title"

    def test_header_without_icon(self):
        sections = [Section(kind="header", value="Title")]
        assert render_plain(sections) == "Title"

    def test_field(self):
        sections = [Section(kind="field", label="Task", value="abc")]
        # Label is not included in plain text — only value
        assert render_plain(sections) == "abc"

    def test_body(self):
        sections = [Section(kind="body", value="hello world")]
        assert render_plain(sections) == "hello world"

    def test_separator_skipped(self):
        sections = [Section(kind="header", value="A"), Section(kind="separator")]
        assert render_plain(sections) == "A"

    def test_full_message(self):
        sections = [
            Section(kind="header", value="Cycle #5", icon="✅"),
            Section(kind="separator"),
            Section(kind="field", label="Task", value="abc-123"),
            Section(kind="body", value="Work done"),
        ]
        plain = render_plain(sections)
        assert plain == "✅ Cycle #5\nabc-123\nWork done"


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------

class TestRenderHtml:
    def test_header_bold(self):
        sections = [Section(kind="header", value="Title", icon="✅")]
        html = render_html(sections)
        assert html.startswith("<b>")
        assert "✅" in html
        assert "Title" in html

    def test_field_rendered(self):
        sections = [Section(kind="field", label="Task", value="abc")]
        html = render_html(sections)
        assert "<b>Task:</b>" in html
        assert "abc" in html

    def test_body_escaped(self):
        sections = [Section(kind="body", value="a < b & c")]
        html = render_html(sections)
        assert "&lt;" in html
        assert "&amp;" in html

    def test_code_block_pre(self):
        sections = [Section(kind="code", value="x=1")]
        html = render_html(sections)
        assert "<pre>" in html

    def test_separator_present(self):
        sections = [Section(kind="separator")]
        html = render_html(sections)
        assert "\u2501" in html

    def test_header_value_escaped(self):
        sections = [Section(kind="header", value="A < B")]
        html = render_html(sections)
        assert "&lt;" in html


# ---------------------------------------------------------------------------
# apply_translated_text
# ---------------------------------------------------------------------------

class TestApplyTranslatedText:
    def _make_sections(self):
        return [
            Section(kind="header", value="Worker completed", icon="✅"),
            Section(kind="separator"),
            Section(kind="field", label="Task", value="abc-123"),
            Section(kind="body", value="Work is done"),
        ]

    def test_identity(self):
        sections = self._make_sections()
        plain = render_plain(sections)
        result = apply_translated_text(sections, plain)
        assert [s.value for s in result if s.kind != "separator"] == [
            "Worker completed", "abc-123", "Work is done"
        ]

    def test_translated_values(self):
        sections = self._make_sections()
        # Simulate translation — plain text only has values, no labels
        translated = "✅ Воркер завершён\nабц-123\nРабота выполнена"
        result = apply_translated_text(sections, translated)
        assert result[0].value == "Воркер завершён"
        assert result[2].value == "абц-123"
        assert result[3].value == "Работа выполнена"

    def test_separator_preserved(self):
        sections = self._make_sections()
        plain = render_plain(sections)
        result = apply_translated_text(sections, plain)
        assert result[1].kind == "separator"

    def test_icons_preserved(self):
        sections = self._make_sections()
        plain = render_plain(sections)
        result = apply_translated_text(sections, plain)
        assert result[0].icon == "✅"

    def test_labels_preserved(self):
        sections = self._make_sections()
        plain = render_plain(sections)
        result = apply_translated_text(sections, plain)
        assert result[2].label == "Task"

    def test_roundtrip_render_html(self):
        sections = self._make_sections()
        plain = render_plain(sections)
        translated = apply_translated_text(sections, plain)
        html = render_html(translated)
        assert "<b>" in html
        assert "<b>Task:</b>" in html
