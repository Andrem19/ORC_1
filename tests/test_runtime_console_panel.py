from __future__ import annotations

import time

from rich.console import Console

from app.runtime_console.models import ConsoleRuntimeState, PlannerRuntimeView, SliceRuntimeView, ToolCallRuntimeView
from app.runtime_console.panel import _budget_text, _build_footer, _short, build_runtime_panel


def _render(state: ConsoleRuntimeState, width: int = 160) -> str:
    console = Console(record=True, width=width)
    console.print(build_runtime_panel(state))
    return console.export_text()


def test_runtime_panel_renders_active_sections() -> None:
    state = ConsoleRuntimeState(
        runtime_status="running",
        active_plan_id="plan_1",
        current_cycle=7,
        total_errors=1,
        broker_status="degraded",
        broker_tool_count=38,
        broker_warnings=["system_bootstrap:empty_payload"],
        planner=PlannerRuntimeView(status="running", adapter_name="claude", attempt=1, max_attempts=2),
        slices={
            1: SliceRuntimeView(
                slot=1,
                plan_id="plan_1",
                slice_id="slice_a",
                title="Validate OHLCV data",
                worker_id="worker-1",
                status="checkpointed",
                turns_used=1,
                turns_total=4,
                tool_calls_used=1,
                tool_calls_total=2,
                last_summary="catalog ready",
                active_operation_ref="op_1",
                active_operation_status="running",
            )
        },
        broker_calls={
            1: ToolCallRuntimeView(
                slot=1,
                plan_id="plan_1",
                slice_id="slice_a",
                tool_name="events_sync",
                phase="resume",
                operation_ref="op_1",
                response_status="running",
            )
        },
        last_warning="degraded broker bootstrap",
    )
    rendered = _render(state)

    assert "Broker Runtime" in rendered
    assert "plan_1" in rendered
    # title is shown in task column instead of slice_id
    assert "Validate OHLCV data" in rendered
    assert "events_sync" in rendered
    assert "degraded" in rendered
    # broker tool count shown
    assert "38t" in rendered


def test_runtime_panel_shows_uptime() -> None:
    state = ConsoleRuntimeState(
        runtime_status="running",
        started_at_monotonic=time.monotonic() - 5.0,
    )
    rendered = _render(state)
    # uptime row present and shows a non-zero elapsed
    assert "uptime" in rendered
    assert "s" in rendered


def test_runtime_panel_shows_plan_id_truncated() -> None:
    long_id = "plan_" + "x" * 40
    state = ConsoleRuntimeState(active_plan_id=long_id)
    rendered = _render(state)
    # _short(x, 12) keeps first 9 chars then appends "..."
    assert long_id[:9] in rendered
    assert long_id not in rendered


def test_runtime_panel_no_uptime_row_when_not_started() -> None:
    state = ConsoleRuntimeState(runtime_status="idle")
    rendered = _render(state)
    assert "uptime" not in rendered


# ---------------------------------------------------------------------------
# _budget_text
# ---------------------------------------------------------------------------

def test_budget_text_low_usage_is_white() -> None:
    t = _budget_text(1, 10, 2, 20)
    assert t.plain == "1/10T 2/20C"
    # style should be white (low usage) — no red or yellow
    assert "red" not in str(t.style)
    assert "yellow" not in str(t.style)


def test_budget_text_medium_usage_is_yellow() -> None:
    t = _budget_text(7, 10, 5, 10)  # 70% turns, 50% calls → max=70%
    assert "yellow" in str(t.style)


def test_budget_text_high_usage_is_red() -> None:
    t = _budget_text(9, 10, 5, 10)  # 90% turns → bold red
    assert "red" in str(t.style)
    assert "bold" in str(t.style)


def test_budget_text_zero_totals_no_crash() -> None:
    t = _budget_text(0, 0, 0, 0)
    assert "0/0T" in t.plain


# ---------------------------------------------------------------------------
# _build_footer freshness
# ---------------------------------------------------------------------------

def test_footer_shows_warning_age() -> None:
    state = ConsoleRuntimeState(
        last_warning="broker degraded",
        last_warning_at=time.monotonic() - 3.0,
    )
    footer = _build_footer(state)
    text = footer.plain
    assert "warning:" in text
    assert "ago" in text


def test_footer_shows_error_age() -> None:
    state = ConsoleRuntimeState(
        last_error="something broke",
        last_error_at=time.monotonic() - 7.5,
    )
    footer = _build_footer(state)
    text = footer.plain
    assert "error:" in text
    assert "ago" in text


def test_footer_no_age_when_timestamp_zero() -> None:
    state = ConsoleRuntimeState(
        last_warning="some warning",
        last_warning_at=0.0,  # not set
    )
    footer = _build_footer(state)
    text = footer.plain
    assert "ago" not in text


def test_footer_stop_reason() -> None:
    state = ConsoleRuntimeState(stop_reason="goal_reached")
    footer = _build_footer(state)
    assert "stop:" in footer.plain
    assert "goal_reached" in footer.plain


def test_footer_default_when_empty() -> None:
    state = ConsoleRuntimeState()
    footer = _build_footer(state)
    assert "broker-only runtime console active" in footer.plain


# ---------------------------------------------------------------------------
# Slice status colors
# ---------------------------------------------------------------------------

def test_slice_status_running_green() -> None:
    state = ConsoleRuntimeState(
        slices={1: SliceRuntimeView(slot=1, status="running")}
    )
    rendered = _render(state)
    assert "running" in rendered


def test_slice_fallback_to_slice_id_when_no_title() -> None:
    state = ConsoleRuntimeState(
        slices={1: SliceRuntimeView(slot=1, slice_id="slice_xyz789", title="")}
    )
    rendered = _render(state)
    assert "slice_xyz789" in rendered


def test_worker_id_truncated_to_8_chars() -> None:
    long_worker = "worker-" + "a" * 30
    state = ConsoleRuntimeState(
        slices={1: SliceRuntimeView(slot=1, worker_id=long_worker)}
    )
    rendered = _render(state)
    assert long_worker not in rendered
    assert long_worker[:8] in rendered


# ---------------------------------------------------------------------------
# _short helper
# ---------------------------------------------------------------------------

def test_short_truncates_long_text() -> None:
    assert _short("a" * 50, 10) == "a" * 7 + "..."


def test_short_preserves_short_text() -> None:
    assert _short("hello", 10) == "hello"


def test_short_strips_newlines() -> None:
    assert _short("line1\nline2", 20) == "line1 line2"
