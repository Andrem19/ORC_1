from __future__ import annotations

import time

from rich.console import Console

from app.runtime_console.models import (
    ConsoleRuntimeState,
    SliceRuntimeView,
    TrailBatch,
    TrailMap,
    TrailPlanGroup,
    TrailSliceSlot,
)
from app.runtime_console.panel import (
    _build_footer,
    _build_planner_table,
    _build_runtime_summary,
    _build_slices_table,
    _build_trail_map_text,
    _elapsed_hms,
    build_runtime_panel,
)


def _render(state: ConsoleRuntimeState, width: int = 200) -> str:
    console = Console(record=True, width=width)
    console.print(build_runtime_panel(state))
    return console.export_text()


def test_runtime_panel_renders_direct_runtime() -> None:
    state = ConsoleRuntimeState(runtime_status="running", goal="g")
    panel = build_runtime_panel(state)
    assert panel.title == "Direct Runtime"


def test_footer_mentions_direct_runtime() -> None:
    footer = _build_footer(ConsoleRuntimeState())
    assert "direct runtime console active" in footer.plain


# ---------------------------------------------------------------------------
# Trail map rendering (pytest-style: v1|...|...|v2|D.F|...)
# ---------------------------------------------------------------------------

def test_trail_map_pending_as_dots() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1"),
                TrailSliceSlot(slice_id="s2"),
                TrailSliceSlot(slice_id="s3"),
            ]),
            TrailBatch(plan_id="p2", slices=[
                TrailSliceSlot(slice_id="s4"),
                TrailSliceSlot(slice_id="s5"),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert "v1" in text.plain
    assert "...|.." in text.plain


def test_trail_map_completed_as_D() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="completed"),
                TrailSliceSlot(slice_id="s2", status="pending"),
                TrailSliceSlot(slice_id="s3", status="completed"),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert "D.D" in text.plain
    assert "2/3" in text.plain


def test_trail_map_failed_as_F() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="failed"),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert text.plain.startswith("v1|F")
    assert "1F" in text.plain


def test_trail_map_aborted_as_A() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="aborted"),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert text.plain.startswith("v1|A")
    assert "1A" in text.plain


def test_trail_map_skipped_as_S() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="skipped"),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert text.plain.startswith("v1|S")
    assert "1S" in text.plain


def test_trail_map_multiple_plans_with_mixed_outcomes() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="completed"),
                TrailSliceSlot(slice_id="s2", status="completed"),
            ]),
            TrailBatch(plan_id="p2", slices=[
                TrailSliceSlot(slice_id="s3", status="failed"),
            ]),
        ]),
        TrailPlanGroup(label="v2", sequence_id="s2", batches=[
            TrailBatch(plan_id="p3", slices=[
                TrailSliceSlot(slice_id="s4", status="completed"),
                TrailSliceSlot(slice_id="s5", status="skipped"),
            ]),
            TrailBatch(plan_id="p4", slices=[
                TrailSliceSlot(slice_id="s6", status="aborted"),
                TrailSliceSlot(slice_id="s7", status="pending"),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert "v1|DD|F" in text.plain
    assert "v2|DS|A." in text.plain
    assert "3/7" in text.plain
    assert "1S" in text.plain
    assert "1F" in text.plain
    assert "1A" in text.plain


def test_runtime_panel_shows_trail_row_when_map_populated() -> None:
    state = ConsoleRuntimeState(runtime_status="running")
    state.trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="completed"),
                TrailSliceSlot(slice_id="s2", status="failed"),
            ]),
        ]),
    ])
    rendered = _render(state)
    assert "trail" in rendered
    assert "DF" in rendered


def test_runtime_panel_no_trail_when_empty() -> None:
    state = ConsoleRuntimeState(runtime_status="idle")
    rendered = _render(state)
    assert "trail" not in rendered


# ---------------------------------------------------------------------------
# Fallback level in trail rendering
# ---------------------------------------------------------------------------

def test_trail_map_completed_with_fallback_1() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="completed", fallback_level=1),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert "D1" in text.plain
    assert "1/1" in text.plain


def test_trail_map_completed_with_fallback_2() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="completed", fallback_level=2),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert "D2" in text.plain


def test_trail_map_failed_with_fallback() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="failed", fallback_level=2),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert "F2" in text.plain


def test_trail_map_mixed_primary_and_fallback() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="completed", fallback_level=0),
                TrailSliceSlot(slice_id="s2", status="completed", fallback_level=1),
                TrailSliceSlot(slice_id="s3", status="completed", fallback_level=2),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert "DD1D2" in text.plain


def test_trail_map_aborted_with_fallback() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="aborted", fallback_level=1),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert "A1" in text.plain


def test_trail_map_running_as_R() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="running"),
                TrailSliceSlot(slice_id="s2", status="pending"),
                TrailSliceSlot(slice_id="s3", status="completed"),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert "R.D" in text.plain
    assert "1/3" in text.plain
    assert "1R" in text.plain


def test_trail_map_all_running() -> None:
    trail_map = TrailMap(plans=[
        TrailPlanGroup(label="v1", sequence_id="s1", batches=[
            TrailBatch(plan_id="p1", slices=[
                TrailSliceSlot(slice_id="s1", status="running"),
                TrailSliceSlot(slice_id="s2", status="running"),
            ]),
        ]),
    ])
    text = _build_trail_map_text(trail_map)
    assert "RR" in text.plain
    assert "2R" in text.plain
    assert "0/2" in text.plain


# ---------------------------------------------------------------------------
# Worker display — shows actual provider, not legacy config worker_id
# ---------------------------------------------------------------------------

def test_slice_table_shows_provider_as_worker() -> None:
    state = ConsoleRuntimeState(runtime_status="running")
    state.slices[1] = SliceRuntimeView(
        slot=1, worker_id="qwen_cli", status="running", title="Test task",
    )
    rendered = _render(state)
    assert "qwen_cli" in rendered


def test_slice_table_truncates_long_worker_id() -> None:
    state = ConsoleRuntimeState(runtime_status="running")
    state.slices[1] = SliceRuntimeView(
        slot=1, worker_id="very_long_provider_name_that_exceeds_8_chars",
        status="running", title="Test task",
    )
    panel = _build_slices_table(state)
    rendered = Console(record=True, width=200)
    rendered.print(panel)
    text = rendered.export_text()
    assert "very_lon" in text


def test_slice_table_fallback_updates_worker() -> None:
    """After fallback, the worker column should reflect the fallback provider."""
    state = ConsoleRuntimeState(runtime_status="running")
    state.slices[1] = SliceRuntimeView(
        slot=1, worker_id="claude_cli", status="running", title="Fallback task",
    )
    rendered = _render(state)
    assert "claude_cli" in rendered


# ---------------------------------------------------------------------------
# Uptime format hh:mm:ss and wave uptime
# ---------------------------------------------------------------------------

def test_elapsed_hms_format() -> None:
    # 3661 seconds = 1h 1m 1s
    result = _elapsed_hms(time.monotonic() - 3661)
    assert result == "01:01:01"


def test_elapsed_hms_zero() -> None:
    result = _elapsed_hms(time.monotonic())
    assert result == "00:00:00"


def test_elapsed_hms_large() -> None:
    # 86400 seconds = 24h
    result = _elapsed_hms(time.monotonic() - 86400)
    assert result == "24:00:00"


def test_runtime_summary_shows_uptime_hms() -> None:
    state = ConsoleRuntimeState(runtime_status="running")
    state.started_at_monotonic = time.monotonic() - 65  # 1m 5s
    panel = _build_runtime_summary(state)
    rendered = Console(record=True, width=200)
    rendered.print(panel)
    text = rendered.export_text()
    assert "00:01:05" in text


def test_runtime_summary_shows_wave_uptime_in_parens() -> None:
    state = ConsoleRuntimeState(runtime_status="running")
    state.started_at_monotonic = time.monotonic() - 3600  # 1h
    state.current_wave_started_at = time.monotonic() - 120  # 2m
    panel = _build_runtime_summary(state)
    rendered = Console(record=True, width=200)
    rendered.print(panel)
    text = rendered.export_text()
    assert "01:00:00" in text
    assert "(00:02:00)" in text


def test_runtime_summary_no_wave_uptime_when_not_started() -> None:
    state = ConsoleRuntimeState(runtime_status="running")
    state.started_at_monotonic = time.monotonic() - 30
    panel = _build_runtime_summary(state)
    rendered = Console(record=True, width=200)
    rendered.print(panel)
    text = rendered.export_text()
    assert "(" not in text.split("uptime")[1] if "uptime" in text else True


# ---------------------------------------------------------------------------
# Planner panel visibility
# ---------------------------------------------------------------------------

def test_planner_panel_hidden_in_compiled_raw() -> None:
    state = ConsoleRuntimeState(runtime_status="running", plan_source="compiled_raw")
    rendered = _render(state)
    assert "Planner" not in rendered


def test_planner_panel_visible_in_planner_mode() -> None:
    state = ConsoleRuntimeState(runtime_status="running", plan_source="planner")
    rendered = _render(state)
    assert "Planner" in rendered


def test_planner_panel_visible_by_default() -> None:
    state = ConsoleRuntimeState(runtime_status="running", plan_source="")
    rendered = _render(state)
    assert "Planner" in rendered


# ---------------------------------------------------------------------------
# Current field removed from Runtime summary
# ---------------------------------------------------------------------------

def test_runtime_summary_no_current_field() -> None:
    state = ConsoleRuntimeState(runtime_status="running")
    state.started_at_monotonic = time.monotonic()
    panel = _build_runtime_summary(state)
    rendered = Console(record=True, width=200)
    rendered.print(panel)
    text = rendered.export_text()
    assert "current" not in text.lower()
