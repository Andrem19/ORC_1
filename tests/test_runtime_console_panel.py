from __future__ import annotations

from rich.console import Console

from app.runtime_console.models import (
    ConsoleRuntimeState,
    TrailBatch,
    TrailMap,
    TrailPlanGroup,
    TrailSliceSlot,
)
from app.runtime_console.panel import _build_footer, _build_trail_map_text, build_runtime_panel


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
