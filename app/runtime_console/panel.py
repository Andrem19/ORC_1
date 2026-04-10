"""
Rich renderables for the direct runtime console.
"""

from __future__ import annotations

import time

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.runtime_console.models import ConsoleRuntimeState, TrailMap, TrailSliceSlot


def build_runtime_panel(state: ConsoleRuntimeState):
    summary = _build_runtime_summary(state)
    slices = _build_slices_table(state)
    columns = [summary]
    if state.plan_source != "compiled_raw":
        columns.append(_build_planner_table(state))
    columns.append(slices)
    footer = _build_footer(state)
    return Panel(
        Columns(columns, expand=True, equal=False),
        title="Direct Runtime",
        subtitle=footer,
        border_style="cyan",
        padding=(0, 1),
    )


def _build_runtime_summary(state: ConsoleRuntimeState) -> Panel:
    table = Table.grid(padding=(0, 1))
    progress = state.sequence_progress
    table.add_row("status", _status_text(state.runtime_status, drain_mode=state.drain_mode))
    sequence_label = progress.raw_plan_label or (_short(state.active_plan_id, 12) if state.active_plan_id else "-")
    table.add_row("sequence", Text(sequence_label, style="bold cyan"))
    table.add_row("progress", Text(_progress_label(state), style="white"))
    table.add_row("errors", Text(str(state.total_errors), style="yellow" if state.total_errors else "green"))
    table.add_row("direct", Text(state.direct_status or state.runtime_status or "idle", style="green" if state.runtime_status == "running" else "white"))
    if state.started_at_monotonic:
        uptime_text = _elapsed_hms(state.started_at_monotonic)
        if state.current_wave_started_at:
            uptime_text += f" ({_elapsed_hms(state.current_wave_started_at)})"
        table.add_row("uptime", Text(uptime_text, style="cyan"))
    if state.trail_map.plans:
        table.add_row("trail", _build_trail_map_text(state.trail_map))
    return Panel(table, title="Runtime", border_style="blue")


def _build_planner_table(state: ConsoleRuntimeState) -> Panel:
    planner = state.planner
    table = Table.grid(padding=(0, 1))
    table.add_row("state", Text(planner.status or "idle", style="yellow" if planner.status == "running" else "white"))
    table.add_row("adapter", Text(planner.adapter_name or "-", style="magenta"))
    if planner.max_attempts:
        table.add_row("attempt", Text(f"{planner.attempt}/{planner.max_attempts}", style="white"))
    if planner.status == "running" and planner.started_at_monotonic:
        table.add_row("elapsed", Text(_elapsed_hms(planner.started_at_monotonic), style="cyan"))
    elif planner.last_error:
        table.add_row("last", Text(_short(planner.last_error, 44), style="red"))
    else:
        table.add_row("last", Text("-", style="dim"))
    return Panel(table, title="Planner", border_style="magenta")


def _build_slices_table(state: ConsoleRuntimeState) -> Panel:
    table = Table("slot", "task", "worker", "budget", "status", "summary", expand=True, box=None, padding=(0, 1))
    for slot in sorted(state.slices):
        item = state.slices[slot]
        task_text = (item.title or item.slice_id or "-").replace("\n", " ").strip()
        worker_label = (item.worker_id or "-")[:12]
        budget = _budget_text(item.turns_used, item.turns_total, item.tool_calls_used, item.tool_calls_total)
        status = _slice_status_text(item.status, item.active_operation_ref, item.active_operation_status)
        table.add_row(
            str(slot),
            task_text,
            worker_label,
            budget,
            status,
            _short(item.last_summary or "-", 60),
        )
    if not state.slices:
        table.add_row("-", "-", "-", "-", Text("idle", style="dim"), "-")
    return Panel(table, title="Active Slices", border_style="green")


def _build_footer(state: ConsoleRuntimeState) -> Text:
    parts: list[tuple[str, str]] = []
    if state.current_cycle:
        parts.append((f"loop #{state.current_cycle}", "dim"))
    if state.last_warning:
        age = f" ({_elapsed(state.last_warning_at)} ago)" if state.last_warning_at else ""
        parts.append((f"warning: {_short(state.last_warning, 80)}{age}", "yellow"))
    if state.last_error:
        age = f" ({_elapsed(state.last_error_at)} ago)" if state.last_error_at else ""
        parts.append((f"error: {_short(state.last_error, 80)}{age}", "red"))
    if state.stop_reason:
        parts.append((f"stop: {state.stop_reason}", "bold red"))
    if not parts:
        parts.append(("direct runtime console active", "dim"))
    text = Text()
    for index, (part, style) in enumerate(parts):
        if index:
            text.append(" | ", "dim")
        text.append(part, style)
    return text


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _status_text(status: str, *, drain_mode: bool) -> Text:
    if drain_mode:
        return Text("draining", style="bold yellow")
    if status == "running":
        return Text("running", style="bold green")
    if status == "finished":
        return Text("finished", style="bold cyan")
    if status == "error":
        return Text("error", style="bold red")
    return Text(status or "idle", style="white")


def _slice_status_text(status: str, operation_ref: str, operation_status: str) -> Text:
    _STATUS_STYLES: dict[str, str] = {
        "running": "green",
        "checkpointed": "blue",
        "completed": "cyan",
        "failed": "bold red",
        "aborted": "yellow",
    }
    style = _STATUS_STYLES.get(status, "dim")
    text = Text(status or "idle", style=style)
    if operation_ref:
        op_status = operation_status or "active"
        text.append(f":{op_status}", style="dim")
    return text


def _budget_text(turns_used: int, turns_total: int, calls_used: int, calls_total: int) -> Text:
    turn_pct = turns_used / max(1, turns_total)
    call_pct = calls_used / max(1, calls_total)
    ratio = max(turn_pct, call_pct)
    label = f"{turns_used}/{turns_total}T {calls_used}/{calls_total}C"
    if ratio >= 0.8:
        return Text(label, style="bold red")
    if ratio >= 0.6:
        return Text(label, style="yellow")
    return Text(label, style="white")


def _progress_label(state: ConsoleRuntimeState) -> str:
    progress = state.sequence_progress
    if not progress.current_batch_index:
        return "-"
    total_batches = "?" if progress.total_batches <= 0 else str(progress.total_batches)
    total_slices = "?" if progress.total_slices_in_batch <= 0 else str(progress.total_slices_in_batch)
    current_slice = progress.current_slice_index or 0
    return f"batch {progress.current_batch_index}/{total_batches} | slice {current_slice}/{total_slices}"


def _elapsed(started_at_monotonic: float) -> str:
    return f"{max(0.0, time.monotonic() - started_at_monotonic):.1f}s"


def _elapsed_hms(started_at_monotonic: float) -> str:
    total = max(0, int(time.monotonic() - started_at_monotonic))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _short(text: str, limit: int) -> str:
    cleaned = str(text or "").replace("\n", " ").strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit - 3]}..."


# ---------------------------------------------------------------------------
# Trail map rendering (pytest-style)
# ---------------------------------------------------------------------------

_TRAIL_CHARS: dict[str, tuple[str, str]] = {
    "running": ("R", "bold blue"),
    "completed": ("D", "green"),
    "failed": ("F", "bold red"),
    "aborted": ("A", "yellow"),
    "skipped": ("S", "cyan"),
}


def _trail_slot_char(slot: TrailSliceSlot) -> tuple[str, str]:
    """Return (character, style) for one trail slot."""
    char_style = _TRAIL_CHARS.get(slot.status)
    if char_style:
        char, style = char_style
        if slot.fallback_level > 0:
            char = f"{char}{slot.fallback_level}"
        return char, style
    return (".", "dim")


def _build_trail_map_text(trail_map: TrailMap) -> Text:
    """Render full trail map as pytest-style: v1|R.R|DD1|...|

    R  (green)   = running
    D  (green)   = completed (primary)
    D1 (green)   = completed via fallback_1
    D2 (green)   = completed via fallback_2
    F  (red)     = failed
    A  (yellow)  = aborted
    S  (cyan)    = skipped
    .  (dim)     = pending
    """
    text = Text()
    stats: dict[str, int] = {}
    total = 0

    for pg_idx, pg in enumerate(trail_map.plans):
        if pg_idx > 0:
            text.append(" ", style="dim")
        text.append(pg.label, style="bold cyan")
        for batch in pg.batches:
            text.append("|", style="dim")
            for slot in batch.slices:
                total += 1
                char, style = _trail_slot_char(slot)
                text.append(char, style=style)
                if slot.status != "pending":
                    stats[slot.status] = stats.get(slot.status, 0) + 1

    # Summary counters
    if total:
        passed = stats.get("completed", 0)
        running = stats.get("running", 0)
        skipped = stats.get("skipped", 0)
        failed = stats.get("failed", 0)
        aborted = stats.get("aborted", 0)
        text.append(f"  {passed}/{total}", style="green")
        if running:
            text.append(f" {running}R", style="bold blue")
        if skipped:
            text.append(f" {skipped}S", style="cyan")
        if failed:
            text.append(f" {failed}F", style="bold red")
        if aborted:
            text.append(f" {aborted}A", style="yellow")

    return text
