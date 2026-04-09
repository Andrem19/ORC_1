"""
Rich renderables for the broker-only runtime console.
"""

from __future__ import annotations

import time

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.runtime_console.models import ConsoleRuntimeState


def build_runtime_panel(state: ConsoleRuntimeState):
    summary = _build_runtime_summary(state)
    planner = _build_planner_table(state)
    slices = _build_slices_table(state)
    broker = _build_broker_table(state)
    footer = _build_footer(state)
    return Panel(
        Columns([summary, planner, slices, broker], expand=True, equal=False),
        title="Broker Runtime",
        subtitle=footer,
        border_style="cyan",
        padding=(0, 1),
    )


def _build_runtime_summary(state: ConsoleRuntimeState) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_row("status", _status_text(state.runtime_status, drain_mode=state.drain_mode))
    plan_label = _short(state.active_plan_id, 12) if state.active_plan_id else "-"
    table.add_row("plan", Text(plan_label, style="bold cyan"))
    table.add_row("cycle", Text(str(state.current_cycle), style="white"))
    table.add_row("errors", Text(str(state.total_errors), style="yellow" if state.total_errors else "green"))
    table.add_row("broker", _broker_health_text(state))
    if state.started_at_monotonic:
        table.add_row("uptime", Text(_elapsed(state.started_at_monotonic), style="cyan"))
    return Panel(table, title="Runtime", border_style="blue")


def _build_planner_table(state: ConsoleRuntimeState) -> Panel:
    planner = state.planner
    table = Table.grid(padding=(0, 1))
    table.add_row("state", Text(planner.status or "idle", style="yellow" if planner.status == "running" else "white"))
    table.add_row("adapter", Text(planner.adapter_name or "-", style="magenta"))
    if planner.max_attempts:
        table.add_row("attempt", Text(f"{planner.attempt}/{planner.max_attempts}", style="white"))
    if planner.status == "running" and planner.started_at_monotonic:
        table.add_row("elapsed", Text(_elapsed(planner.started_at_monotonic), style="cyan"))
    elif planner.last_error:
        table.add_row("last", Text(_short(planner.last_error, 44), style="red"))
    else:
        table.add_row("last", Text("-", style="dim"))
    return Panel(table, title="Planner", border_style="magenta")


def _build_slices_table(state: ConsoleRuntimeState) -> Panel:
    table = Table("slot", "task", "worker", "budget", "status", "summary", expand=True, box=None, padding=(0, 1))
    for slot in sorted(state.slices):
        item = state.slices[slot]
        task_label = _short(item.title or item.slice_id or "-", 24)
        worker_label = (item.worker_id or "-")[:8]
        budget = _budget_text(item.turns_used, item.turns_total, item.tool_calls_used, item.tool_calls_total)
        status = _slice_status_text(item.status, item.active_operation_ref, item.active_operation_status)
        table.add_row(
            str(slot),
            task_label,
            worker_label,
            budget,
            status,
            _short(item.last_summary or "-", 40),
        )
    if not state.slices:
        table.add_row("-", "-", "-", "-", Text("idle", style="dim"), "-")
    return Panel(table, title="Slices", border_style="green")


def _build_broker_table(state: ConsoleRuntimeState) -> Panel:
    table = Table("slot", "tool", "phase", "elapsed", "status", "op", expand=True, box=None, padding=(0, 1))
    for slot in sorted(state.broker_calls):
        item = state.broker_calls[slot]
        status = item.response_status or item.tool_response_status or "-"
        if item.error_class:
            status = f"{status} / {item.error_class}"
        elapsed_val = _elapsed(item.started_at_monotonic) if item.started_at_monotonic else f"{item.duration_ms}ms"
        table.add_row(
            str(slot),
            item.tool_name or "-",
            item.phase or "-",
            elapsed_val,
            status,
            _short(item.operation_ref or "-", 22),
        )
    if not state.broker_calls:
        table.add_row("-", "-", "-", "-", "-", "-")
    return Panel(table, title="Broker", border_style="yellow")


def _build_footer(state: ConsoleRuntimeState) -> Text:
    parts: list[tuple[str, str]] = []
    if state.last_warning:
        age = f" ({_elapsed(state.last_warning_at)} ago)" if state.last_warning_at else ""
        parts.append((f"warning: {_short(state.last_warning, 80)}{age}", "yellow"))
    if state.last_error:
        age = f" ({_elapsed(state.last_error_at)} ago)" if state.last_error_at else ""
        parts.append((f"error: {_short(state.last_error, 80)}{age}", "red"))
    if state.stop_reason:
        parts.append((f"stop: {state.stop_reason}", "bold red"))
    if not parts:
        parts.append(("broker-only runtime console active", "dim"))
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


def _broker_health_text(state: ConsoleRuntimeState) -> Text:
    tool_part = f"{state.broker_tool_count}t, " if state.broker_tool_count else ""
    warn_part = f"{len(state.broker_warnings)}w"
    detail = f"({tool_part}{warn_part})"
    if state.broker_status == "healthy":
        return Text(f"healthy {detail}", style="green")
    if state.broker_status == "degraded":
        return Text(f"degraded {detail}", style="yellow")
    if state.broker_status == "error":
        return Text("error", style="red")
    return Text(state.broker_status or "unknown", style="white")


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


def _elapsed(started_at_monotonic: float) -> str:
    return f"{max(0.0, time.monotonic() - started_at_monotonic):.1f}s"


def _short(text: str, limit: int) -> str:
    cleaned = str(text or "").replace("\n", " ").strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit - 3]}..."
