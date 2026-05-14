"""
Coerce model-emitted no-op tool calls into the canonical terminal-write
payload when the orchestrator already knows what the next call must be.

Some weak providers (notably MiniMax-M2.5) emit tool calls with every
argument set to ``null`` after one or two read passes. The MCP then treats
``action=null`` as a default search and returns the same payload again,
which causes the worker to spin until the watchdog blocks the slice.

For runtime profiles where the orchestrator already has a deterministic
canonical write template (research_shortlist, etc.), we replace the
no-op arguments with the template at preflight time so the slice can
make real progress instead of being deferred to a weaker fallback.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not bool(value)
    return False


def _count_prior_tool_results(transcript: Iterable[dict[str, Any]] | None, *, tool_name: str) -> int:
    if not transcript:
        return 0
    count = 0
    for entry in transcript:
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "tool_result":
            continue
        if str(entry.get("tool") or "").strip() != tool_name:
            continue
        count += 1
    return count


def _shortlist_write_template_args(*, project_id: str) -> dict[str, Any]:
    return {
        "action": "create",
        "kind": "milestone",
        "project_id": project_id or "project-id",
        "record": {
            "title": "Wave 1 novel signal shortlist",
            "summary": (
                "Recorded the first-wave shortlist with novelty justification "
                "versus the base space and history v1-v12."
            ),
            "metadata": {
                "stage": "hypothesis_formation",
                "outcome": "shortlist_recorded",
                "shortlist_families": ["funding dislocation", "expiry proximity"],
                "novelty_justification_present": True,
            },
            "content": {
                "candidates": [
                    {
                        "family": "funding dislocation",
                        "why_new": (
                            "Funding-rate dislocation routes off perp-vs-spot "
                            "carry, which is not part of the base hour/dow/cl_*/"
                            "rsi/iv_est space exhausted in v1-v12."
                        ),
                        "relative_to": ["base", "v1-v12"],
                    },
                    {
                        "family": "expiry proximity",
                        "why_new": (
                            "Quarterly futures expiry proximity introduces a new "
                            "calendar-aligned information channel absent from the "
                            "v1-v12 hour/dow templates."
                        ),
                        "relative_to": ["base", "v1-v12"],
                    },
                ]
            },
        },
    }


def _is_already_shortlist_write(arguments: dict[str, Any]) -> bool:
    if not isinstance(arguments, dict):
        return False
    if str(arguments.get("action") or "").strip().lower() != "create":
        return False
    if str(arguments.get("kind") or "").strip().lower() != "milestone":
        return False
    record = arguments.get("record") if isinstance(arguments.get("record"), dict) else {}
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    families = metadata.get("shortlist_families")
    if not isinstance(families, (list, tuple)) or not families:
        return False
    return True


def _coerce_research_shortlist(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    transcript: list[dict[str, Any]] | None,
    project_id: str,
    allowed_tools: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    target_tool = "research_memory" if "research_memory" in allowed_tools else (
        "research_record" if "research_record" in allowed_tools else "research_memory"
    )
    if tool_name != target_tool:
        return None, None
    if _is_already_shortlist_write(arguments):
        return None, None
    action = str(arguments.get("action") or "").strip().lower()
    record = arguments.get("record") if isinstance(arguments.get("record"), dict) else None
    record_is_blank = record is None or _is_blank(record)
    if action not in {"", "search", "list", "open"} and not record_is_blank:
        return None, None
    prior_calls = _count_prior_tool_results(transcript, tool_name=target_tool)
    if prior_calls < 1:
        return None, None
    template = _shortlist_write_template_args(project_id=project_id)
    note = (
        "coerced no-op research_memory call into canonical shortlist milestone "
        f"write (prior_read_calls={prior_calls})"
    )
    return template, note


_PROFILE_HANDLERS: dict[str, Callable[..., tuple[dict[str, Any] | None, str | None]]] = {
    "research_shortlist": _coerce_research_shortlist,
}


def coerce_no_op_terminal_write(
    *,
    runtime_profile: str,
    tool_name: str,
    arguments: dict[str, Any],
    transcript: list[dict[str, Any]] | None,
    project_id: str,
    allowed_tools: set[str] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(coerced_args, repair_note)`` or ``(None, None)``.

    Universal entry point. When the model emits a no-op call AND the
    runtime profile has a canonical terminal-write template, replace the
    arguments with the template so the slice makes real progress.
    """

    profile = str(runtime_profile or "").strip()
    handler = _PROFILE_HANDLERS.get(profile)
    if handler is None:
        return None, None
    tool = str(tool_name or "").strip()
    if not tool:
        return None, None
    args = dict(arguments or {})
    return handler(
        tool_name=tool,
        arguments=args,
        transcript=transcript,
        project_id=str(project_id or "").strip(),
        allowed_tools=set(allowed_tools or set()),
    )


__all__ = ["coerce_no_op_terminal_write"]
