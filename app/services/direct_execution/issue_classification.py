"""
Shared classification helpers for contract-misuse vs infra/tool-availability failures.
"""

from __future__ import annotations

import json
from typing import Any

_CONTRACT_MISUSE_MARKERS = (
    "agent_contract_misuse",
    "schema_validation_failed",
    "tool_not_in_allowlist",
    "tool_prefixed_namespace_forbidden",
    "worker_action_type_invalid",
    "final_report_requires",
    "tool_call_requires",
    "abort_requires",
    "checkpoint_status_invalid",
    "not valid under any of the given schemas",
    "unknown research project",
    "direct_contract_blocker",
)
_INFRA_UNAVAILABLE_MARKERS = (
    "dev_space1_tools_unavailable",
    "qwen_mcp_tools_unavailable",
    "qwen_tool_registry_missing",
    "mcp_catalog_tool_missing",
)


def classify_issue_payload(payload: Any) -> str:
    return classify_issue_text(json.dumps(payload, ensure_ascii=False))


def classify_issue_text(*parts: Any) -> str:
    haystack = " | ".join(str(item or "").strip().lower() for item in parts if str(item or "").strip())
    if not haystack:
        return "unknown"
    if any(marker in haystack for marker in _CONTRACT_MISUSE_MARKERS):
        return "contract_misuse"
    if any(marker in haystack for marker in _INFRA_UNAVAILABLE_MARKERS):
        return "infra_unavailable"
    return "other"


__all__ = ["classify_issue_payload", "classify_issue_text"]
