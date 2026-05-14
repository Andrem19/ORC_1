"""
Unit tests for direct-runtime no-op terminal-write coercion.
"""

from __future__ import annotations

from app.services.direct_execution.no_op_coercion import coerce_no_op_terminal_write


def _read_result(tool: str = "research_memory") -> dict:
    return {
        "kind": "tool_result",
        "tool": tool,
        "arguments": {"action": None, "project_id": "proj_1"},
        "payload": {"ok": True, "payload": {"structuredContent": {"data": {}}}},
    }


class TestResearchShortlistCoercion:
    def test_no_coercion_when_profile_unknown(self) -> None:
        coerced, note = coerce_no_op_terminal_write(
            runtime_profile="generic_read",
            tool_name="research_memory",
            arguments={"action": None, "kind": None, "project_id": None, "record": None},
            transcript=[_read_result()],
            project_id="proj_1",
            allowed_tools={"research_memory"},
        )
        assert coerced is None and note is None

    def test_no_coercion_without_prior_call(self) -> None:
        coerced, note = coerce_no_op_terminal_write(
            runtime_profile="research_shortlist",
            tool_name="research_memory",
            arguments={"action": None, "kind": None, "project_id": None, "record": None},
            transcript=[],
            project_id="proj_1",
            allowed_tools={"research_memory"},
        )
        assert coerced is None and note is None

    def test_coerces_all_null_args_after_one_read(self) -> None:
        coerced, note = coerce_no_op_terminal_write(
            runtime_profile="research_shortlist",
            tool_name="research_memory",
            arguments={"action": None, "kind": None, "project_id": None, "record": None},
            transcript=[_read_result()],
            project_id="cycle-invariants-v1-9f1fb256",
            allowed_tools={"research_memory", "research_map"},
        )
        assert coerced is not None
        assert note and "coerced no-op research_memory" in note
        assert coerced["action"] == "create"
        assert coerced["kind"] == "milestone"
        assert coerced["project_id"] == "cycle-invariants-v1-9f1fb256"
        record = coerced["record"]
        assert record["metadata"]["shortlist_families"]
        assert record["metadata"]["novelty_justification_present"] is True
        assert len(record["content"]["candidates"]) >= 1
        for candidate in record["content"]["candidates"]:
            assert candidate["family"]
            assert candidate["why_new"]
            assert candidate["relative_to"] == ["base", "v1-v12"]

    def test_coerces_repeated_search_action(self) -> None:
        coerced, _ = coerce_no_op_terminal_write(
            runtime_profile="research_shortlist",
            tool_name="research_memory",
            arguments={"action": "search", "project_id": "proj_1"},
            transcript=[_read_result(), _read_result()],
            project_id="proj_1",
            allowed_tools={"research_memory"},
        )
        assert coerced is not None
        assert coerced["action"] == "create"

    def test_does_not_touch_legitimate_milestone_write(self) -> None:
        existing = {
            "action": "create",
            "kind": "milestone",
            "project_id": "proj_1",
            "record": {
                "title": "real",
                "metadata": {
                    "shortlist_families": ["x"],
                    "novelty_justification_present": True,
                },
                "content": {"candidates": [{"family": "x", "why_new": "y", "relative_to": ["base"]}]},
            },
        }
        coerced, note = coerce_no_op_terminal_write(
            runtime_profile="research_shortlist",
            tool_name="research_memory",
            arguments=existing,
            transcript=[_read_result()],
            project_id="proj_1",
            allowed_tools={"research_memory"},
        )
        assert coerced is None and note is None

    def test_does_not_coerce_create_other_kind(self) -> None:
        # If model is creating a non-milestone with non-blank record, leave it.
        coerced, _ = coerce_no_op_terminal_write(
            runtime_profile="research_shortlist",
            tool_name="research_memory",
            arguments={
                "action": "create",
                "kind": "note",
                "project_id": "proj_1",
                "record": {"title": "interim", "summary": "interim"},
            },
            transcript=[_read_result()],
            project_id="proj_1",
            allowed_tools={"research_memory"},
        )
        assert coerced is None

    def test_falls_back_to_research_record_when_research_memory_not_allowed(self) -> None:
        coerced, _ = coerce_no_op_terminal_write(
            runtime_profile="research_shortlist",
            tool_name="research_record",
            arguments={"action": None, "project_id": None, "record": None},
            transcript=[_read_result(tool="research_record")],
            project_id="proj_1",
            allowed_tools={"research_record", "research_map"},
        )
        assert coerced is not None
        assert coerced["action"] == "create"

    def test_uses_placeholder_project_id_when_unconfirmed(self) -> None:
        coerced, _ = coerce_no_op_terminal_write(
            runtime_profile="research_shortlist",
            tool_name="research_memory",
            arguments={"action": None, "project_id": None, "record": None},
            transcript=[_read_result()],
            project_id="",
            allowed_tools={"research_memory"},
        )
        assert coerced is not None
        assert coerced["project_id"] == "project-id"
