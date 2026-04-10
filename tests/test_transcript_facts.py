from __future__ import annotations

from app.services.direct_execution.transcript_facts import derive_facts_from_transcript


def test_derive_facts_from_transcript_collects_project_shortlist_and_hypothesis_refs() -> None:
    facts = derive_facts_from_transcript(
        [
            {
                "kind": "tool_result",
                "tool": "research_map",
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "data": {
                                "project_id": "proj_1",
                                "dimensions": [{"name": "anchor"}, {"name": "execution"}, {"name": "symbol"}],
                            }
                        }
                    }
                },
            },
            {
                "kind": "tool_result",
                "tool": "research_record",
                "arguments": {"kind": "hypothesis"},
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "data": {
                                "record": {
                                    "node_id": "node_1",
                                    "project_id": "proj_1",
                                    "title": "Hypothesis 1: Cross-Market Correlation Signals",
                                }
                            }
                        }
                    }
                },
            },
            {
                "kind": "tool_result",
                "tool": "research_record",
                "arguments": {"kind": "milestone"},
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "data": {
                                "record": {
                                    "content": {
                                        "text": "1. Cross-Market Correlation Signals - Novelty\n2. Order Flow Imbalance Signals - Novelty"
                                    }
                                }
                            }
                        }
                    }
                },
            },
        ]
    )

    assert facts["research.project_id"] == "proj_1"
    assert facts["research.hypothesis_refs"] == ["node_1"]
    assert facts["research.shortlist_families"] == [
        "Cross-Market Correlation Signals",
        "Order Flow Imbalance Signals",
    ]
    assert facts["atlas_dimensions"] == ["anchor", "execution", "symbol"]
