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
                                    "metadata": {
                                        "shortlist_families": [
                                            "Cross-Market Correlation Signals",
                                            "Order Flow Imbalance Signals",
                                        ]
                                    },
                                    "content": {
                                        "candidates": [
                                            {"family": "Cross-Market Correlation Signals", "why_new": "new"},
                                            {"family": "Order Flow Imbalance Signals", "why_new": "new"},
                                        ]
                                    },
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


def test_derive_facts_from_transcript_does_not_infer_shortlist_from_generic_text() -> None:
    facts = derive_facts_from_transcript(
        [
            {
                "kind": "tool_result",
                "tool": "research_memory",
                "arguments": {"kind": "milestone"},
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {
                                "record": {
                                    "summary": "Baseline explicitly configured.",
                                    "content": {"text": "Naming convention recorded."},
                                }
                            },
                        }
                    }
                },
            },
        ]
    )

    assert "research.shortlist_families" not in facts


def test_derive_facts_from_transcript_collects_research_setup_requirements() -> None:
    facts = derive_facts_from_transcript(
        [
            {
                "kind": "tool_result",
                "tool": "research_project",
                "arguments": {"action": "create"},
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {
                                "project": {
                                    "project_id": "proj_1",
                                    "default_branch_id": "branch_1",
                                }
                            },
                        }
                    }
                },
            },
            {
                "kind": "tool_result",
                "tool": "research_project",
                "arguments": {
                    "action": "set_baseline",
                    "project_id": "proj_1",
                    "snapshot_id": "active-signal-v1",
                    "version": 1,
                },
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {"project_id": "proj_1"},
                        }
                    }
                },
            },
            {
                "kind": "tool_result",
                "tool": "research_map",
                "arguments": {"action": "define", "project_id": "proj_1"},
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {
                                "state_summary": {"atlas_defined": True, "dimension_count": 3},
                            },
                        }
                    }
                },
            },
            {
                "kind": "tool_result",
                "tool": "research_memory",
                "arguments": {
                    "action": "create",
                    "project_id": "proj_1",
                    "record": {
                        "metadata": {
                            "invariants": {"baseline": "active-signal-v1 @1"},
                            "naming_convention": {"project": "v{N}-cycle-invariants"},
                        }
                    },
                },
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {
                                "record_refs": {"memory_node_id": "node_mem_1"},
                                "record": {"node_id": "node_mem_1"},
                            },
                        }
                    }
                },
            },
        ],
        runtime_profile="research_setup",
    )

    assert facts["research.project_id"] == "proj_1"
    assert facts["research.branch_id"] == "branch_1"
    assert facts["research.baseline_configured"] is True
    assert facts["research.baseline_snapshot_id"] == "active-signal-v1"
    assert facts["research.baseline_version"] == 1
    assert facts["research.atlas_defined"] is True
    assert facts["research.memory_node_id"] == "node_mem_1"
    assert facts["research.invariants_recorded"] is True
    assert facts["research.naming_recorded"] is True


def test_failed_setup_tool_result_does_not_satisfy_research_setup_facts() -> None:
    facts = derive_facts_from_transcript(
        [
            {
                "kind": "tool_result",
                "tool": "research_project",
                "arguments": {
                    "action": "set_baseline",
                    "project_id": "proj_1",
                    "snapshot_id": "active-signal-v1",
                    "version": 1,
                },
                "payload": {
                    "error": "baseline failed",
                    "payload": {
                        "structuredContent": {
                            "status": "error",
                            "data": {"project_id": "proj_1"},
                        }
                    },
                },
            }
        ],
        runtime_profile="research_setup",
    )

    assert "research.baseline_configured" not in facts


def test_derive_facts_from_transcript_ignores_transport_ids_and_derives_backtests_handoff() -> None:
    facts = derive_facts_from_transcript(
        [
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {
                                "request_id": 1,
                                "correlation_id": 1,
                                "server_session_id": "ddc3dce8bbc14437877f5a1f045b3d2b",
                                "feature_long_job": "cond-f07199b451c1",
                                "feature_short_job": "cond-8972f7822bb2",
                                "diagnostics_run": "20260411-193208-40dbd831",
                            },
                        }
                    }
                },
            }
        ]
    )

    assert "1" not in facts.get("direct.created_ids", [])
    assert "ddc3dce8bbc14437877f5a1f045b3d2b" not in facts.get("direct.created_ids", [])
    assert facts["backtests.candidate_handles"] == {
        "feature_long_job": "cond-f07199b451c1",
        "feature_short_job": "cond-8972f7822bb2",
    }
    assert "20260411-193208-40dbd831" in facts["backtests.analysis_refs"]


def test_derive_facts_from_transcript_extracts_shortlist_novelty_contract() -> None:
    facts = derive_facts_from_transcript(
        [
            {
                "kind": "tool_result",
                "tool": "research_record",
                "arguments": {
                    "action": "create",
                    "kind": "milestone",
                    "project_id": "proj_1",
                    "record": {
                        "metadata": {
                            "shortlist_families": ["funding dislocation", "expiry proximity"],
                            "novelty_justification_present": True,
                        },
                        "content": {
                            "candidates": [
                                {
                                    "family": "funding dislocation",
                                    "why_new": "New versus the base space and v1-v12.",
                                    "relative_to": ["base", "v1-v12"],
                                }
                            ]
                        },
                    },
                },
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {
                                "project_id": "proj_1",
                                "record_refs": {"memory_node_id": "node_mem_1"},
                                "record": {"node_id": "node_mem_1"},
                            },
                        }
                    }
                },
            }
        ],
        runtime_profile="research_shortlist",
    )

    assert facts["research.project_id"] == "proj_1"
    assert facts["research.memory_node_id"] == "node_mem_1"
    assert facts["research.shortlist_families"] == ["funding dislocation", "expiry proximity"]
    assert facts["research.novelty_justification_present"] is True


def test_derive_facts_from_transcript_collects_supported_refs_and_mutation_counts() -> None:
    facts = derive_facts_from_transcript(
        [
            {
                "kind": "tool_result",
                "tool": "research_memory",
                "arguments": {
                    "action": "create",
                    "project_id": "proj_1",
                    "record": {"title": "Wave-1 shortlist"},
                },
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {
                                "project_id": "proj_1",
                                "record": {"node_id": "node_1"},
                                "refs": ["note_1"],
                            },
                        }
                    }
                },
            },
            {
                "kind": "tool_result",
                "tool": "research_memory",
                "arguments": {
                    "action": "search",
                    "query": "wave-1 shortlist",
                },
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {"project_id": "proj_1"},
                        }
                    }
                },
            },
        ]
    )

    assert facts["direct.successful_tool_count"] == 2
    assert facts["direct.successful_mutating_tool_count"] == 1
    assert facts["direct.successful_tool_names"] == ["research_memory"]
    assert "transcript:1:research_memory" in facts["direct.supported_evidence_refs"]
    assert "node_1" in facts["direct.supported_evidence_refs"]
    assert "note_1" in facts["direct.supported_evidence_refs"]


def test_derive_facts_from_transcript_excludes_transport_ids_from_created_and_supported_refs() -> None:
    facts = derive_facts_from_transcript(
        [
            {
                "kind": "tool_result",
                "tool": "backtests_runs",
                "arguments": {"action": "inspect", "view": "list"},
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "request_id": 1,
                            "correlation_id": "3",
                            "server_session_id": "0f116f24435a4d9ebe19eb11105e183f",
                            "data": {
                                "id": 1,
                                "result_id": "0f116f24435a4d9ebe19eb11105e183f",
                                "record": {"node_id": "node-valid"},
                                "saved_runs": [{"run_id": "20260413-143910-3db43a14"}],
                            },
                        }
                    }
                },
            }
        ]
    )

    assert facts["direct.created_ids"] == ["node-valid"]
    assert "node-valid" in facts["direct.supported_evidence_refs"]
    assert "1" not in facts["direct.supported_evidence_refs"]
    assert "3" not in facts["direct.supported_evidence_refs"]
    assert "0f116f24435a4d9ebe19eb11105e183f" not in facts["direct.supported_evidence_refs"]


# ---------- research setup metadata fact derivation ----------


def test_derive_research_invariants_from_metadata() -> None:
    facts = derive_facts_from_transcript(
        [
            {
                "kind": "tool_result",
                "tool": "research_memory",
                "arguments": {
                    "action": "create",
                    "kind": "note",
                    "record": {
                        "content": {"title": "Invariants", "text": "..."},
                        "metadata": {"invariants": "close_price > 0"},
                    },
                },
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "data": {
                                "record": {
                                    "node_id": "node_inv",
                                },
                            }
                        }
                    }
                },
            },
        ]
    )
    assert facts.get("research.invariants_recorded") is True


def test_derive_research_naming_convention_from_metadata() -> None:
    facts = derive_facts_from_transcript(
        [
            {
                "kind": "tool_result",
                "tool": "research_memory",
                "arguments": {
                    "action": "create",
                    "kind": "note",
                    "record": {
                        "content": {"title": "Naming", "text": "..."},
                        "metadata": {"naming_convention": "cf_<family>_<direction>"},
                    },
                },
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "data": {
                                "record": {
                                    "node_id": "node_naming",
                                },
                            }
                        }
                    }
                },
            },
        ]
    )
    assert facts.get("research.naming_recorded") is True


def test_derive_research_setup_missing_when_no_metadata() -> None:
    facts = derive_facts_from_transcript(
        [
            {
                "kind": "tool_result",
                "tool": "research_memory",
                "arguments": {
                    "action": "create",
                    "kind": "note",
                    "record": {
                        "content": {"title": "Something", "text": "..."},
                    },
                },
                "payload": {
                    "payload": {
                        "structuredContent": {
                            "data": {
                                "record": {
                                    "node_id": "node_no_meta",
                                },
                            }
                        }
                    }
                },
            },
        ]
    )
    assert "research.invariants_recorded" not in facts
    assert "research.naming_recorded" not in facts

