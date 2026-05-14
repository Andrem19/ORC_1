"""Tests for _coerce_string_json_fields and _coerce_nested_json_strings in lmstudio_tool_loop."""

from __future__ import annotations

import json

from app.services.direct_execution.lmstudio_tool_loop import (
    _coerce_nested_json_strings,
    _coerce_string_json_fields,
)


def test_coerces_record_string_to_dict():
    arguments = {
        "action": "create",
        "kind": "milestone",
        "project_id": "proj-123",
        "record": json.dumps({"title": "test", "metadata": {"stage": "formation"}}),
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert isinstance(result["record"], dict)
    assert result["record"]["title"] == "test"
    assert any("coerced record" in n for n in notes)


def test_coerces_record_string_with_candidates():
    """The exact MiniMax pattern from the failing run."""
    record_dict = {
        "content": {
            "candidates": [
                {"family": "funding dislocation", "why_new": "new", "relative_to": ["base"]}
            ]
        },
        "metadata": {
            "outcome": "shortlist_recorded",
            "shortlist_families": ["funding dislocation"],
        },
    }
    arguments = {
        "action": "create",
        "kind": "milestone",
        "project_id": "active-signal-v1-cycle-1",
        "record": json.dumps(record_dict),
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert isinstance(result["record"], dict)
    assert result["record"]["content"]["candidates"][0]["family"] == "funding dislocation"
    assert result["record"]["metadata"]["outcome"] == "shortlist_recorded"


def test_coerces_spec_string_to_dict():
    arguments = {
        "action": "materialize",
        "spec": json.dumps({"input_fields": ["a", "b"], "target": "label"}),
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert isinstance(result["spec"], dict)
    assert result["spec"]["input_fields"] == ["a", "b"]


def test_coerces_payload_string_to_dict():
    arguments = {
        "action": "create",
        "payload": json.dumps({"name": "test", "signals": [{"direction": "long"}]}),
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert isinstance(result["payload"], dict)


def test_coerces_array_string_to_list():
    arguments = {
        "action": "create",
        "record": json.dumps([{"a": 1}, {"b": 2}]),
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert isinstance(result["record"], list)
    assert len(result["record"]) == 2


def test_skips_non_json_string():
    arguments = {
        "action": "create",
        "record": "just a plain title string",
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert result["record"] == "just a plain title string"
    assert not notes


def test_skips_already_dict():
    arguments = {
        "action": "create",
        "record": {"title": "already a dict"},
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert result["record"] == {"title": "already a dict"}
    assert not notes


def test_skips_invalid_json_string():
    arguments = {
        "action": "create",
        "record": "{not valid json",
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert result["record"] == "{not valid json"
    assert not notes


def test_skips_empty_string():
    arguments = {
        "action": "create",
        "record": "",
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert result["record"] == ""
    assert not notes


def test_skips_non_string_values():
    arguments = {
        "action": "create",
        "record": 42,
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert result["record"] == 42
    assert not notes


def test_does_not_touch_unknown_fields():
    """Fields not in _STRING_JSON_FIELDS should not be coerced."""
    arguments = {
        "action": "create",
        "some_other_field": '{"key": "value"}',
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert result["some_other_field"] == '{"key": "value"}'
    assert not notes


def test_multiple_fields_coerced():
    arguments = {
        "record": json.dumps({"title": "test"}),
        "spec": json.dumps({"input_fields": ["a"]}),
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert isinstance(result["record"], dict)
    assert isinstance(result["spec"], dict)
    assert len(notes) == 2


def test_mixed_valid_and_invalid():
    arguments = {
        "record": json.dumps({"title": "test"}),
        "spec": "not json at all",
        "payload": None,
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert isinstance(result["record"], dict)
    assert result["spec"] == "not json at all"
    assert result["payload"] is None
    assert len(notes) == 1


# ---------- nested string coercion ----------


def test_coerces_python_repr_list_inside_metadata():
    """MiniMax pattern: metadata.shortlist_families as Python repr string."""
    arguments = {
        "action": "create",
        "kind": "milestone",
        "project_id": "proj-123",
        "record": {
            "content": "Shortlist recorded",
            "metadata": {
                "outcome": "shortlist_recorded",
                "shortlist_families": "['cross_tf_momentum_divergence', 'liquidity_exhaustion_reversal']",
                "stage": "hypothesis_formation",
            },
        },
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert isinstance(result["record"]["metadata"]["shortlist_families"], list)
    assert result["record"]["metadata"]["shortlist_families"] == [
        "cross_tf_momentum_divergence",
        "liquidity_exhaustion_reversal",
    ]
    assert any("nested" in n for n in notes)


def test_coerces_python_repr_dict_inside_candidates():
    """MiniMax pattern: candidates as list of Python repr dicts."""
    arguments = {
        "action": "create",
        "kind": "milestone",
        "record": {
            "candidates": [
                "{'family': 'funding_dislocation', 'relative_to': ['base'], 'why_new': 'new approach'}",
            ],
        },
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert isinstance(result["record"]["candidates"][0], dict)
    assert result["record"]["candidates"][0]["family"] == "funding_dislocation"
    assert result["record"]["candidates"][0]["relative_to"] == ["base"]


def test_coerces_json_string_nested_inside_dict():
    """JSON-formatted string inside nested dict."""
    arguments = {
        "action": "create",
        "record": {
            "metadata": {
                "tags": '["momentum", "volatility"]',
            },
        },
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert result["record"]["metadata"]["tags"] == ["momentum", "volatility"]


def test_coerces_deeply_nested_string():
    """Two levels deep: string → dict → string → list."""
    arguments = {
        "record": {
            "level1": {
                "level2": '["deep_value"]',
            },
        },
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert result["record"]["level1"]["level2"] == ["deep_value"]


def test_skips_non_collection_string():
    """Plain string that starts with [ or { but is not parseable."""
    arguments = {
        "record": {
            "content": "[not a real list",
        },
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert result["record"]["content"] == "[not a real list"


def test_skips_plain_text_string():
    """Normal text strings are not coerced."""
    arguments = {
        "record": {
            "content": "Some plain text description",
            "title": "Just a title",
        },
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert result["record"]["content"] == "Some plain text description"
    assert result["record"]["title"] == "Just a title"


def test_coerces_top_level_then_nested():
    """Top-level record is JSON string, AND nested value is Python repr."""
    arguments = {
        "action": "create",
        "record": json.dumps({
            "metadata": {
                "shortlist_families": "['momentum', 'funding']",
            },
        }),
    }
    result, notes = _coerce_string_json_fields(dict(arguments))
    assert isinstance(result["record"], dict)
    assert result["record"]["metadata"]["shortlist_families"] == ["momentum", "funding"]


def test_exact_incident_pattern():
    """Exact pattern from incident 20260415T234606Z-ae9fbc5c."""
    arguments = {
        "action": "create",
        "kind": "milestone",
        "project_id": "cycle-invariants-v1-c619a3cc",
        "record": {
            "content": "Первая волна новых гипотез сигналов",
            "metadata": {
                "outcome": "shortlist_recorded",
                "shortlist_families": (
                    "['cross_tf_momentum_divergence', 'liquidity_exhaustion_reversal', "
                    "'smart_money_flow_divergence', 'volatility_contraction_breakout', "
                    "'funding_sentiment_extreme', 'cycle_phase_alignment']"
                ),
                "stage": "hypothesis_formation",
                "novelty_justification_present": True,
            },
            "candidates": [
                "{'family': 'cross_tf_momentum_divergence', 'relative_to': ['base', 'v1-v12'], "
                "'why_new': 'Использует дивергенцию между 4h и 1h импульсом как фильтр входов.'}",
                "{'family': 'liquidity_exhaustion_reversal', 'relative_to': ['base', 'v1-v12'], "
                "'why_new': 'Динамическое отслеживание истощения пула ликвидности.'}",
            ],
        },
    }
    result, notes = _coerce_string_json_fields(dict(arguments))

    # shortlist_families should be a real list
    families = result["record"]["metadata"]["shortlist_families"]
    assert isinstance(families, list)
    assert len(families) == 6
    assert "cross_tf_momentum_divergence" in families
    assert "cycle_phase_alignment" in families

    # candidates should be real dicts
    assert isinstance(result["record"]["candidates"][0], dict)
    assert result["record"]["candidates"][0]["family"] == "cross_tf_momentum_divergence"
    assert isinstance(result["record"]["candidates"][1], dict)
    assert result["record"]["candidates"][1]["family"] == "liquidity_exhaustion_reversal"


# ---------- _coerce_nested_json_strings direct tests ----------


def test_nested_json_strings_returns_changed_flag():
    value = {"key": "['a', 'b']"}
    result, changed = _coerce_nested_json_strings(value)
    assert changed is True
    assert result == {"key": ["a", "b"]}


def test_nested_json_strings_no_change():
    value = {"key": "plain text", "num": 42}
    result, changed = _coerce_nested_json_strings(value)
    assert changed is False
    assert result == {"key": "plain text", "num": 42}


def test_nested_json_strings_depth_limit():
    """Deep nesting is capped at depth 10."""
    deep = "['value']"
    for _ in range(15):
        deep = {"nested": deep}
    result, changed = _coerce_nested_json_strings(deep)
    # Should not crash and should coerce some levels
    assert isinstance(result, dict)
