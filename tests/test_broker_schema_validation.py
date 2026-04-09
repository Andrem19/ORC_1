from __future__ import annotations

import pytest

from app.broker.schema_validation import ToolArgumentValidationError, validate_arguments


def test_validate_arguments_rejects_missing_required_field() -> None:
    schema = {
        "type": "object",
        "required": ["symbol", "timeframe"],
        "properties": {
            "symbol": {"type": "string"},
            "timeframe": {"type": "string"},
        },
    }

    with pytest.raises(ToolArgumentValidationError, match="missing required field 'timeframe'"):
        validate_arguments(schema=schema, arguments={"symbol": "BTCUSDT"}, tool_name="features_custom")


def test_validate_arguments_rejects_wrong_type() -> None:
    schema = {
        "type": "object",
        "required": ["bins"],
        "properties": {
            "bins": {"type": "integer"},
        },
    }

    with pytest.raises(ToolArgumentValidationError, match="expected integer"):
        validate_arguments(schema=schema, arguments={"bins": "twelve"}, tool_name="backtests_conditions")

