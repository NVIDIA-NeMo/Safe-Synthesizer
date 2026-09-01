# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.artifacts.base.fields import FieldType
from nemo_safe_synthesizer.data_processing.dataset_profile import (
    BinaryConstraints,
    CategoricalConstraints,
    ColumnProfile,
    DatasetProfile,
    FloatConstraints,
    IntegerConstraints,
    _ColumnStats,
    _discover_column_profile,
    discover_dataset_profile,
    relax_numeric_bounds,
)
from nemo_safe_synthesizer.errors import DataError


def test_discovery_classifies_expected_column_types():
    df = pd.DataFrame(
        {
            "empty": [None] * 16,
            "binary": [True, False] * 8,
            "integer": list(range(16)),
            "float": [float(value) / 10 for value in range(16)],
            "categorical": ["red", "blue"] * 8,
            "text": [f"a sentence with several words {value}" for value in range(16)],
            "other": [f"id-{value}" for value in range(16)],
        }
    )

    profile = discover_dataset_profile(df)

    assert {name: column.field_type for name, column in profile.columns.items()} == {
        "empty": FieldType.EMPTY,
        "binary": FieldType.BINARY,
        "integer": FieldType.INTEGER,
        "float": FieldType.FLOAT,
        "categorical": FieldType.BINARY,
        "text": FieldType.TEXT,
        "other": FieldType.OTHER,
    }
    assert isinstance(profile.columns["binary"].constraints, BinaryConstraints)
    assert profile.columns["binary"].constraints.enum_values == [False, True]
    assert isinstance(profile.columns["float"].constraints, FloatConstraints)
    assert profile.columns["float"].constraints.min_value == 0.0
    assert profile.columns["float"].constraints.max_value == 1.5


def test_nullable_enum_export_includes_null():
    df = pd.DataFrame({"category": ["red", "blue", None] * 6})

    profile_schema = discover_dataset_profile(df).to_json_schema()

    assert profile_schema["properties"]["category"]["enum"] == ["blue", "red", None]
    assert "category" not in profile_schema["required"]


def test_low_cardinality_integer_classified_as_categorical():
    """Enum heuristic must win over integer dtype (legacy schema inference order)."""
    column = _discover_column_profile(
        "status",
        _ColumnStats(
            row_count=60,
            non_null_count=60,
            unique_count=3,
            missing_count=0,
            singleton_count=0,
            min_str_length=1,
            max_str_length=1,
            avg_space_count=0.0,
            is_float=False,
            is_integer=True,
            min_value=1,
            max_value=3,
            enum_values=(1, 2, 3),
        ),
    )

    assert column.field_type == FieldType.CATEGORICAL
    assert isinstance(column.constraints, CategoricalConstraints)
    assert column.constraints.enum_values == [1, 2, 3]


def test_json_schema_maps_scalar_types_and_nullability():
    df = pd.DataFrame(
        {
            "integer": list(range(11, 23)),
            "float": [float(value) / 10 for value in range(11)] + [None],
            "text": [f"one two three {value}" for value in range(12)],
            "empty": [None] * 12,
        }
    )

    schema = discover_dataset_profile(df).to_json_schema()

    assert schema["properties"]["integer"] == {"type": "integer", "minimum": 11, "maximum": 22}
    # A nullable column becomes a type list, which the grammar cannot bound, so
    # its range is dropped rather than left as a validation-only rejection gate.
    assert schema["properties"]["float"] == {"type": ["number", "null"]}
    assert schema["properties"]["empty"] == {"type": "null"}
    assert schema["required"] == ["integer", "text"]


def test_nullable_columns_drop_unenforceable_bounds():
    df = pd.DataFrame(
        {
            "code": [f"code-{value}" for value in range(11)] + [None],
            "amount": [float(value) for value in range(11)] + [None],
        }
    )

    schema = discover_dataset_profile(df).to_json_schema()

    assert schema["properties"]["code"] == {"type": ["string", "null"]}
    assert schema["properties"]["amount"] == {"type": ["number", "null"]}
    assert schema["required"] == []


def test_discover_column_profile_from_stats_classifies_without_series():
    empty = _discover_column_profile(
        "empty",
        _ColumnStats(
            row_count=5,
            non_null_count=0,
            unique_count=0,
            missing_count=5,
            singleton_count=0,
            min_str_length=0,
            max_str_length=0,
            avg_space_count=None,
            is_float=False,
            is_integer=False,
            min_value=None,
            max_value=None,
            enum_values=(),
        ),
    )
    assert empty.field_type == FieldType.EMPTY
    assert empty.nullable is True

    integer = _discover_column_profile(
        "age",
        _ColumnStats(
            row_count=100,
            non_null_count=100,
            unique_count=50,
            missing_count=0,
            singleton_count=50,
            min_str_length=1,
            max_str_length=3,
            avg_space_count=0.0,
            is_float=False,
            is_integer=True,
            min_value=0,
            max_value=99,
            enum_values=tuple(range(50)),
        ),
    )
    assert integer.field_type == FieldType.INTEGER
    assert isinstance(integer.constraints, IntegerConstraints)
    assert integer.constraints.min_value == 0
    assert integer.constraints.max_value == 99

    text = _discover_column_profile(
        "notes",
        _ColumnStats(
            row_count=11,
            non_null_count=10,
            unique_count=10,
            missing_count=1,
            singleton_count=10,
            min_str_length=8,
            max_str_length=40,
            avg_space_count=3.5,
            is_float=False,
            is_integer=False,
            min_value=None,
            max_value=None,
            enum_values=(),
        ),
    )
    assert text.field_type == FieldType.TEXT
    assert text.nullable is True


def test_nested_columns_are_rejected():
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "conversation": [[{"role": "user"}], [{"role": "assistant"}]],
            "metadata": [{"flagged": False}, {"flagged": True}],
        }
    )

    with pytest.raises(DataError, match=r"\['conversation', 'metadata'\]"):
        discover_dataset_profile(df)


def test_categorical_constraints_require_enum_values():
    with pytest.raises(ValidationError):
        CategoricalConstraints(enum_values=[])


def test_binary_constraints_require_exactly_two_values():
    with pytest.raises(ValidationError):
        BinaryConstraints(enum_values=["only-one"])


def test_profile_override_requires_exact_column_names():
    profile = DatasetProfile(
        columns={
            "a": ColumnProfile(
                name="a",
                nullable=False,
                constraints=IntegerConstraints(min_value=1, max_value=2),
            )
        }
    )

    with pytest.raises(ValueError, match="must exactly match"):
        profile.validate_against_dataframe(pd.DataFrame({"different": [1, 2]}))


def test_relax_numeric_bounds_strips_float_bounds():
    schema = {
        "type": "object",
        "properties": {
            "ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["ratio"],
    }
    relaxed = relax_numeric_bounds(schema)
    assert "minimum" not in relaxed["properties"]["ratio"]
    assert "maximum" not in relaxed["properties"]["ratio"]
    assert relaxed["properties"]["ratio"]["type"] == "number"


def test_relax_numeric_bounds_keeps_integer_and_enum_bounds():
    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": 10},
            "category": {"enum": [1, 2, 3]},
        },
    }
    relaxed = relax_numeric_bounds(schema)
    assert relaxed["properties"]["count"] == {"type": "integer", "minimum": 1, "maximum": 10}
    assert relaxed["properties"]["category"] == {"enum": [1, 2, 3]}


def test_relax_numeric_bounds_handles_type_list_with_null():
    schema = {
        "type": "object",
        "properties": {
            "maybe_float": {"type": ["number", "null"], "minimum": -5.0, "maximum": 5.0},
            "maybe_int": {"type": ["integer", "null"], "minimum": 0, "maximum": 9},
        },
    }
    relaxed = relax_numeric_bounds(schema)
    # A nullable float is still relaxed; a nullable integer keeps its bounds.
    assert "minimum" not in relaxed["properties"]["maybe_float"]
    assert "maximum" not in relaxed["properties"]["maybe_float"]
    assert relaxed["properties"]["maybe_int"]["minimum"] == 0
    assert relaxed["properties"]["maybe_int"]["maximum"] == 9


def test_relax_numeric_bounds_does_not_mutate_input():
    schema = {
        "type": "object",
        "properties": {"ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
    }
    relax_numeric_bounds(schema)
    assert schema["properties"]["ratio"]["minimum"] == 0.0
    assert schema["properties"]["ratio"]["maximum"] == 1.0


def test_relax_numeric_bounds_on_profile_json_schema_output():
    df = pd.DataFrame(
        {
            "flt": [0.11, 0.52, 0.93, 0.34, 0.75, 0.16, 0.27, 0.68, 0.49, 0.80, 0.13, 0.94],
            "txt": ["a", "bb", "ccc", "dd", "e", "ff", "g", "hh", "iii", "j", "kk", "l"],
        }
    )
    schema = discover_dataset_profile(df).to_json_schema()
    assert schema["properties"]["flt"]["type"] == "number"
    assert "maximum" in schema["properties"]["flt"]

    relaxed = relax_numeric_bounds(schema)
    assert "minimum" not in relaxed["properties"]["flt"]
    assert "maximum" not in relaxed["properties"]["flt"]
    # String bounds are untouched.
    assert relaxed["properties"]["txt"].get("maxLength") == schema["properties"]["txt"].get("maxLength")
