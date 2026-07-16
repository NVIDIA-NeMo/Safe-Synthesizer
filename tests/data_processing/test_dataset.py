# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from nemo_safe_synthesizer.data_processing.dataset import make_json_schema, relax_numeric_bounds


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


def test_relax_numeric_bounds_on_make_json_schema_output():
    df = pd.DataFrame(
        {
            "flt": [0.11, 0.52, 0.93, 0.34, 0.75, 0.16, 0.27, 0.68, 0.49, 0.80, 0.13, 0.94],
            "txt": ["a", "bb", "ccc", "dd", "e", "ff", "g", "hh", "iii", "j", "kk", "l"],
        }
    )
    schema = make_json_schema(df)
    assert schema["properties"]["flt"]["type"] == "number"
    assert "maximum" in schema["properties"]["flt"]

    relaxed = relax_numeric_bounds(schema)
    assert "minimum" not in relaxed["properties"]["flt"]
    assert "maximum" not in relaxed["properties"]["flt"]
    # String bounds are untouched.
    assert relaxed["properties"]["txt"].get("maxLength") == schema["properties"]["txt"].get("maxLength")
