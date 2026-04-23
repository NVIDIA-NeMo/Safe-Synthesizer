# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_safe_synthesizer.data_processing.value_post_processor import (
    ValuePostProcessingError,
    ValuePostProcessor,
)


@pytest.fixture
def schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "intent": {"enum": ["play_music", "weather", "calendar"]},
            "label": {"type": "string", "minLength": 1, "maxLength": 10},
            "note": {"type": ["string", "null"]},
            "active": {"type": "boolean"},
            "height_nullable": {"type": ["number", "null"], "minimum": 0.0, "maximum": 3.0},
        },
        "required": ["age", "score", "intent", "label", "active"],
    }


def test_columns_order_preserved(schema):
    p = ValuePostProcessor(schema)
    assert p.columns == ["age", "score", "intent", "label", "note", "active", "height_nullable"]


class TestNumeric:
    def test_integer_parses_and_clamps(self, schema):
        p = ValuePostProcessor(schema)
        assert p.process("age", "42") == 42
        assert p.process("age", "200") == 120
        assert p.process("age", "-5") == 0

    def test_integer_rounds(self, schema):
        p = ValuePostProcessor(schema)
        assert p.process("age", "42.7") == 43

    def test_number_returns_float(self, schema):
        p = ValuePostProcessor(schema)
        result = p.process("score", "0.5")
        assert result == 0.5
        assert isinstance(result, float)

    def test_number_clamps_to_bounds(self, schema):
        p = ValuePostProcessor(schema)
        assert p.process("score", "1.5") == 1.0
        assert p.process("score", "-0.2") == 0.0

    def test_unparseable_raises(self, schema):
        p = ValuePostProcessor(schema)
        with pytest.raises(ValuePostProcessingError):
            p.process("age", "not-a-number")

    def test_nan_inf_raises(self, schema):
        p = ValuePostProcessor(schema)
        with pytest.raises(ValuePostProcessingError):
            p.process("score", "nan")
        with pytest.raises(ValuePostProcessingError):
            p.process("score", "inf")


class TestEnum:
    def test_exact_match(self, schema):
        p = ValuePostProcessor(schema)
        assert p.process("intent", "play_music") == "play_music"

    def test_case_insensitive(self, schema):
        p = ValuePostProcessor(schema)
        assert p.process("intent", "PLAY_MUSIC") == "play_music"
        assert p.process("intent", "Weather") == "weather"

    def test_close_match_snaps(self, schema):
        p = ValuePostProcessor(schema)
        assert p.process("intent", "play_musik") == "play_music"

    def test_no_match_raises(self, schema):
        p = ValuePostProcessor(schema)
        with pytest.raises(ValuePostProcessingError):
            p.process("intent", "completely_different")


class TestString:
    def test_passthrough(self, schema):
        p = ValuePostProcessor(schema)
        assert p.process("label", "hello") == "hello"

    def test_truncates_to_max_length(self, schema):
        p = ValuePostProcessor(schema)
        assert p.process("label", "x" * 50) == "x" * 10


class TestNullable:
    def test_empty_string_becomes_none(self, schema):
        p = ValuePostProcessor(schema)
        assert p.process("note", "") is None
        assert p.process("note", "null") is None
        assert p.process("note", "None") is None

    def test_non_null_value_passes_through(self, schema):
        p = ValuePostProcessor(schema)
        assert p.process("note", "a comment") == "a comment"

    def test_non_nullable_empty_raises(self, schema):
        p = ValuePostProcessor(schema)
        with pytest.raises(ValuePostProcessingError):
            p.process("label", "")

    def test_nullable_numeric(self, schema):
        p = ValuePostProcessor(schema)
        assert p.process("height_nullable", "null") is None
        assert p.process("height_nullable", "1.8") == pytest.approx(1.8)
        assert p.process("height_nullable", "5.0") == 3.0  # clamped


class TestBoolean:
    def test_true_variants(self, schema):
        p = ValuePostProcessor(schema)
        for raw in ("true", "True", "1", "yes", "Y"):
            assert p.process("active", raw) is True

    def test_false_variants(self, schema):
        p = ValuePostProcessor(schema)
        for raw in ("false", "False", "0", "no", "N"):
            assert p.process("active", raw) is False

    def test_unknown_raises(self, schema):
        p = ValuePostProcessor(schema)
        with pytest.raises(ValuePostProcessingError):
            p.process("active", "maybe")


def test_unknown_column_raises(schema):
    p = ValuePostProcessor(schema)
    with pytest.raises(ValuePostProcessingError):
        p.process("does_not_exist", "foo")
