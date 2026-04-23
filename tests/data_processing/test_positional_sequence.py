# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from nemo_safe_synthesizer.data_processing.record_utils import (
    POSITIONAL_EOR,
    POSITIONAL_SEP,
    extract_and_validate_positional_records,
    extract_records_from_positional_string,
    records_to_positional_sequence,
)
from nemo_safe_synthesizer.data_processing.value_post_processor import ValuePostProcessor


@pytest.fixture
def schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
            "intent": {"enum": ["play_music", "weather", "calendar"]},
            "note": {"type": "string", "maxLength": 100},
        },
        "required": ["age", "intent", "note"],
    }


@pytest.fixture
def column_order() -> list[str]:
    return ["age", "intent", "note"]


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"age": 42, "intent": "play_music", "note": "turn it up"},
            {"age": 30, "intent": "weather", "note": "what's the forecast"},
        ]
    )


class TestSerializer:
    def test_basic_serialization(self, df, column_order):
        out = records_to_positional_sequence(df, column_order)
        assert POSITIONAL_SEP in out
        assert POSITIONAL_EOR in out
        expected_row = f"42{POSITIONAL_SEP}play_music{POSITIONAL_SEP}turn it up{POSITIONAL_EOR}"
        assert expected_row in out

    def test_uses_explicit_column_order(self, df):
        reordered = ["intent", "age", "note"]
        out = records_to_positional_sequence(df, reordered)
        first_line = out.split("\n")[0]
        assert first_line.startswith("play_music" + POSITIONAL_SEP + "42")

    def test_handles_none_and_nan(self, column_order):
        records = [{"age": 42, "intent": None, "note": float("nan")}]
        out = records_to_positional_sequence(records, column_order)
        assert f"42{POSITIONAL_SEP}{POSITIONAL_SEP}{POSITIONAL_EOR}" in out

    def test_missing_column_raises(self, df):
        with pytest.raises(ValueError, match="missing columns"):
            records_to_positional_sequence(df, ["age", "intent", "note", "missing"])

    def test_empty_records(self, column_order):
        assert records_to_positional_sequence(pd.DataFrame(columns=column_order), column_order) == ""


class TestExtractor:
    def test_splits_on_eor(self):
        text = (
            f"42{POSITIONAL_SEP}play_music{POSITIONAL_SEP}hi{POSITIONAL_EOR}\n"
            f"30{POSITIONAL_SEP}weather{POSITIONAL_SEP}sunny{POSITIONAL_EOR}\n"
        )
        rows = extract_records_from_positional_string(text)
        assert len(rows) == 2

    def test_drops_truncated_trailing(self):
        text = (
            f"42{POSITIONAL_SEP}play_music{POSITIONAL_SEP}hi{POSITIONAL_EOR}\n"
            f"30{POSITIONAL_SEP}weather{POSITIONAL_SEP}sunny-but-truncated"
        )
        rows = extract_records_from_positional_string(text)
        assert len(rows) == 1

    def test_no_eor_returns_empty(self):
        assert extract_records_from_positional_string("just plain text") == []


class TestRoundTrip:
    def test_roundtrip_preserves_values(self, df, column_order, schema):
        serialized = records_to_positional_sequence(df, column_order)
        processor = ValuePostProcessor(schema)
        parsed = extract_and_validate_positional_records(serialized, processor, column_order)
        assert len(parsed.valid_records) == 2
        assert parsed.valid_records[0] == {"age": 42, "intent": "play_music", "note": "turn it up"}
        assert parsed.valid_records[1] == {"age": 30, "intent": "weather", "note": "what's the forecast"}
        assert parsed.invalid_records == []

    def test_column_count_mismatch_is_invalid(self, column_order, schema):
        text = f"42{POSITIONAL_SEP}play_music{POSITIONAL_EOR}\n"  # missing note
        processor = ValuePostProcessor(schema)
        parsed = extract_and_validate_positional_records(text, processor, column_order)
        assert parsed.valid_records == []
        assert len(parsed.invalid_records) == 1
        assert parsed.errors[0][1] == "Positional"
        assert "expected 3 values" in parsed.errors[0][0]

    def test_out_of_range_numeric_clamps(self, column_order, schema):
        text = f"200{POSITIONAL_SEP}weather{POSITIONAL_SEP}note{POSITIONAL_EOR}\n"
        processor = ValuePostProcessor(schema)
        parsed = extract_and_validate_positional_records(text, processor, column_order)
        assert parsed.valid_records == [{"age": 120, "intent": "weather", "note": "note"}]

    def test_enum_miss_drops_row(self, column_order, schema):
        text = f"42{POSITIONAL_SEP}nonsense-intent{POSITIONAL_SEP}note{POSITIONAL_EOR}\n"
        processor = ValuePostProcessor(schema)
        parsed = extract_and_validate_positional_records(text, processor, column_order)
        assert parsed.valid_records == []
        assert len(parsed.invalid_records) == 1
        assert "intent" in parsed.errors[0][0]

    def test_mixed_valid_and_invalid(self, column_order, schema):
        text = (
            f"42{POSITIONAL_SEP}play_music{POSITIONAL_SEP}ok{POSITIONAL_EOR}\n"
            f"not-a-number{POSITIONAL_SEP}weather{POSITIONAL_SEP}bad{POSITIONAL_EOR}\n"
            f"60{POSITIONAL_SEP}calendar{POSITIONAL_SEP}ok2{POSITIONAL_EOR}\n"
        )
        processor = ValuePostProcessor(schema)
        parsed = extract_and_validate_positional_records(text, processor, column_order)
        assert len(parsed.valid_records) == 2
        assert len(parsed.invalid_records) == 1
        assert parsed.valid_records[0]["age"] == 42
        assert parsed.valid_records[1]["age"] == 60
