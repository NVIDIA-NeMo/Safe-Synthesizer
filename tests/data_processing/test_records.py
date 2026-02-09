# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import sys
from io import BytesIO

import numpy as np
import pandas as pd
from nemo_safe_synthesizer.data_processing.record_utils import (
    _extract_timestamp_seconds,
    _validate_time_interval,
    check_record_for_large_numbers,
    extract_and_validate_records,
    extract_and_validate_timeseries_records,
    is_safe_for_float_conversion,
    normalize_dataframe,
)


def test_is_safe_for_float_conversion():
    # Test with safe values
    assert is_safe_for_float_conversion(100) is True
    assert is_safe_for_float_conversion(100.0) is True
    assert is_safe_for_float_conversion(-100) is True
    assert is_safe_for_float_conversion(0) is True

    # Test with values at the edge of float64
    max_safe = sys.float_info.max
    assert is_safe_for_float_conversion(max_safe) is True
    assert is_safe_for_float_conversion(-max_safe) is True

    # Test with unsafe values
    unsafe_value = 10**309  # This is way beyond float64's capacity
    assert is_safe_for_float_conversion(unsafe_value) is False
    assert is_safe_for_float_conversion(-unsafe_value) is False

    # Test with non-numeric values
    assert is_safe_for_float_conversion("100") is True  # strings are considered safe
    assert is_safe_for_float_conversion(None) is True
    assert is_safe_for_float_conversion([]) is True
    assert is_safe_for_float_conversion({}) is True


def test_check_record_for_large_numbers():
    # Test with safe records
    safe_record = {"id": 100, "name": "test", "value": 100.0, "empty": None}
    errors = check_record_for_large_numbers(safe_record)

    assert errors is None

    # Test with unsafe records
    unsafe_record = {
        "id": 10**309,  # too large
        "name": "test",
        "value": 100.0,
        "another_large": 2**1000,  # also too large
    }
    errors = check_record_for_large_numbers(unsafe_record)

    assert isinstance(errors, str)
    assert "id" in errors

    # Test with mixed safe and unsafe values
    mixed_record = {"safe": 100, "unsafe": 10**309, "string": "test", "none": None}
    errors = check_record_for_large_numbers(mixed_record)
    assert isinstance(errors, str)
    assert "unsafe" in errors

    # Test with empty record
    empty_record = {}
    errors = check_record_for_large_numbers(empty_record)
    assert errors is None

    # Test with edge case values
    edge_case_record = {
        "max_safe": sys.float_info.max,
        "min_safe": -sys.float_info.max,
        "zero": 0,
    }
    errors = check_record_for_large_numbers(edge_case_record)
    assert errors is None


def test_extract_and_validate_records_with_invalid_records_with_large_numbers(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    valid_jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema

    # intentionally add a corrupt record with a large number
    corrupt_record = {
        "sepal.length": 10**309,
        "sepal.width": 3.6,
        "petal.length": 1.4,
        "petal.width": 0.2,
        "variety": "Setosa",
    }
    # set the maximum to bypass the schema check
    jsonl_schema["properties"]["sepal.length"]["maximum"] = 10**309
    corrupt_record_str = json.dumps(corrupt_record)
    valid_jsonl_str += f"""{corrupt_record_str}"""
    valid, invalid, errors = extract_and_validate_records(valid_jsonl_str, schema=jsonl_schema)
    assert len(valid) == 5
    assert len(invalid) == 1
    assert len(errors) == 1
    assert len(errors[0]) == 2
    assert "is too large to convert to float64" in errors[0][0]
    assert errors[0][1] == "Float Conversion"


def test_extract_and_validate_records(fixture_valid_iris_dataset_jsonl_and_schema):
    valid_jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    valid, invalid, errors = extract_and_validate_records(valid_jsonl_str, schema=jsonl_schema)
    assert len(valid) == 5
    assert len(invalid) == 0
    assert len(errors) == 0


def test_extract_and_validate_records_with_invalid_records(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    valid_jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema

    # intentionally corrupt the last record with invalid text that would lead to a json decode error
    valid_jsonl_str = valid_jsonl_str[:-2] + "invalid entry }"
    valid, invalid, errors = extract_and_validate_records(valid_jsonl_str, schema=jsonl_schema)
    assert len(valid) == 4
    assert len(invalid) == 1
    assert len(errors) == 1
    assert len(errors[0]) == 2
    assert errors[0][0] == "Invalid JSON: Expecting ',' delimiter"
    assert errors[0][1] == "Invalid JSON"


def test_extract_and_validate_records_with_invalid_schema(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    valid_jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema

    # tack on a new json obj with a different schema than in the spec
    valid_jsonl_str = valid_jsonl_str + f"{json.dumps(dict(a=1, b=2))}"
    valid, invalid, errors = extract_and_validate_records(valid_jsonl_str, schema=jsonl_schema)
    assert len(valid) == 5
    assert len(invalid) == 1
    assert len(errors) == 1
    assert len(errors[0]) == 2
    assert errors[0][0] == "'sepal.length' is a required property"
    assert errors[0][1] == "required"


def test_extract_and_validate_records_with_invalid_utf_characters_embedded(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    valid_jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema

    # Modify the valid jsonl str sample to have valid and invalid
    # utf-8 charactres. Existence of these should not have an effect on the validity
    # of these json objects.
    existing_petal_variety = "Setosa"
    new_petal_variety_with_invalid_utf8 = f"\udfff{existing_petal_variety}\U0001f600"
    valid_jsonl_str = valid_jsonl_str.replace(existing_petal_variety, new_petal_variety_with_invalid_utf8)
    jsonl_schema["properties"]["variety"]["enum"].append(new_petal_variety_with_invalid_utf8)

    valid, invalid, errors = extract_and_validate_records(valid_jsonl_str, schema=jsonl_schema)
    assert len(valid) == 5
    assert len(invalid) == 0
    assert len(errors) == 0


def test_extract_and_validate_records_with_non_english_characters(fixture_lmsys_dataset_jsonl_and_schema):
    valid_jsonl_str, jsonl_schema = fixture_lmsys_dataset_jsonl_and_schema
    valid, invalid, errors = extract_and_validate_records(valid_jsonl_str, schema=jsonl_schema)
    assert len(valid) == 5
    assert len(invalid) == 0
    assert len(errors) == 0

    # Check that the non-English content is preserved correctly
    assert valid[0]["conversation"][0]["content"] == "ПРИВЕТ"
    assert valid[1]["conversation"][0]["content"] == "一家4000人的化工企业需要配备几名安全员"
    assert valid[2]["conversation"][0]["content"][:10] == "경찰 관계자는 “범"
    assert valid[3]["conversation"][0]["content"] == "نعم انا خبير في ادارة تطبيقات وحسابات التواصل الاجتماعي بحترافيه"
    assert valid[4]["conversation"][0]["content"][41:74] == "\"Ім'я і національність\n@Ukrostap\n"


def _run_csv_writer(df: pd.DataFrame):
    """Runs the MIF model csv writer.

    Any normalized DataFrame should successfully write without errors.
    """

    output = BytesIO()
    df.to_csv(output, index=False)


def test_normalize_dataframe_invalid_unicode():
    # LLMs will sometimes produce invalid unicode or surrogates, we want to make
    # sure we don't crash in these situations. Our approach is just to drop the
    # bad unicode.
    text = (
        "text with valid unicode literals like \U0001f600, \U00000021, \u20ac and invalid ones like this\udfff, \ud83c"
    )
    expected_clean_text = "text with valid unicode literals like 😀, !, € and invalid ones like this, "
    text_in_json_str = f'{{"some_key": "{text}"}}'
    stub_df = pd.DataFrame(
        [
            dict(text="just some stream of text", label=np.nan),
            dict(text=text, label="good"),
            dict(text=text_in_json_str, label="average"),
        ]
    )

    normalized_df = normalize_dataframe(stub_df)
    assert math.isnan(normalized_df.iloc[0]["label"])
    assert normalized_df.iloc[0]["text"] == "just some stream of text"
    assert normalized_df.iloc[1]["label"] == "good"
    assert normalized_df.iloc[1]["text"] == expected_clean_text
    assert normalized_df.iloc[2]["text"] == f'{{"some_key": "{expected_clean_text}"}}'

    # Ensure the normalized DataFrame writes successfully using our csv writer.
    _run_csv_writer(normalized_df)


def test_normalize_dataframe_invalid_unicode_pandas_na():
    # LLMs will sometimes produce invalid unicode or surrogates, we want to make
    # sure we don't crash in these situations. Our approach is just to drop the
    # bad unicode.
    text = (
        "text with valid unicode literals like \U0001f600, \U00000021, \u20ac and invalid ones like this\udfff, \ud83c"
    )
    expected_clean_text = "text with valid unicode literals like 😀, !, € and invalid ones like this, "
    text_in_json_str = f'{{"some_key": "{text}"}}'
    stub_df = pd.DataFrame(
        [
            dict(text="just some stream of text", label=pd.NA),
            dict(text=text, label="good"),
            dict(text=text_in_json_str, label="average"),
        ]
    )

    normalized_df = normalize_dataframe(stub_df)
    assert math.isnan(normalized_df.iloc[0]["label"])
    assert normalized_df.iloc[0]["text"] == "just some stream of text"
    assert normalized_df.iloc[1]["label"] == "good"
    assert normalized_df.iloc[1]["text"] == expected_clean_text
    assert normalized_df.iloc[2]["text"] == f'{{"some_key": "{expected_clean_text}"}}'

    # Ensure the normalized DataFrame writes successfully using our csv writer.
    _run_csv_writer(normalized_df)


def test_normalize_dataframe_carriage_return():
    # Embedded carriage returns, newlines, and combinations should be preserved.
    df = pd.DataFrame(
        [
            dict(text="regular string", count=105, label=np.nan),
            dict(text="embedded\nnewline", count=10, label="newline"),
            dict(text="""embedded\rcarriage return""", count=3, label="carriage return"),
            dict(
                text="embedded\\rescaped carriage return",
                count=1,
                label="carriage return",
            ),
            dict(text="embedded\r\ncarriage return and newline", count=8, label="both"),
        ]
    )
    normalized_df = normalize_dataframe(df)
    pd.testing.assert_frame_equal(df, normalized_df)

    # Ensure the normalized DataFrame writes successfully using our csv writer.
    _run_csv_writer(normalized_df)


def test_normalize_dataframe_carriage_return_repro(
    fixture_embeddd_carriage_return_dataframe,
):
    # In RDS-1082, we observed prod and dev failures related to embedded
    # carriage returns, but resulting in a different error (possible malformed
    # input file, versus wrong sized DataFrame) than the manually created
    # DataFrame in test_normalize_dataframe_carriage_return. Here we directly
    # test a minimal DataFrame from one such failure. The difference is probably
    # around dtypes and how the DataFrames used in NavFT are created via json,
    # instead of from csv or manually.
    normalized_df = normalize_dataframe(fixture_embeddd_carriage_return_dataframe)

    # The normalization changes some of the dtypes in this example, so we don't
    # check for equality before and after. We just want to make sure the
    # DataFrame writes successfully.
    _run_csv_writer(normalized_df)


def test_extract_timestamp_seconds():
    """Test _extract_timestamp_seconds with datetime, elapsed, missing, and invalid cases."""
    # Datetime format
    record_dt = {"timestamp": "2024-01-01 12:00:00", "value": 1}
    result, error = _extract_timestamp_seconds(record_dt, "timestamp", "%Y-%m-%d %H:%M:%S")
    assert error is None
    assert result is not None
    assert isinstance(result, int)

    # Elapsed seconds format
    record_elapsed = {"elapsed_seconds": 3600, "value": 1}
    result, error = _extract_timestamp_seconds(record_elapsed, "elapsed_seconds", "elapsed_seconds")
    assert error is None
    assert result == 3600

    # Missing timestamp column
    record_missing = {"value": 1}
    result, error = _extract_timestamp_seconds(record_missing, "timestamp", "%Y-%m-%d %H:%M:%S")
    assert result is None
    assert error is not None
    assert "Missing 'timestamp'" in error[0]
    assert error[1] == "TimeSeries"

    # Invalid timestamp value
    record_invalid = {"timestamp": "not-a-timestamp", "value": 1}
    result, error = _extract_timestamp_seconds(record_invalid, "timestamp", "%Y-%m-%d %H:%M:%S")
    assert result is None
    assert error is not None
    assert "Invalid 'timestamp'" in error[0]
    assert error[1] == "TimeSeries"


def test_validate_time_interval_cases():
    """Test _validate_time_interval in multiple cases: first record, valid, invalid, day rollover."""

    # First record (no previous timestamp)
    result_seconds, result_offset, error = _validate_time_interval(
        timestamp_seconds=0,
        last_absolute_seconds=None,
        day_offset=0,
        interval_seconds=3600,
        time_column="timestamp",
        allow_rollover=True,
    )
    assert error is None
    assert result_seconds == 0
    assert result_offset == 0

    # Valid interval step
    result_seconds, result_offset, error = _validate_time_interval(
        timestamp_seconds=3600,
        last_absolute_seconds=0,
        day_offset=0,
        interval_seconds=3600,
        time_column="timestamp",
        allow_rollover=True,
    )
    assert error is None
    assert result_seconds == 3600

    # Invalid interval step
    result_seconds, result_offset, error = _validate_time_interval(
        timestamp_seconds=7200,  # 2 hours, but expected 1 hour
        last_absolute_seconds=0,
        day_offset=0,
        interval_seconds=3600,
        time_column="timestamp",
        allow_rollover=True,
    )
    assert error is not None
    assert "must advance in 3600 seconds increments" in error[0]
    assert error[1] == "TimeSeries"

    # Day rollover (23:00 -> 00:00 transition)
    # 23:00 = 82800 seconds, 00:00 = 0 seconds
    result_seconds, result_offset, error = _validate_time_interval(
        timestamp_seconds=0,  # 00:00
        last_absolute_seconds=82800,  # 23:00
        day_offset=0,
        interval_seconds=3600,
        time_column="timestamp",
        allow_rollover=True,
    )
    assert error is None
    assert result_offset == 86400
    assert result_seconds == 86400  # 0 + 86400


def test_extract_and_validate_timeseries_records_valid():
    """Test extracting valid series records."""
    schema = {
        "type": "object",
        "properties": {"timestamp": {"type": "string"}, "value": {"type": "integer"}},
        "required": ["timestamp", "value"],
    }
    jsonl = (
        '{"timestamp": "2024-01-01 00:00:00", "value": 1}\n'
        '{"timestamp": "2024-01-01 01:00:00", "value": 2}\n'
        '{"timestamp": "2024-01-01 02:00:00", "value": 3}\n'
    )

    valid, invalid, errors = extract_and_validate_timeseries_records(
        jsonl, schema, time_column="timestamp", interval_seconds=3600, time_format="%Y-%m-%d %H:%M:%S"
    )

    assert len(valid) == 3
    assert len(invalid) == 0
    assert len(errors) == 0


def test_extract_and_validate_timeseries_records_no_interval_validation():
    """Test that records pass when interval_seconds is None."""
    schema = {
        "type": "object",
        "properties": {"timestamp": {"type": "string"}, "value": {"type": "integer"}},
        "required": ["timestamp", "value"],
    }
    # Irregular intervals - would fail with interval validation
    jsonl = (
        '{"timestamp": "2024-01-01 00:00:00", "value": 1}\n'
        '{"timestamp": "2024-01-01 02:30:00", "value": 2}\n'
        '{"timestamp": "2024-01-01 05:00:00", "value": 3}\n'
    )

    valid, invalid, errors = extract_and_validate_timeseries_records(
        jsonl, schema, time_column="timestamp", interval_seconds=None, time_format="%Y-%m-%d %H:%M:%S"
    )

    assert len(valid) == 3
    assert len(invalid) == 0
    assert len(errors) == 0


def test_extract_and_validate_timeseries_records_invalid_interval():
    """Test that invalid interval marks remaining records as invalid."""
    schema = {
        "type": "object",
        "properties": {"timestamp": {"type": "string"}, "value": {"type": "integer"}},
        "required": ["timestamp", "value"],
    }
    # Second record skips an hour
    jsonl = (
        '{"timestamp": "2024-01-01 00:00:00", "value": 1}\n'
        '{"timestamp": "2024-01-01 02:00:00", "value": 2}\n'  # Should be 01:00:00
        '{"timestamp": "2024-01-01 03:00:00", "value": 3}\n'
    )

    valid, invalid, errors = extract_and_validate_timeseries_records(
        jsonl, schema, time_column="timestamp", interval_seconds=3600, time_format="%Y-%m-%d %H:%M:%S"
    )

    assert len(valid) == 1  # Only first record is valid
    assert len(invalid) == 2  # Second and third are invalid
    assert len(errors) == 2


def test_extract_and_validate_timeseries_records_invalid_json():
    """Test that invalid JSON stops validation."""
    schema = {
        "type": "object",
        "properties": {"timestamp": {"type": "string"}, "value": {"type": "integer"}},
    }
    jsonl = (
        '{"timestamp": "2024-01-01 00:00:00", "value": 1}\n'
        '{"timestamp": "2024-01-01 01:00:00", "value": invalid}\n'  # Invalid JSON
        '{"timestamp": "2024-01-01 02:00:00", "value": 3}\n'
    )

    valid, invalid, errors = extract_and_validate_timeseries_records(
        jsonl, schema, time_column="timestamp", interval_seconds=3600, time_format="%Y-%m-%d %H:%M:%S"
    )

    assert len(valid) == 1
    assert len(invalid) == 1
    assert len(errors) == 1
    assert "Invalid JSON" in errors[0][0]


def test_extract_and_validate_timeseries_records_elapsed_time():
    """Test with elapsed_seconds time format."""
    schema = {
        "type": "object",
        "properties": {"elapsed_seconds": {"type": "integer"}, "value": {"type": "integer"}},
        "required": ["elapsed_seconds", "value"],
    }
    jsonl = (
        '{"elapsed_seconds": 0, "value": 1}\n'
        '{"elapsed_seconds": 3600, "value": 2}\n'
        '{"elapsed_seconds": 7200, "value": 3}\n'
    )

    valid, invalid, errors = extract_and_validate_timeseries_records(
        jsonl, schema, time_column="elapsed_seconds", interval_seconds=3600, time_format="elapsed_seconds"
    )

    assert len(valid) == 3
    assert len(invalid) == 0
    assert len(errors) == 0


def test_extract_and_validate_timeseries_records_missing_timestamp():
    """Test that missing timestamp column stops validation."""
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    jsonl = '{"value": 1}\n{"value": 2}\n'

    valid, invalid, errors = extract_and_validate_timeseries_records(
        jsonl, schema, time_column="timestamp", interval_seconds=3600, time_format="%Y-%m-%d %H:%M:%S"
    )

    assert len(valid) == 0
    assert len(invalid) == 1
    assert len(errors) == 1
    assert "Missing 'timestamp'" in errors[0][0]
