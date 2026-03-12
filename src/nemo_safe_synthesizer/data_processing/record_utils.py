# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for extracting, validating, and converting JSONL records.

Provides regex-based JSONL extraction, JSON-schema validation (including
time-series interval checks), DataFrame normalization, and JSONL serialization.
"""

from __future__ import annotations

import calendar
import json
import re
from csv import QUOTE_NONNUMERIC
from datetime import datetime
from io import StringIO

import jsonschema
import pandas as pd

from ..observability import get_logger

RECORD_REGEX_PATTERN = r"{.+?}(?:\n|$)"
RECORD_REGEX_PATTEN_LOOKAHEAD = r"{.+?}(?=\n|$)"

logger = get_logger()


def is_safe_for_float_conversion(value: str | int | float | None | list | dict) -> bool:
    """Check if a value can be safely converted to float64 without overflow.

    Only ``int`` values can cause overflow; all other types are considered safe.

    Args:
        value: The value to check.

    Returns:
        True if the value can be safely converted to float64, False otherwise.
    """
    # not considering Decimal because the input of this validation
    # is coming from converting a jsonl string to JSON object.
    # JSON object only supports int or float for numeric numbers

    # only int could have overflow error
    if isinstance(value, int):
        try:
            float(value)
            return True
        except (OverflowError, ValueError):
            return False
    return True


def check_record_for_large_numbers(record: dict) -> str | None:
    """Check if a record contains any numbers that would cause float64 overflow.

    Args:
        record: Dictionary of field names to values.

    Returns:
        An error message describing the first unsafe value found,
        or None if all values are safe.
    """
    for key, value in record.items():
        if not is_safe_for_float_conversion(value):
            # If a column contains a value that is too large to convert to float64,
            # then the entire record is invalid
            return f"Value {value} in field '{key}' is too large to convert to float64"

    return None


def check_if_records_are_ordered(records: list[dict], order_by: str) -> bool:
    """Check if the records are in ascending order based on the given `order_by` column.

    Args:
        records: List of of JSONL records.
        order_by: Column to check for ordering.

    Returns:
        True if the records are ordered by the given column, otherwise False.
    """
    order_by_values = [rec[order_by] for rec in records]
    sorted_values = sorted([rec[order_by] for rec in records])
    return order_by_values == sorted_values


def extract_records_from_jsonl_string(jsonl_string: str) -> list[str]:
    """Extract and return tabular records from the given JSONL string."""
    return re.findall(RECORD_REGEX_PATTEN_LOOKAHEAD, jsonl_string)


def extract_groups_from_jsonl_string(jsonl_string: str, bos: str, eos: str) -> list[str]:
    """Extract groups of records from the given JSONL string.

    This function assumes that the complete group of records
    is enclosed by the given beginning-of-sequence (bos) and
    end-of-sequence (eos) tokens.

    Args:
        jsonl_string: Single JSONL string containing grouped tabular records.
        bos: Beginning-of-sequence token used to identify the start of a group.
        eos: End-of-sequence token used to identify the end of a group.

    Returns:
        Substrings matching complete bos/eos-delimited record groups.
    """
    bos_re = re.escape(rf"{bos}")
    eos_re = re.escape(rf"{eos}")
    return re.findall(rf"{bos_re}\s?(?:{RECORD_REGEX_PATTERN}\s?)+\s?{eos_re}", jsonl_string)


def extract_and_validate_records(
    jsonl_string: str, schema: dict
) -> tuple[list[dict], list[str], list[tuple[str, str]]]:
    """Extract and validate records from the given JSONL string.

    The records are validated against the given schema using jsonschema.

    Args:
        jsonl_string: Single JSONL string containing tabular records.
        schema: JSON schema as a dictionary.

    Returns:
        valid_records (list[dict]): List of valid records.
        invalid_records (list[str]): List of invalid records.
        invalid_record_errors (list[tuple[str, str]]): List of errors for invalid records, each a (message, validator) tuple.
    """
    valid_records = []
    invalid_records = []
    invalid_record_errors = []

    for matched_json in extract_records_from_jsonl_string(jsonl_string):
        matched_dict, error = _parse_and_validate_json(matched_json, schema)
        if error:
            invalid_records.append(matched_json)
            invalid_record_errors.append(error)
        else:
            valid_records.append(matched_dict)

    return valid_records, invalid_records, invalid_record_errors


def _parse_timestamp_to_seconds(value: object, time_format: str) -> int:
    """Convert a timestamp value to seconds based on the specified format.

    Args:
        value: The timestamp value (can be string, int, or float depending on format).
        time_format: The format of the timestamp. Special value "elapsed_seconds" means
                     the value is already in seconds. Otherwise, it's a strptime format string.

    Returns:
        The timestamp converted to seconds (either elapsed seconds or epoch seconds).

    Raises:
        ValueError: If the timestamp cannot be parsed with the given format.
    """
    if time_format == "elapsed_seconds":
        # Value is already in seconds (int for now and float for future)
        return int(float(value))

    # Parse using strptime format
    dt = datetime.strptime(str(value), time_format)

    # If the format includes date components, return epoch seconds.
    # Otherwise, return seconds since midnight (for time-only formats).
    date_tokens = ("%Y", "%y", "%m", "%b", "%B", "%d", "%j", "%U", "%W", "%V", "%x", "%c")
    has_date = any(tok in time_format for tok in date_tokens)
    if has_date:
        # Honor timezone if present; otherwise treat naive datetime as UTC.
        if dt.tzinfo is not None:
            return int(dt.timestamp())
        return calendar.timegm(dt.timetuple())
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def _parse_and_validate_json(matched_json: str, schema: dict) -> tuple[dict | None, tuple[str, str] | None]:
    """Parse JSON string and validate against schema.

    Args:
        matched_json: JSON string to parse.
        schema: JSON schema for validation.

    Returns:
        Tuple of (parsed_dict, error). If successful, error is None.
        If failed, parsed_dict is None and error is (message, validator).
    """
    try:
        matched_dict = json.loads(matched_json)
        jsonschema.validate(matched_dict, schema)

        error_msg = check_record_for_large_numbers(matched_dict)
        if error_msg:
            return None, (error_msg, "Float Conversion")

        return matched_dict, None

    except json.JSONDecodeError as err:
        return None, (f"Invalid JSON: {err.msg}", "Invalid JSON")
    except jsonschema.exceptions.ValidationError as err:
        return None, (err.message, err.validator)


def _extract_timestamp_seconds(
    record: dict, time_column: str, time_format: str
) -> tuple[int | None, tuple[str, str] | None]:
    """Extract and parse timestamp from a record.

    Args:
        record: The record dict.
        time_column: Column containing the timestamp.
        time_format: Format of the timestamp.

    Returns:
        Tuple of (timestamp_seconds, error). If successful, error is None.
        If failed, timestamp_seconds is None and error is (message, validator).
    """
    timestamp_value = record.get(time_column)
    if timestamp_value is None:
        return None, (f"Missing '{time_column}' required for interval validation", "TimeSeries")

    try:
        timestamp_seconds = _parse_timestamp_to_seconds(timestamp_value, time_format)
        return timestamp_seconds, None
    except (ValueError, TypeError) as e:
        return None, (f"Invalid '{time_column}' value '{timestamp_value}': {e}", "TimeSeries")


def _validate_time_interval(
    timestamp_seconds: int,
    last_absolute_seconds: int | None,
    day_offset: int,
    interval_seconds: int,
    time_column: str,
    allow_rollover: bool,
) -> tuple[int, int, tuple[str, str] | None]:
    """Validate time interval between consecutive records.

    Handles day rollover for time-only formats (e.g., %H:%M:%S) where
    _parse_timestamp_to_seconds returns seconds-since-midnight (0-86399).
    If data crosses midnight (23:00 -> 00:00), raw seconds go from 82800 to 0.
    The day_offset mechanism adds 86400 to keep values monotonically increasing.

    For formats with date components, _parse_timestamp_to_seconds returns epoch
    seconds which are already monotonic, so day_offset stays 0 and rollover is disabled.

    See test_validate_time_interval_cases for examples of expected behavior.

    Args:
        timestamp_seconds: Current timestamp in seconds (from _parse_timestamp_to_seconds).
        last_absolute_seconds: Previous absolute timestamp (with day offset applied).
        day_offset: Current day offset in seconds (multiples of 86400) for handling midnight rollovers.
        interval_seconds: Expected interval between timestamps.
        time_column: Name of time column (for error messages).
        allow_rollover: Whether to allow midnight rollover (True for time-only formats).

    Returns:
        Tuple of (new_absolute_seconds, new_day_offset, error).
        If validation passes, error is None.
    """
    absolute_seconds = timestamp_seconds + day_offset

    if last_absolute_seconds is not None:
        if allow_rollover:
            # Handle day rollover for time-only formats (e.g., 23:00 -> 00:00)
            while absolute_seconds <= last_absolute_seconds:
                day_offset += 24 * 60 * 60
                absolute_seconds = timestamp_seconds + day_offset
        else:
            # For date-inclusive formats, timestamps must be strictly increasing
            if absolute_seconds <= last_absolute_seconds:
                return (
                    absolute_seconds,
                    day_offset,
                    (
                        f"'{time_column}' must be strictly increasing",
                        "TimeSeries",
                    ),
                )

        if absolute_seconds - last_absolute_seconds != interval_seconds:
            error = (
                f"'{time_column}' must advance in {interval_seconds} seconds increments with no gaps",
                "TimeSeries",
            )
            return absolute_seconds, day_offset, error

    return absolute_seconds, day_offset, None


def extract_and_validate_timeseries_records(
    jsonl_string: str,
    schema: dict,
    time_column: str,
    interval_seconds: int | None,
    time_format: str,
) -> tuple[list[dict], list[str], list[tuple[str, str]]]:
    """Extract and validate sequential records with enforced time interval constraints.

    Args:
        jsonl_string: JSONL string containing series data.
        schema: JSON schema describing the records.
        time_column: Column containing the timestamp used for interval validation.
        interval_seconds: (Optional) Expected interval in seconds between consecutive
            timestamps. If not provided, no time interval validation is performed.
        time_format: Format of the timestamp column (required, should be set from config).

    Returns:
        Tuple of valid records, invalid record strings, and their associated errors.
    """
    valid_records: list[dict] = []
    invalid_records: list[str] = []
    invalid_record_errors: list[tuple[str, str]] = []

    last_absolute_seconds: int | None = None
    day_offset = 0

    # Allow rollover only for time-only formats (no date components)
    # If time_format is "elapsed_seconds", treat as time-only (allow rollover)
    date_tokens = ("%Y", "%y", "%m", "%b", "%B", "%d", "%j", "%U", "%W", "%V", "%x", "%c")
    if time_format == "elapsed_seconds":
        allow_rollover = True
    else:
        has_date = any(tok in time_format for tok in date_tokens)
        allow_rollover = not has_date

    all_json_records = list(extract_records_from_jsonl_string(jsonl_string))

    for idx, matched_json in enumerate(all_json_records):
        # Step 1: Parse and validate JSON/schema
        matched_dict, error = _parse_and_validate_json(matched_json, schema)
        if error or matched_dict is None:
            invalid_records.append(matched_json)
            if error:
                invalid_record_errors.append(error)
            break

        # Step 2: Extract and parse timestamp
        timestamp_seconds, error = _extract_timestamp_seconds(matched_dict, time_column, time_format)
        if error or timestamp_seconds is None:
            invalid_records.append(matched_json)
            if error:
                invalid_record_errors.append(error)
            break

        # Step 3: Validate time interval (if interval_seconds is specified)
        if interval_seconds is not None:
            absolute_seconds, day_offset, error = _validate_time_interval(
                timestamp_seconds,
                last_absolute_seconds,
                day_offset,
                interval_seconds,
                time_column,
                allow_rollover,
            )
            if error:
                # Mark current record with the specific error, and remaining records with cascade error
                invalid_records.append(matched_json)
                invalid_record_errors.append(error)
                # Mark remaining records (after current) as invalid due to previous error
                remaining_records = all_json_records[idx + 1 :]
                cascade_error = ("Invalid due to previous record error", "TimeSeries")
                invalid_records.extend(remaining_records)
                invalid_record_errors.extend([cascade_error] * len(remaining_records))
                break
            last_absolute_seconds = absolute_seconds

        valid_records.append(matched_dict)

    return valid_records, invalid_records, invalid_record_errors


def normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame of generated records via a CSV round-trip.

    Serializes to CSV and reads back to standardize missing-value
    representations (NaN/None/NA) across mixed-type columns. Falls back
    to ignoring encoding errors if the initial round-trip fails.

    Args:
        dataframe: DataFrame to normalize.

    Returns:
        DataFrame with missing values normalized and invalid UTF-8 characters
        dropped.
    """
    # HACK: Handle NaN/None/NA values with mixed types by
    # normalizing through pandas csv io format, which will match
    # the format in reports generated via the gretel client.
    try:
        # try without trying to resolve utf-8 issues first
        return pd.read_csv(StringIO(dataframe.to_csv(index=False, quoting=QUOTE_NONNUMERIC)))
    except Exception as exc_info:
        msg = (
            "An exception was raised while normalizing the pandas dataframe with records generated for Safe Synth. "
            "Retrying with flags to ignore encoding errors."
        )
        logger.error(msg, exc_info=exc_info)
        return pd.read_csv(
            StringIO(dataframe.to_csv(index=False, quoting=QUOTE_NONNUMERIC)),
            encoding="utf-8",
            encoding_errors="ignore",
        )


def records_to_jsonl(records: pd.DataFrame | list[dict] | dict) -> str:
    """Convert list of records to a JSONL string.

    Args:
        records: DataFrame, list of records, or dict.

    Returns:
        The JSONL string.
    """
    if isinstance(records, pd.DataFrame):
        return records.to_json(orient="records", lines=True, force_ascii=False)
    elif isinstance(records, (list, dict)):
        return pd.DataFrame(records).to_json(orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported type: {type(records)}")
