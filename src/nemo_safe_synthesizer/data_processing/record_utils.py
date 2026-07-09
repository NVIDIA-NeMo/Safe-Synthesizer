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
import time
from collections.abc import Callable, Mapping, Sequence
from csv import QUOTE_NONNUMERIC
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from typing import Any

import jsonschema
import pandas as pd

from ..observability import get_logger
from .records.json_types import JsonSchema, JsonValue, is_json_object

RECORD_REGEX_PATTERN = r"{.+?}(?:\n|$)"
RECORD_REGEX_PATTEN_LOOKAHEAD = r"{.+?}(?=\n|$)"

logger = get_logger()

RecordDict = dict[str, Any]
RecordMapping = Mapping[str, Any]
RawRecordMapping = Mapping[Any, Any]


@dataclass
class ParsedRecord:
    """A single record extracted from an LLM completion.

    Validity is tracked by the invariant that exactly one of ``parsed``
    and ``error`` is non-``None``: a valid record has ``parsed`` set and
    ``error`` as ``None``, an invalid record has ``error`` set and
    ``parsed`` as ``None``.
    [`is_valid`][nemo_safe_synthesizer.data_processing.record_utils.ParsedRecord.is_valid]
    is the canonical accessor.

    ``text`` and ``token_count`` are captured at extraction time and
    remain invariant even if the record is reclassified later (e.g. by
    group-level checks or data-fidelity filters) via
    [`invalidate`][nemo_safe_synthesizer.data_processing.record_utils.ParsedRecord.invalidate].
    """

    text: str
    """Original regex-matched JSON string (invariant under reclassification)."""

    parsed: RecordDict | None = None
    """Parsed dict when validation succeeded, ``None`` when invalid."""

    error: tuple[str, str] | None = None
    """``(detailed_msg, validator)`` when invalid, ``None`` when valid."""

    token_count: int = 0
    """Number of tokens in ``text``; 0 when no tokenizer was provided."""

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when this record passed validation."""
        return self.error is None

    def invalidate(self, error: tuple[str, str]) -> None:
        """Reclassify this record as invalid.

        ``text`` and ``token_count`` are kept intact; ``parsed`` is
        cleared so downstream consumers don't accidentally use a stale
        dict.

        Args:
            error: ``(detailed_msg, validator)`` tuple describing the
                reason for invalidation.
        """
        self.error = error
        self.parsed = None


@dataclass
class ParsedResponse:
    """Parsed result of a single LLM prompt response.

    Holds a flat list of
    [`ParsedRecord`][nemo_safe_synthesizer.data_processing.record_utils.ParsedRecord]
    objects (in input order) plus aggregated tokenization timing.
    ``valid_records`` / ``invalid_records`` / ``errors`` are convenience
    views that project the record list into the shapes expected by
    downstream aggregation code (parsed dicts, original text,
    ``(msg, validator)`` tuples respectively).
    """

    records: list[ParsedRecord] = field(default_factory=list)
    """Per-record extraction + validation outcomes, in input order."""

    tokenization_time_sec: float = 0.0
    """Wall-clock seconds spent tokenizing records in this response."""

    prompt_number: int | None = None
    """Index of the prompt within the batch (set by the processor call)."""

    @property
    def valid_records(self) -> list[RecordDict]:
        """Parsed dicts for records that passed validation."""
        return [r.parsed for r in self.records if r.is_valid and r.parsed is not None]

    @property
    def invalid_records(self) -> list[str]:
        """Original text for records that failed validation."""
        return [r.text for r in self.records if not r.is_valid]

    @property
    def errors(self) -> list[tuple[str, str]]:
        """``(detailed_msg, validator)`` tuples for each invalid record."""
        return [r.error for r in self.records if r.error is not None]


def is_safe_for_float_conversion(value: JsonValue) -> bool:
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


def check_record_for_large_numbers(record: RecordMapping) -> str | None:
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


def check_if_records_are_ordered(records: Sequence[RecordMapping], order_by: str) -> bool:
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


def normalize_record_keys(record: RawRecordMapping) -> RecordDict:
    """Return a record with string keys, matching JSON object semantics."""
    return {str(key): value for key, value in record.items()}


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


def timed_encode(
    encode: Callable[[str], list[int]] | None,
) -> Callable[[str], tuple[int, float]]:
    """Wrap an encode callable with timing, or return a no-op.

    Returns a function ``timed(text)`` that returns ``(n_tokens,
    elapsed_seconds)``.  When *encode* is ``None`` the returned
    function always returns ``(0, 0.0)``.
    """
    if encode is None:

        def _noop(_text: str) -> tuple[int, float]:
            return 0, 0.0

        return _noop

    def _timed(text: str) -> tuple[int, float]:
        t0 = time.monotonic()
        n = len(encode(text))
        return n, time.monotonic() - t0

    return _timed


def extract_and_validate_records(
    jsonl_string: str,
    schema: JsonSchema,
    encode: Callable[[str], list[int]] | None = None,
) -> ParsedResponse:
    """Extract and validate records from the given JSONL string.

    Each regex-matched JSON string is tokenized (when *encode* is
    provided) before validation so that exact token counts are
    available for every record regardless of later reclassification.

    Args:
        jsonl_string: Single JSONL string containing tabular records.
        schema: JSON schema as a dictionary.
        encode: Optional tokenizer encode callable.  When provided,
            each matched record string is tokenized and its token count
            is stored on the corresponding
            [`ParsedRecord`][nemo_safe_synthesizer.data_processing.record_utils.ParsedRecord].

    Returns:
        A
        [`ParsedResponse`][nemo_safe_synthesizer.data_processing.record_utils.ParsedResponse]
        whose ``records`` list is in input order, with ``parsed`` set
        for valid records and ``error`` set for invalid ones.
    """
    records: list[ParsedRecord] = []
    tokenization_time = 0.0
    timed = timed_encode(encode)

    for matched_json in extract_records_from_jsonl_string(jsonl_string):
        n_tokens, dt = timed(matched_json)
        tokenization_time += dt

        parsed, error = _parse_and_validate_json(matched_json, schema)
        records.append(ParsedRecord(text=matched_json, parsed=parsed, error=error, token_count=n_tokens))

    return ParsedResponse(records=records, tokenization_time_sec=tokenization_time)


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
        return int(float(str(value)))

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


def _parse_and_validate_json(matched_json: str, schema: JsonSchema) -> tuple[RecordDict | None, tuple[str, str] | None]:
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
        if not isinstance(matched_dict, dict):
            return None, ("Expected a JSON object", "Invalid JSON type")
        if not is_json_object(matched_dict):
            return None, ("Object contains a value that is not valid JSON", "Invalid JSON value")

        jsonschema.validate(matched_dict, schema)

        error_msg = check_record_for_large_numbers(matched_dict)
        if error_msg:
            return None, (error_msg, "Float Conversion")

        return matched_dict, None

    except json.JSONDecodeError as err:
        return None, (f"Invalid JSON: {err.msg}", "Invalid JSON")
    except jsonschema.exceptions.ValidationError as err:
        return None, (err.message, str(err.validator))


def _extract_timestamp_seconds(
    record: RecordMapping, time_column: str, time_format: str
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
    schema: JsonSchema,
    time_column: str,
    interval_seconds: int | None,
    time_format: str,
    encode: Callable[[str], list[int]] | None = None,
) -> ParsedResponse:
    """Extract and validate sequential records with time-interval constraints.

    Each regex-matched JSON string is tokenized (when *encode* is
    provided) before validation so that exact token counts are captured
    for both validated and cascade-invalidated records.

    Args:
        jsonl_string: JSONL string containing series data.
        schema: JSON schema describing the records.
        time_column: Column containing the timestamp used for interval
            validation.
        interval_seconds: Expected interval in seconds between
            consecutive timestamps.  When ``None``, no interval check
            is performed.
        time_format: Format of the timestamp column (required).
        encode: Optional tokenizer encode callable.  When provided,
            each matched record string is tokenized and its token count
            is stored on the corresponding
            [`ParsedRecord`][nemo_safe_synthesizer.data_processing.record_utils.ParsedRecord].

    Returns:
        A
        [`ParsedResponse`][nemo_safe_synthesizer.data_processing.record_utils.ParsedResponse]
        in input order. Once a record fails, every subsequent record is
        marked invalid with a cascade error.
    """
    records: list[ParsedRecord] = []
    tokenization_time = 0.0
    timed = timed_encode(encode)

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
    cascade_error = ("Invalid due to previous record error", "TimeSeries")

    for idx, matched_json in enumerate(all_json_records):
        n_tokens, dt = timed(matched_json)
        tokenization_time += dt

        # Step 1: Parse and validate JSON/schema.
        parsed, error = _parse_and_validate_json(matched_json, schema)
        if error or parsed is None:
            records.append(ParsedRecord(text=matched_json, error=error, token_count=n_tokens))
            # Parse/schema errors stop validation without cascading to later records.
            break

        # Step 2: Extract and parse timestamp.
        timestamp_seconds, error = _extract_timestamp_seconds(parsed, time_column, time_format)
        if error or timestamp_seconds is None:
            records.append(ParsedRecord(text=matched_json, error=error, token_count=n_tokens))
            # Missing timestamp stops validation without cascading to later records.
            break

        # Step 3: Validate time interval (if interval_seconds is specified).
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
                records.append(ParsedRecord(text=matched_json, error=error, token_count=n_tokens))
                # Interval errors cascade: mark all remaining records invalid so the
                # caller can report how many were affected.
                for remaining in all_json_records[idx + 1 :]:
                    rem_tokens, rem_dt = timed(remaining)
                    tokenization_time += rem_dt
                    records.append(ParsedRecord(text=remaining, error=cascade_error, token_count=rem_tokens))
                break
            last_absolute_seconds = absolute_seconds

        records.append(ParsedRecord(text=matched_json, parsed=parsed, token_count=n_tokens))

    return ParsedResponse(records=records, tokenization_time_sec=tokenization_time)


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
    # the format in reports generated via the nss client.
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


def records_to_jsonl(records: pd.DataFrame | list[RawRecordMapping] | RawRecordMapping) -> str:
    """Convert list of records to a JSONL string.

    Args:
        records: DataFrame, list of records, or dict.

    Returns:
        The JSONL string.
    """
    if isinstance(records, pd.DataFrame):
        return records.to_json(orient="records", lines=True, force_ascii=False)
    if isinstance(records, list | Mapping):
        return pd.DataFrame(records).to_json(orient="records", lines=True, force_ascii=False)
    raise ValueError(f"Unsupported type: {type(records)}")
