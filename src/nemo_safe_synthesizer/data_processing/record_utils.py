# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from csv import QUOTE_NONNUMERIC
from io import StringIO

import jsonschema
import pandas as pd

from nemo_safe_synthesizer.observability import get_logger

RECORD_REGEX_PATTERN = r"{.+?}(?:\n|$)"
RECORD_REGEX_PATTEN_LOOKAHEAD = r"{.+?}(?=\n|$)"

logger = get_logger()


def is_safe_for_float_conversion(value: str | int | float | None | list | dict) -> bool:
    """Check if a value can be safely converted to float64 without overflow.

    Args:
        value: The value to check

    Returns:
        bool: True if the value can be safely converted to float64
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
    """Check if a record contains any numbers that would cause float conversion errors.

    Args:
        record: The record to check

    Returns:
        Tuple[bool, str]: (is_safe, err_msg)
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
        try:
            matched_dict = json.loads(matched_json)
            jsonschema.validate(matched_dict, schema)

            # Check for large numbers that would cause float conversion errors
            # If is safe, error_msg is None
            error_msg = check_record_for_large_numbers(matched_dict)
            if not error_msg:
                valid_records.append(matched_dict)
            else:
                invalid_records.append(matched_json)
                invalid_record_errors.append((error_msg, "Float Conversion"))

        except json.JSONDecodeError as e:
            invalid_records.append(matched_json)
            invalid_record_errors.append((f"Invalid JSON: {e.msg}", "Invalid JSON"))
        except jsonschema.exceptions.ValidationError as e:
            invalid_records.append(matched_json)
            invalid_record_errors.append((e.message, e.validator))

    return valid_records, invalid_records, invalid_record_errors


def normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize the given pandas dataframe
    - Convert missing values to consistent pd.NA
    - Resolve any utf-8 encoding errors

    Args:
        dataframe: DataFrame to normalize.

    Returns:
        DataFrame with missing values normalized, invalid utf-8 characters dropped.
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
