# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated adapters for ordered record batches."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

from ..errors import ParameterError
from .types import JsonObject, JsonValue


def validate_json_value(value: object) -> JsonValue:
    """Validate the strict NSS JSON value domain without coercion."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise ParameterError("Records must contain finite JSON values.")
    if isinstance(value, list):
        return [validate_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ParameterError("Records must contain JSON objects with string keys.")
        return {cast(str, key): validate_json_value(item) for key, item in value.items()}
    raise ParameterError("Records must contain finite JSON values without coercion.")


def columnar_batch_to_records(
    columns: Mapping[str, Sequence[object]],
    *,
    row_count: int | None = None,
) -> tuple[JsonObject, ...]:
    """Convert an ordered columnar batch to independent ordered row mappings.

    Args:
        columns: Ordered string column names mapped to equally sized sequences.
        row_count: Explicit row count, required to represent positive-row
            zero-column batches.

    Returns:
        Ordered row mappings preserving column insertion order.

    Raises:
        ParameterError: If names, lengths, values, or ``row_count`` are invalid.
    """
    if not isinstance(columns, Mapping) or not all(isinstance(name, str) for name in columns):
        raise ParameterError("Record column names must be strings.")
    if row_count is not None and (not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0):
        raise ParameterError("Record row_count must be a non-negative integer.")

    normalized: list[tuple[str, Sequence[object]]] = []
    lengths: set[int] = set()
    for name, values in columns.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise ParameterError(f"Record column {name!r} must be a concrete sequence.")
        normalized.append((name, values))
        lengths.add(len(values))
    if len(lengths) > 1:
        raise ParameterError("Record columns must have equal lengths.")

    inferred = next(iter(lengths), 0)
    if row_count is not None and normalized and row_count != inferred:
        raise ParameterError("Explicit record row_count must match all column lengths.")
    count = inferred if row_count is None else row_count
    return tuple({name: validate_json_value(values[index]) for name, values in normalized} for index in range(count))
