# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feature extraction for DataFrame columns.

Analyzes each column to produce a ``FieldFeatures`` profile containing type
classification, value distribution, missing-data rates, and numeric precision.
The ``evaluation`` and ``pii_replacer`` packages import ``describe_field``,
``FieldFeatures``, and ``FieldType`` from this module.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from math import copysign, floor, log10
from typing import Any

from pandas import Series
from pandas.core.dtypes.common import is_float_dtype, is_numeric_dtype

from ..base.fields import (
    FieldFeatures,
    FieldType,
)

_TEXT_FIELD_AVG_SPACE_COUNT_THRESHOLD = 2


def float_precision(data: Series) -> tuple[int | None, int | None]:
    """Compute the range of decimal-digit counts in a float Series.

    Converts each value to its string representation, counts digits after
    the decimal point, and returns the (min, max) across the Series.

    Args:
        data: A pandas Series to analyze. Non-float dtypes return ``(None, None)``
            immediately.

    Returns:
        ``(min_precision, max_precision)`` tuple, or ``(None, None)`` if the
        Series is not float-typed or contains no meaningful decimal digits.
    """
    if not is_float_dtype(data.dtype):
        return None, None

    def _value_precision(value: float) -> int | None:
        parts = str(value).split(".")
        if len(parts) > 1 and parts[1] != "0":
            return len(parts[1])
        return 0

    precision_values = list(_filter_nan(_value_precision(item) for item in data))
    if len(precision_values) == 0:
        return None, None

    precision = min(precision_values), max(precision_values)
    if precision == (0, 0):
        return None, None

    return precision


def _filter_nan(iterable: Iterable[Any]) -> Iterable[Any]:
    """Remove ``None`` values from an iterable."""
    return [x for x in iterable if x is not None]


def describe_field(field_name: str, data: Series) -> FieldFeatures:
    """Build a statistical profile for a single DataFrame column.

    Computes value counts, uniqueness, missing-data rates, string-length
    statistics, and numeric precision. Infers a ``FieldType`` using the
    following heuristic priority:

    1. ``EMPTY`` -- all values are null.
    2. ``BINARY`` -- exactly two unique non-null values.
    3. ``NUMERIC`` -- numeric dtype (float or int); integer columns with
       <= 10 non-negative unique values are classified as ``CATEGORICAL``.
    4. ``CATEGORICAL`` -- high duplicate ratio (>= 90%, or >= 70% when
       the sample has <= 50 rows).
    5. ``TEXT`` -- average space count per value exceeds the threshold.
    6. ``OTHER`` -- none of the above rules matched.

    Args:
        field_name: Column name used as the ``FieldFeatures.name``.
        data: The column data as a pandas Series.

    Returns:
        A ``FieldFeatures`` instance with all statistics populated.
    """
    total_count = len(data)

    non_na_data = data.dropna()
    non_na_count = int(non_na_data.count())
    unique_values_list = non_na_data.unique().tolist()
    unique_count = len(unique_values_list)
    missing_count = total_count - non_na_count

    lengths = [len(str(entry)) for entry in non_na_data]
    features = FieldFeatures(
        name=field_name,
        type=FieldType.OTHER,
        count=non_na_count,
        unique_values_list=unique_values_list,
        unique_count=unique_count,
        unique_percent=(round(unique_count / non_na_count * 100, 4) if non_na_count > 0 else 0),
        missing_count=missing_count,
        missing_percent=(round(missing_count / total_count * 100, 4) if total_count > 0 else 0),
        min_str_length=min(lengths) if non_na_count > 0 else 0,
        max_str_length=max(lengths) if non_na_count > 0 else 0,
        avg_str_length=round(sum(lengths) / non_na_count, 4) if non_na_count > 0 else 0,
    )

    if non_na_count == 0:
        features.type = FieldType.EMPTY
        return features

    if unique_count == 2:
        features.type = FieldType.BINARY
        return features

    data_type = non_na_data.dtype
    if is_float_dtype(data_type):
        features.type = FieldType.NUMERIC
        features.min_value = floor_power_of_10(float(non_na_data.min()))
        features.max_value = floor_power_of_10(float(non_na_data.max()))
        features.min_precision, features.max_precision = float_precision(non_na_data)
        return features

    if is_numeric_dtype(data_type):
        min_value = floor_power_of_10(int(non_na_data.min()))
        if unique_count <= 10 and min_value >= 0:
            features.type = FieldType.CATEGORICAL
            return features

        features.type = FieldType.NUMERIC
        features.min_value = min_value
        features.max_value = floor_power_of_10(int(non_na_data.max()))
        return features

    # See https://jeffreymorgan.io/articles/identifying-categorical-data/
    diff = non_na_count - unique_count
    diff_percent = diff / non_na_count

    # ENGPROD-5: ``entry`` needs to be a string here regardless
    # in order to get the space count
    space_count = sum(str(entry).count(" ") for entry in non_na_data)
    features.space_count = space_count

    if diff_percent >= 0.9 or (diff_percent >= 0.7 and len(non_na_data) <= 50):
        features.type = FieldType.CATEGORICAL
        return features

    if space_count / non_na_count > _TEXT_FIELD_AVG_SPACE_COUNT_THRESHOLD:
        features.type = FieldType.TEXT
        return features

    return features


def floor_power_of_10(value: int | float) -> int | float:
    """Return the largest power of 10 that does not exceed ``abs(value)``.

    Special cases: returns 1 for zero, and passes through infinity and NaN
    unchanged. Negative values return the negative of the result.

    Args:
        value: The number to compute the floor power-of-10 for.

    Returns:
        A power of 10 with the same sign as ``value``.
    """
    if value == 0:
        return 1

    if math.isinf(value):
        return value

    if math.isnan(value):
        return value

    floor_log = floor(log10(abs(value)))
    if value < 0:
        floor_log += 1

    return copysign(10**floor_log, value)
