# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DataFrame column normalization for training and evaluation inputs.

Provides utilities for standardizing DataFrames (type coercion and
missing-value handling) before profiling or downstream modeling.
"""

from __future__ import annotations

import pandas as pd

CONVERT_TO_STR_TYPES = [
    "mixed",
    "mixed-integer",
    "mixed-integer-float",
    "datetime64",
    "datetime",
    "date",
    "timedelta64",
    "timedelta",
    "time",
]

CONVERT_TO_FLOAT_TYPES = [
    "decimal",
]


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame meets standards for use in Safe Synthesizer models.

    Pandas may be used to construct a lot of odd DataFrames with weird corner
    cases that can violate assumptions of downstream code and other libraries.
    This includes differences creating a DataFrame from different sources (like
    csv vs json vs jsonl) or when a manually crafted DataFrame is provided, such
    as using SafeSynthesizerDataset for testing. Rather than be defensive in every
     model where we use SafeSynthesizerDataset, we do some standardization of
    all DataFrames here.

    Enforced standards, i.e., assumptions models that use SafeSynthesizerDataset may make:
    - Every column has a single datatype, e.g. all float, all str, or all int,
      with the exception of missing values in object columns, where we keep the
      pandas behavior of representing missing with a float numpy.nan for now.

    - Date, time, datetime, and timedelta types are converted to string for
      downstream consistency between tokenization and schema serialization.
      Decimal types are converted to float.
    """
    column_series = {column_name: normalize_column(df[column_name]) for column_name in df.columns}
    return pd.DataFrame(column_series)


def normalize_column(series: pd.Series) -> pd.Series:
    """Normalize the given pandas series.

    Args:
        series: Series to normalize.

    Returns:
        Normalized series.
    """
    series_type = pd.api.types.infer_dtype(series, skipna=True)
    if series_type in CONVERT_TO_STR_TYPES:
        return series.astype(str).mask(series.isna(), None)
    if series_type in CONVERT_TO_FLOAT_TYPES:
        return series.astype(float).mask(series.isna(), None)
    return series
