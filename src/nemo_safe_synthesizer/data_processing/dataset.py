# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
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


JSON_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}

# schema detection parameters
SCHEMA_ENUM_MAX_DISTINCT_EXP = 1 / 2
SCHEMA_ENUM_MAX_SINGLETONS_EXP = 1 / 3
STRING_LENGTH_MULTIPLE = 1.5


def _handle_enum_value(v: object) -> None | int | float | bool | str:
    # TabFT uses None for na
    if pd.isna(v):
        return None

    if isinstance(v, (float, int, bool, str)):
        return v

    # Anything except builtin python types may cause crashes when we work with
    # the schema later, so we convert to bool, float, int, or str.
    if isinstance(v, np.bool_):
        return bool(v)

    try:
        # Convert to python int if possible, but np.float32 and other float
        # types will be truncated by int(v), so check equality to make sure
        # we haven't lost precision.
        t = int(v)
        if t == v:
            return t
    except Exception:
        pass

    try:
        # Convert to python float if possible
        return float(v)
    except Exception:
        pass

    # Otherwise, ensure we're using a python str to avoid json encoding errors
    return str(v)


def check_enum_type(
    series: pd.Series,
    max_distinct: int | float | None = None,
    max_singletons: int | float | None = None,
) -> dict | None:
    """Return enum schema if the series is an enum, otherwise return None.

    Args:
        series: Data object to check for enum type.
        max_distinct: Maximum number of distinct values to be considered an enum.
        max_singletons: Maximum number of values with a single occurrence to be
            considered an enum.

    Returns:
        The enum schema if the series is an enum, otherwise None.
    """
    if max_distinct is None:
        max_distinct = len(series) ** SCHEMA_ENUM_MAX_DISTINCT_EXP
    if max_singletons is None:
        max_singletons = len(series) ** SCHEMA_ENUM_MAX_SINGLETONS_EXP

    value_counts = series.value_counts(dropna=False)

    is_enum = value_counts.count() <= max_distinct and (value_counts == 1).sum() <= max_singletons

    return {"enum": [_handle_enum_value(v) for v in value_counts.index.sort_values()]} if is_enum else None


def make_json_schema(df: pd.DataFrame, string_length_multiple: float = STRING_LENGTH_MULTIPLE) -> dict:
    """Generate a JSON schema from the given DataFrame.

    Info on the schema definition: https://json-schema.org

    Args:
        df: Generate a JSON schema for this DataFrame.
        string_length_multiple: A multiplier to apply to the min and max string

    Returns:
        A dictionary representing the JSON schema.
    """
    schema = {"type": "object", "properties": {}, "required": []}

    for col in df.columns:
        series = df[col]
        col_schema = check_enum_type(series) or {}

        if not col_schema:
            col_types = [JSON_TYPE_MAP.get(t, "string") for t in series.apply(lambda x: type(x).__name__).unique()]

            col_types = list(set(col_types))

            if series.isna().any():
                col_types.append("null")

            if set(col_types).issubset(["integer", "number"]):
                col_schema.update(
                    {
                        "type": col_types[0],  # actual element instead of list
                        "minimum": float(series.min()),
                        "maximum": float(series.max()),
                    }
                )
            elif col_types == ["string"]:
                str_length = series.astype(str).apply(len)
                col_schema.update(
                    {
                        "type": "string",
                        "minLength": round(str_length.min() / string_length_multiple),
                        "maxLength": round(string_length_multiple * str_length.max()),
                    }
                )
            else:
                col_schema.update({"type": col_types})

        schema["properties"][col] = col_schema

        if not series.isna().any():
            schema["required"].append(col)

    return schema


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
