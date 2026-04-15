# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from nemo_safe_synthesizer.data_processing.validation import (
    validate_groupby_column,
    validate_orderby_column,
)
from nemo_safe_synthesizer.errors import DataError, ParameterError


def test_validate_groupby_column_noop_when_groupby_is_none() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validate_groupby_column(df, None)


def test_validate_groupby_column_passes_when_column_exists() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "group_col": ["g1", "g1", "g2", "g2", "g3"],
        }
    )
    validate_groupby_column(df, "group_col")


def test_validate_groupby_column_raises_for_missing_column() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(
        ParameterError,
        match=r"Group by column 'missing_group' not found in input dataset columns.*disable grouping",
    ):
        validate_groupby_column(df, "missing_group")


def test_validate_groupby_column_raises_for_comma_in_name() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ParameterError, match="multi-column grouping is not supported"):
        validate_groupby_column(df, "col1,col2")


def test_validate_groupby_column_raises_for_missing_values() -> None:
    df = pd.DataFrame({"group": ["x", None], "value": [1, 2]})
    with pytest.raises(DataError, match="missing values"):
        validate_groupby_column(df, "group")


def test_validate_orderby_column_noop_when_orderby_is_none() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validate_orderby_column(df, None)


def test_validate_orderby_column_raises_for_missing_column() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ParameterError, match="not found in the input data"):
        validate_orderby_column(df, "missing_order")
