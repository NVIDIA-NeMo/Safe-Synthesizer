# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from nemo_safe_synthesizer.data_processing.validation import (
    check_groupby_column,
    check_orderby_column,
    check_timestamp_column,
)
from nemo_safe_synthesizer.errors import DataError, ParameterError


def test_check_groupby_column_noop_when_groupby_is_none() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    check_groupby_column(df, None)


def test_check_groupby_column_passes_when_column_exists() -> None:
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "group_col": ["g1", "g1", "g2", "g2", "g3"],
        }
    )
    check_groupby_column(df, "group_col")


def test_check_groupby_column_raises_for_missing_column() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(
        ParameterError,
        match=r"Group by column 'missing_group' not found in input dataset columns.*disable grouping",
    ):
        check_groupby_column(df, "missing_group")


def test_check_groupby_column_raises_for_comma_in_name() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ParameterError, match="multi-column grouping is not supported"):
        check_groupby_column(df, "col1,col2")


def test_check_groupby_column_raises_for_missing_values() -> None:
    df = pd.DataFrame({"group": ["x", None], "value": [1, 2]})
    with pytest.raises(DataError, match="missing values"):
        check_groupby_column(df, "group")


def test_check_orderby_column_noop_when_orderby_is_none() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    check_orderby_column(df, None)


def test_check_orderby_column_raises_for_missing_column() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ParameterError, match="not found in input dataset columns"):
        check_orderby_column(df, "missing_order")


def test_check_orderby_column_timeseries_with_generated_timestamp_is_noop() -> None:
    """Time-series runs with a generated timestamp intentionally bypass order-by validation."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    check_orderby_column(df, "will_be_generated", is_timeseries=True, timestamp_column=None)


def test_check_timestamp_column_raises_for_missing_column() -> None:
    df = pd.DataFrame({"val": [1, 2, 3]})
    with pytest.raises(ParameterError, match="not found"):
        check_timestamp_column(df, "ts")


def test_check_timestamp_column_raises_for_missing_values() -> None:
    df = pd.DataFrame({"ts": [1, None, 3]})
    with pytest.raises(DataError, match="missing values"):
        check_timestamp_column(df, "ts")


def test_multiindex_columns_are_rejected_with_actionable_message() -> None:
    # ``"ts" in MultiIndex[("ts","","ns")]`` returns False even though
    # level 0 contains ``"ts"``; the primitives can't answer "is column
    # present?" meaningfully on a MultiIndex schema, so they fail fast.
    df = pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples([("ts", "a"), ("ts", "b")]))
    with pytest.raises(ParameterError, match="MultiIndex columns are not supported"):
        check_groupby_column(df, "ts")
    with pytest.raises(ParameterError, match="MultiIndex"):
        check_timestamp_column(df, "ts")
