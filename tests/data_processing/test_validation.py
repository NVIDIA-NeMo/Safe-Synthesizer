# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from nemo_safe_synthesizer.data_processing.validation import (
    MISSING_GROUP_BY_COLUMN_ERROR,
    MISSING_GROUP_BY_VALUES_ERROR,
    MISSING_ORDER_BY_COLUMN_ERROR,
    validate_groupby_column,
    validate_orderby_column,
)
from nemo_safe_synthesizer.errors import DataError, ParameterError


def test_validate_groupby_column_noop_when_groupby_is_none() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validate_groupby_column(df, None)


def test_validate_groupby_column_raises_for_missing_column() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ParameterError) as excinfo:
        validate_groupby_column(df, "missing_group")
    assert str(excinfo.value) == MISSING_GROUP_BY_COLUMN_ERROR.format(group_by="missing_group")


def test_validate_groupby_column_raises_for_missing_values() -> None:
    df = pd.DataFrame({"group": ["x", None], "value": [1, 2]})
    with pytest.raises(DataError) as excinfo:
        validate_groupby_column(df, "group")
    assert str(excinfo.value) == MISSING_GROUP_BY_VALUES_ERROR.format(group_by="group")


def test_validate_orderby_column_noop_when_orderby_is_none() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validate_orderby_column(df, None)


def test_validate_orderby_column_raises_for_missing_column() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ParameterError) as excinfo:
        validate_orderby_column(df, "missing_order")
    assert str(excinfo.value) == MISSING_ORDER_BY_COLUMN_ERROR.format(order_by="missing_order")
