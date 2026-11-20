# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.errors import DataError, ParameterError
from nemo_safe_synthesizer.holdout.holdout import (
    HOLDOUT_TOO_SMALL_ERROR,
    INPUT_DATA_TOO_SMALL_ERROR,
    Holdout,
    grouped_train_test_split,
    naive_train_test_split,
)


@pytest.fixture
def df() -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "big_cat": np.random.choice(["yes", "no"], size=200),
            "value": np.random.randn(200),
        }
    )

    return df


def test_does_simple_holdout(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params())
    train, test = holdout.train_test_split(df)
    assert len(train) == 190
    assert test is not None
    assert len(test) == 10


def test_does_simple_float_holdout(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=0.1))
    train, test = holdout.train_test_split(df)
    assert len(train) == 180
    assert test is not None
    assert len(test) == 20


@pytest.mark.parametrize("value", [1, 1.0, 20, 20.2, -0.1])
def test_rejects_holdout_outside_fraction_range(value):
    with pytest.raises(ValueError, match=r"data.holdout must be a fraction in \[0, 1\)"):
        SafeSynthesizerParameters.from_params(holdout=value)


def test_rejects_removed_max_holdout_field():
    with pytest.raises(ValueError, match="max_holdout"):
        SafeSynthesizerParameters.from_params(max_holdout=2000)


def test_does_not_cap_large_holdouts():
    data = pd.DataFrame({"value": np.arange(50_000)})
    train, test = Holdout(SafeSynthesizerParameters.from_params(holdout=0.05)).train_test_split(data)
    assert test is not None
    assert len(test) == 2_500
    assert len(train) == 47_500


def test_zero_holdout(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=0))
    train, test = holdout.train_test_split(df)
    assert len(train) == 200
    assert test is None


def test_naive_train_test_split_zero_int_test_size(df):
    train, test = naive_train_test_split(input_df=df, test_size=0)
    assert len(train) == 200
    assert test is None


def test_does_group_by_holdout(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(group_training_examples_by="big_cat"))
    train, test = holdout.train_test_split(df)
    assert len(train) == 100
    assert test is not None
    assert len(test) == 100


@pytest.mark.parametrize(
    ("n_groups", "rows_per_group", "holdout", "expected_groups"),
    [
        (10, 200, 0.05, 1),
        (21, 200, 0.05, 2),
        (128, 400, 0.05, 7),
        (50, 10_000, 0.05, 3),
        (500, 10, 0.05, 25),
        (10, 200, 0.2, 2),
        (20, 200, 0.2, 4),
        (20, 200, 0.5, 10),
    ],
)
def test_group_by_holdout_holds_out_the_requested_fraction_of_groups(
    n_groups, rows_per_group, holdout, expected_groups
):
    n_rows = n_groups * rows_per_group
    data = pd.DataFrame({"group": np.repeat(np.arange(n_groups), rows_per_group), "value": np.arange(n_rows)})
    params = SafeSynthesizerParameters.from_params(group_training_examples_by="group", holdout=holdout, random_state=0)
    train, test = Holdout(params).train_test_split(data)
    assert test is not None
    assert test["group"].nunique() == expected_groups
    assert len(test) == expected_groups * rows_per_group
    assert set(train["group"]).isdisjoint(set(test["group"]))


@pytest.mark.parametrize("test_size", [0, 1, 20])
def test_grouped_train_test_split_requires_a_fraction(df, test_size):
    with pytest.raises(ParameterError, match="fraction"):
        grouped_train_test_split(df, test_size=test_size, group_by="big_cat")


def test_raises_on_group_by_holdout_with_bad_column(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(group_training_examples_by="dne"))
    with pytest.raises(ParameterError, match="Group by column 'dne' not found"):
        holdout.train_test_split(df)


def test_raises_on_group_by_holdout_with_missing_values(df):
    df_with_missing_group = df.copy()
    df_with_missing_group.loc[0, "big_cat"] = None
    holdout = Holdout(SafeSynthesizerParameters.from_params(group_training_examples_by="big_cat"))
    with pytest.raises(DataError, match="Group by column 'big_cat' has missing values"):
        holdout.train_test_split(df_with_missing_group)


def test_complains_when_training_dataset_is_too_small():
    df = pd.DataFrame({"a": range(199)})
    with pytest.raises(ValueError) as excinfo:
        holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=0.05))
        holdout.train_test_split(df)
    assert str(excinfo.value) == f"{INPUT_DATA_TOO_SMALL_ERROR}"


def test_complains_when_holdout_is_too_small(df):
    with pytest.raises(ValueError) as excinfo:
        holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=0.04))
        holdout.train_test_split(df)
    assert str(excinfo.value) == f"{HOLDOUT_TOO_SMALL_ERROR}"
