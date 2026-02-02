# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.holdout.holdout import (
    HOLDOUT_TOO_SMALL_ERROR,
    INPUT_DATA_TOO_SMALL_ERROR,
    Holdout,
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


def test_does_simple_int_holdout(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=20))
    train, test = holdout.train_test_split(df)
    assert len(train) == 180
    assert test is not None
    assert len(test) == 20


def test_does_clip_float_holdout(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=0.2, max_holdout=10))
    train, test = holdout.train_test_split(df)
    assert len(train) == 190
    assert test is not None
    assert len(test) == 10


def test_does_clip_int_holdout(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=20, max_holdout=10))
    train, test = holdout.train_test_split(df)
    assert len(train) == 190
    assert test is not None
    assert len(test) == 10


def test_gt_one_float_holdout(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=20.2, max_holdout=30))
    train, test = holdout.train_test_split(df)
    assert len(train) == 180
    assert test is not None
    assert len(test) == 20


def test_zero_holdout(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=0, max_holdout=30))
    train, test = holdout.train_test_split(df)
    assert len(train) == 200
    assert test is None


def test_zero_max_holdout(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=0.05, max_holdout=0))
    train, test = holdout.train_test_split(df)
    assert len(train) == 200
    assert test is None


def test_does_group_by_holdout(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(group_training_examples_by="big_cat"))
    train, test = holdout.train_test_split(df)
    assert len(train) == 100
    assert test is not None
    assert len(test) == 100


def test_skips_group_by_holdout_with_bad_column(df):
    holdout = Holdout(SafeSynthesizerParameters.from_params(group_training_examples_by="dne"))
    train, test = holdout.train_test_split(df)
    assert len(train) == 190
    assert test is not None
    assert len(test) == 10


def test_complains_when_training_dataset_is_too_small():
    df = pd.DataFrame({"a": range(199)})
    with pytest.raises(ValueError) as excinfo:
        holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=0.05))
        holdout.train_test_split(df)
    assert str(excinfo.value) == f"{INPUT_DATA_TOO_SMALL_ERROR}"


def test_complains_when_holdout_is_too_small(df):
    # When using integer holdout
    with pytest.raises(ValueError) as excinfo:
        holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=9))
        holdout.train_test_split(df)
    assert str(excinfo.value) == f"{HOLDOUT_TOO_SMALL_ERROR}"

    # When using float holdout
    with pytest.raises(ValueError) as excinfo:
        holdout = Holdout(SafeSynthesizerParameters.from_params(holdout=0.04))
        holdout.train_test_split(df)
    assert str(excinfo.value) == f"{HOLDOUT_TOO_SMALL_ERROR}"
