# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import faker
import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.parameters import (
    DifferentialPrivacyHyperparams,
    EvaluationParameters,
    SafeSynthesizerParameters,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import EvaluationDataset
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import PrivacyGrade
from nemo_safe_synthesizer.pii_replacer.transform_result import ColumnStatistics


def make_df(seed: int, n: int = 100):
    fake = faker.Faker("en_US")
    # Use seeds for consistency across tests.
    fake.seed_instance(seed)
    random.seed(seed)
    df = pd.DataFrame(
        {
            "num": [random.random() for _ in range(n)],
            "num_Int64": [random.randint(1, 100) for _ in range(n)],
            # Categorical columns according to gretel core arfifact classifier
            "num_cat": [random.randint(1, 4) for _ in range(n)],
            "num_cat_Int64": [random.randint(1, 4) for _ in range(n)],
            "small_cat": [random.choice(["foo", "bar", "baz", "biff", "barf"]) for _ in range(n)],
            # Neither categorical nor text
            "other": [fake.name() for _ in range(n)],
            "boolean": [random.choice([True, False]) for _ in range(n)],
        }
    )
    df["text"] = df["other"] + " had a little lamb"

    # randomly assign missing values
    df.loc[random.sample(list(df.index), k=10), "num_Int64"] = np.nan
    df.loc[random.sample(list(df.index), k=10), "num_cat_Int64"] = np.nan
    df.loc[random.sample(list(df.index), k=10), "small_cat"] = None
    df.loc[random.sample(list(df.index), k=2), "other"] = ""
    df.loc[random.sample(list(df.index), k=8), "other"] = None
    df.loc[random.sample(list(df.index), k=4), "text"] = ""
    # text_semantic_similarity can't handle None for now
    # df.loc[random.sample(list(df.index), k=6), "text"] = None

    # Convert to nullable dtypes first before assigning NaN values
    df["boolean"] = df["boolean"].astype(pd.BooleanDtype())
    df["num_Int64"] = df["num_Int64"].astype(pd.Int64Dtype())
    df["num_cat_Int64"] = df["num_cat_Int64"].astype(pd.Int64Dtype())

    df.loc[random.sample(list(df.index), k=10), "boolean"] = np.nan
    return df


@pytest.fixture
def train_df():
    return make_df(370)


@pytest.fixture
def train_df_5k():
    return make_df(370, 5000)


@pytest.fixture
def train_df_10k():
    return make_df(370, 10000)


@pytest.fixture
def synth_df():
    return make_df(753)


@pytest.fixture
def synth_df_5k():
    return make_df(753, 5000)


@pytest.fixture
def synth_df_10k():
    return make_df(753, 10000)


@pytest.fixture
def test_df():
    return make_df(476)


@pytest.fixture
def evaluation_dataset_5k(train_df_5k, synth_df_5k, test_df):
    return EvaluationDataset.from_dataframes(train_df_5k, synth_df_5k, test_df)


@pytest.fixture
def skip_privacy_metrics_config():
    return SafeSynthesizerParameters(evaluation=EvaluationParameters(mia_enabled=False, aia_enabled=False))


@pytest.fixture
def skip_privacy_metrics_and_synth_config():
    return SafeSynthesizerParameters(
        enable_synthesis=False, evaluation=EvaluationParameters(mia_enabled=False, aia_enabled=False)
    )


@pytest.fixture
def skip_synth_config():
    return SafeSynthesizerParameters(
        enable_synthesis=False, evaluation=EvaluationParameters(mia_enabled=False, aia_enabled=True)
    )


@pytest.fixture
def dp_enabled_config():
    return SafeSynthesizerParameters(
        privacy=DifferentialPrivacyHyperparams(dp_enabled=True, delta=0.1, epsilon=0.2),
        evaluation=EvaluationParameters(mia_enabled=False, aia_enabled=True),
    )


@pytest.fixture
def dp_not_enabled_config():
    return SafeSynthesizerParameters(evaluation=EvaluationParameters(mia_enabled=False, aia_enabled=True))


@pytest.fixture
def column_statistics(train_df_5k):
    small_cat_values = {"foo", "bar"}
    small_cat_count = len(train_df_5k["small_cat"].to_frame().query("`small_cat` in @small_cat_values"))
    other_cat_values = {"barf"}
    other_cat_count = len(train_df_5k["small_cat"].to_frame().query("`small_cat` in @other_cat_values"))
    small_cat_col_stats = ColumnStatistics(
        assigned_type="text",
        assigned_entity="some_cats",
        detected_entity_counts={"some_cats": small_cat_count, "other_cats": other_cat_count},
        detected_entity_values={"some_cats": small_cat_values, "other_cats": other_cat_values},
        is_transformed=True,
        transform_functions={"fake", "munge"},
    )

    other_values = set(train_df_5k["other"].head(250))
    other_count = len(other_values)
    other_col_stats = ColumnStatistics(
        assigned_type="text",
        assigned_entity="name",
        detected_entity_counts={
            "name": other_count,
        },
        detected_entity_values={
            "name": other_values,
        },
        is_transformed=True,
        transform_functions={"fake"},
    )

    return {
        "small_cat": small_cat_col_stats,
        "other": other_col_stats,
    }


@pytest.fixture
def mia_aia_df():
    fake = faker.Faker("en_US")
    fake.seed_instance(546)
    random.seed(302)
    return pd.DataFrame(
        {
            "Column": [fake.name() for _ in range(15)],
            "Risk": [random.randint(1, 100) for _ in range(15)],
            "Protection": [random.choice([g for g in PrivacyGrade][1:]) for _ in range(15)],
            "Attack Percentage": [random.randint(1, 100) for _ in range(15)],
        }
    )


@pytest.fixture
def mia_aia_df_with_nan_protection():
    """I'm not sure how often we get nans like this
    but we've seen errors like this in the wild:
    """
    fake = faker.Faker("en_US")
    fake.seed_instance(546)
    random.seed(302)
    return pd.DataFrame(
        {
            "Column": [fake.name() for _ in range(15)],
            "Risk": [random.randint(1, 100) for _ in range(15)],
            "Protection": [random.choice([g for g in PrivacyGrade][1:]) for _ in range(14)] + [np.nan],
            "Attack Percentage": [random.randint(1, 100) for _ in range(14)] + [np.nan],
        }
    )
