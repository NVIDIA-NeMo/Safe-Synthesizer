# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation test fixtures: synthetic/training/test DataFrames and evaluation configs."""

from __future__ import annotations

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
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
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
            # Categorical columns according to core artifact classifier
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
def fixture_training_df() -> pd.DataFrame:
    """100-row training DataFrame seeded at 370."""
    return make_df(370)


@pytest.fixture
def fixture_training_df_5k() -> pd.DataFrame:
    """5 000-row training DataFrame seeded at 370."""
    return make_df(370, 5000)


@pytest.fixture
def fixture_training_df_10k() -> pd.DataFrame:
    """10 000-row training DataFrame seeded at 370."""
    return make_df(370, 10000)


@pytest.fixture
def fixture_synthetic_df() -> pd.DataFrame:
    """100-row synthetic DataFrame seeded at 753."""
    return make_df(753)


@pytest.fixture
def fixture_synthetic_df_5k() -> pd.DataFrame:
    """5 000-row synthetic DataFrame seeded at 753."""
    return make_df(753, 5000)


@pytest.fixture
def fixture_synthetic_df_10k() -> pd.DataFrame:
    """10 000-row synthetic DataFrame seeded at 753."""
    return make_df(753, 10000)


@pytest.fixture
def fixture_test_df() -> pd.DataFrame:
    """100-row holdout test DataFrame seeded at 476."""
    return make_df(476)


@pytest.fixture
def fixture_evaluation_datasets_5k(
    fixture_training_df_5k, fixture_synthetic_df_5k, fixture_test_df
) -> EvaluationDatasets:
    """EvaluationDatasets built from 5k training/synthetic and 100-row test."""
    return EvaluationDatasets.from_dataframes(fixture_training_df_5k, fixture_synthetic_df_5k, fixture_test_df)


@pytest.fixture
def fixture_skip_privacy_metrics_config() -> SafeSynthesizerParameters:
    """Config with MIA and AIA disabled."""
    return SafeSynthesizerParameters(evaluation=EvaluationParameters(mia_enabled=False, aia_enabled=False))


@pytest.fixture
def fixture_dp_enabled_config() -> SafeSynthesizerParameters:
    """Config with DP enabled (epsilon=0.2, delta=0.1) and AIA on."""
    return SafeSynthesizerParameters(
        privacy=DifferentialPrivacyHyperparams(dp_enabled=True, delta=0.1, epsilon=0.2),
        evaluation=EvaluationParameters(mia_enabled=False, aia_enabled=True),
    )


@pytest.fixture
def fixture_dp_not_enabled_config() -> SafeSynthesizerParameters:
    """Config with DP disabled but AIA enabled."""
    return SafeSynthesizerParameters(evaluation=EvaluationParameters(mia_enabled=False, aia_enabled=True))


@pytest.fixture
def fixture_column_statistics(fixture_training_df_5k) -> dict[str, ColumnStatistics]:
    """ColumnStatistics for `small_cat` and `other` columns derived from the 5k training DataFrame."""
    small_cat_values = {"foo", "bar"}
    small_cat_count = len(fixture_training_df_5k["small_cat"].to_frame().query("`small_cat` in @small_cat_values"))
    other_cat_values = {"barf"}
    other_cat_count = len(fixture_training_df_5k["small_cat"].to_frame().query("`small_cat` in @other_cat_values"))
    small_cat_col_stats = ColumnStatistics(
        assigned_type="text",
        assigned_entity="some_cats",
        detected_entity_counts={"some_cats": small_cat_count, "other_cats": other_cat_count},
        detected_entity_values={"some_cats": small_cat_values, "other_cats": other_cat_values},
        is_transformed=True,
        transform_functions={"fake", "munge"},
    )

    other_values = set(fixture_training_df_5k["other"].head(250))
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
def fixture_mia_aia_df() -> pd.DataFrame:
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


def make_mixed_text_tabular_df(seed: int, n: int = 100):
    """Create a DataFrame with both text columns (>2 spaces avg) and tabular columns.

    This triggers the hybrid text+tabular nearest neighbor path in AIA/MIA evaluation,
    which uses both sentence-transformers for text similarity and sklearn NearestNeighbors
    for tabular similarity.
    """
    fake = faker.Faker("en_US")
    fake.seed_instance(seed)
    random.seed(seed)

    # Text templates with >2 spaces on average (triggers text classification)
    text_templates = [
        "The customer {} purchased {} items at the {} store on {}",
        "Order {} was shipped to {} via {} delivery service today",
        "Patient {} reported {} symptoms during the {} consultation",
        "Employee {} completed {} tasks in the {} department this quarter",
        "Student {} achieved {} points on the {} examination today",
    ]

    df = pd.DataFrame(
        {
            # Numeric columns (tabular)
            "amount": [round(random.uniform(10.0, 1000.0), 2) for _ in range(n)],
            "quantity": [random.randint(1, 50) for _ in range(n)],
            "score": [round(random.uniform(0.0, 100.0), 1) for _ in range(n)],
            # Categorical columns (tabular)
            "category": [random.choice(["A", "B", "C", "D"]) for _ in range(n)],
            "status": [random.choice(["active", "pending", "completed"]) for _ in range(n)],
            # Text columns (>2 spaces on average - triggers text embedding path)
            "description": [
                random.choice(text_templates).format(fake.name(), random.randint(1, 100), fake.company(), fake.date())
                for _ in range(n)
            ],
            "notes": [
                f"This is a detailed note about {fake.name()} who works at {fake.company()} in {fake.city()}"
                for _ in range(n)
            ],
        }
    )

    # Add some missing values
    df.loc[random.sample(list(df.index), k=min(5, n // 20)), "amount"] = np.nan
    df.loc[random.sample(list(df.index), k=min(3, n // 30)), "category"] = None
    df.loc[random.sample(list(df.index), k=min(2, n // 50)), "description"] = ""

    return df


def make_text_only_df(seed: int, n: int = 100):
    """Create a DataFrame with only text columns (>2 spaces avg).

    This triggers the text-only nearest neighbor path in AIA/MIA evaluation,
    which uses only sentence-transformers for similarity (no sklearn).
    """
    fake = faker.Faker("en_US")
    fake.seed_instance(seed)
    random.seed(seed)

    # All columns have >2 spaces on average to be classified as "text"
    df = pd.DataFrame(
        {
            "description": [
                f"The customer {fake.name()} purchased {random.randint(1, 100)} items at {fake.company()}"
                for _ in range(n)
            ],
            "notes": [
                f"This is a detailed note about {fake.name()} who works at {fake.company()} in {fake.city()}"
                for _ in range(n)
            ],
            "summary": [
                f"Summary for {fake.name()}: completed {random.randint(1, 50)} tasks in {fake.city()} office"
                for _ in range(n)
            ],
        }
    )

    # Add some empty values (but not too many to break the test)
    df.loc[random.sample(list(df.index), k=min(2, n // 50)), "description"] = ""

    return df


@pytest.fixture
def fixture_training_df_text_only() -> pd.DataFrame:
    """Training DataFrame with only text columns (500 rows)."""
    return make_text_only_df(seed=444, n=500)


@pytest.fixture
def fixture_synthetic_df_text_only() -> pd.DataFrame:
    """Synthetic DataFrame with only text columns (500 rows)."""
    return make_text_only_df(seed=555, n=500)


@pytest.fixture
def fixture_test_df_text_only() -> pd.DataFrame:
    """Test DataFrame with only text columns (100 rows)."""
    return make_text_only_df(seed=666, n=100)


@pytest.fixture
def fixture_training_df_mixed_5k() -> pd.DataFrame:
    """Training DataFrame with mixed text+tabular columns (5000 rows)."""
    return make_mixed_text_tabular_df(seed=111, n=5000)


@pytest.fixture
def fixture_synthetic_df_mixed_5k() -> pd.DataFrame:
    """Synthetic DataFrame with mixed text+tabular columns (5000 rows)."""
    return make_mixed_text_tabular_df(seed=222, n=5000)


@pytest.fixture
def fixture_test_df_mixed() -> pd.DataFrame:
    """Test DataFrame with mixed text+tabular columns (100 rows)."""
    return make_mixed_text_tabular_df(seed=333, n=100)


@pytest.fixture
def fixture_mia_aia_df_with_nan_protection() -> pd.DataFrame:
    """MIA/AIA DataFrame with NaN values in the Protection and Attack Percentage columns (regression data)."""
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
