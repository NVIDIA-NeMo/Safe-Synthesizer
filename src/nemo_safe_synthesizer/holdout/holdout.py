# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ..config.data import DEFAULT_HOLDOUT, MIN_HOLDOUT
from ..config.parameters import SafeSynthesizerParameters
from ..observability import get_logger

MIN_RECORDS_FOR_TEXT_AND_PRIVACY_METRICS = 200

HOLDOUT_TOO_SMALL_ERROR = (
    f"Holdout dataset must have at least {MIN_HOLDOUT} records. Please increase the holdout or disable holdout."
)
INPUT_DATA_TOO_SMALL_ERROR = (
    f"Dataset must have at least {MIN_RECORDS_FOR_TEXT_AND_PRIVACY_METRICS} records to use holdout."
)

logger = get_logger(__name__)

DataFrameOptionalTuple = tuple[pd.DataFrame, pd.DataFrame | None]


def naive_train_test_split(df, test_size, random_state=None) -> DataFrameOptionalTuple:
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    if test is None:
        return train, None
    else:
        return train.reset_index(drop=True), test.reset_index(drop=True)


def grouped_train_test_split(df, test_size, group_by, random_state=None) -> DataFrameOptionalTuple:
    # Do not continue the split process if the groupby column has missing values.
    if df[group_by].isna().any():
        msg = f"Group by column '{group_by}' has missing values. Please remove/replace them."
        raise ValueError(msg)

    if test_size > df.groupby(group_by).ngroups or test_size == 1 or test_size == 0:
        logger.info(
            f"test_size ({test_size}) is greater than number of groups ({df.groupby(group_by).ngroups}) or equals to 0 or 1. Proceeding with default test_size ({DEFAULT_HOLDOUT})."
        )
        test_size = DEFAULT_HOLDOUT
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=20, random_state=random_state)
    split = splitter.split(df, groups=df[group_by])
    df_train, df_test = pd.DataFrame(), pd.DataFrame()
    if test_size > 1:
        aim_num_records = test_size
    else:
        aim_num_records = round(len(df) * test_size)
    for train_idx, test_idx in split:
        if len(df_train) == 0:
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
        elif abs(len(df_test) - aim_num_records) > abs(len(df.iloc[test_idx]) - aim_num_records):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
    if len(df_test) == 0:
        logger.info("Failed to do grouped train/test split. Proceeding with original dataframe.")
        return df, None
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


class Holdout:
    def __init__(self, config: SafeSynthesizerParameters):
        self.holdout = config.get("holdout")
        self.max_holdout = config.get("max_holdout")
        self.group_by = config.get("group_training_examples_by")
        self.random_state = config.get("random_state")

    def train_test_split(self, input_df: pd.DataFrame) -> DataFrameOptionalTuple:
        if self.holdout == 0 or self.max_holdout == 0:
            return input_df, None

        # Check if the input dataset is large enough to hold out
        if len(input_df) < MIN_RECORDS_FOR_TEXT_AND_PRIVACY_METRICS:
            raise ValueError(
                INPUT_DATA_TOO_SMALL_ERROR,
            )

        assert self.holdout is not None
        assert self.max_holdout is not None

        # Find the number of records to hold out
        if self.holdout < 1.0:
            final_holdout = len(input_df) * self.holdout
        else:
            final_holdout = self.holdout

        # Clip the number of records to hold out as needed. We always want an int at this point, do a cast.
        final_holdout = int(min(final_holdout, self.max_holdout))

        # Check that the holdout is at least 10 records
        if final_holdout < MIN_HOLDOUT:
            raise ValueError(
                HOLDOUT_TOO_SMALL_ERROR,
            )

        if self.group_by is not None and self.group_by not in input_df.columns:
            logger.warning(f"Group By column {self.group_by} not found in input Dataset columns! Doing a normal split.")
            self.group_by = None
        if self.group_by is not None and input_df[self.group_by].isna().any():
            raise ValueError(f"Group by column '{self.group_by}' has missing values. Please remove/replace them.")

        if self.group_by:
            df, test_df = grouped_train_test_split(
                df=input_df,
                test_size=final_holdout,
                group_by=self.group_by,
                random_state=self.random_state,
            )
        else:
            df, test_df = naive_train_test_split(
                df=input_df,
                test_size=final_holdout,
                random_state=self.random_state,
            )

        return df, test_df
