# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train/test splitting for evaluation holdout.

Provides two splitting strategies -- naive (random) and grouped (preserving
group membership) -- and a ``Holdout`` class that selects the appropriate
strategy based on pipeline configuration.
"""

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

DataFrameOptionalTuple = tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, None]


def naive_train_test_split(input_df, test_size, random_state=None) -> DataFrameOptionalTuple:
    """Split the input dataframe into training and test sets with a random shuffle.

    Thin wrapper around ``sklearn.model_selection.train_test_split`` that
    resets the index on both resulting dataframes.

    Args:
        input_df: Input dataframe to split.
        test_size: Number of rows (int) or fraction (float) to hold out.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of ``(training_df, test_df)``, or ``(training_df, None)`` if the
        split produces no test set.
    """
    if test_size == 0:
        return input_df.reset_index(drop=True), None

    training_df, test_df = train_test_split(input_df, test_size=test_size, random_state=random_state)
    if test_df is None:
        return training_df, None
    else:
        return training_df.reset_index(drop=True), test_df.reset_index(drop=True)


def grouped_train_test_split(input_df, test_size, group_by, random_state=None) -> DataFrameOptionalTuple:
    """Split the input dataframe so that all rows sharing a group stay in the same fold.

    Uses ``GroupShuffleSplit`` with 20 candidate splits and picks the one
    whose test-set size is closest to the requested ``test_size``.  If
    ``test_size`` exceeds the number of groups, equals 0, or equals 1, it
    falls back to ``DEFAULT_HOLDOUT``.

    Args:
        input_df: Input dataframe to split.
        test_size: Desired number of test rows (int) or fraction (float).
        group_by: Column name whose values define the groups.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of ``(training_df, test_df)``, or ``(input_df, None)`` if no valid
        grouped split could be produced.

    Raises:
        ValueError: If the ``group_by`` column contains missing values.
    """
    if input_df[group_by].isna().any():
        msg = f"Group by column '{group_by}' has missing values. Please remove/replace them."
        raise ValueError(msg)

    if test_size > input_df.groupby(group_by).ngroups or test_size == 1 or test_size == 0:
        logger.info(
            f"test_size ({test_size}) is greater than number of groups ({input_df.groupby(group_by).ngroups}) or equals to 0 or 1. Proceeding with default test_size ({DEFAULT_HOLDOUT})."
        )
        test_size = DEFAULT_HOLDOUT
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=20, random_state=random_state)
    split = splitter.split(input_df, groups=input_df[group_by])
    training_df, test_df = pd.DataFrame(), pd.DataFrame()
    if test_size > 1:
        aim_num_records = test_size
    else:
        aim_num_records = round(len(input_df) * test_size)
    for train_idx, test_idx in split:
        if len(training_df) == 0:
            training_df = input_df.iloc[train_idx]
            test_df = input_df.iloc[test_idx]
        elif abs(len(test_df) - aim_num_records) > abs(len(input_df.iloc[test_idx]) - aim_num_records):
            training_df = input_df.iloc[train_idx]
            test_df = input_df.iloc[test_idx]
    if len(test_df) == 0:
        logger.info("Failed to do grouped train/test split. Proceeding with original dataframe.")
        return input_df, None
    return training_df.reset_index(drop=True), test_df.reset_index(drop=True)


class Holdout:
    """Config-driven train/test splitter for the evaluation holdout set.

    Reads holdout parameters from the pipeline configuration and delegates
    to either ``naive_train_test_split`` or ``grouped_train_test_split``
    depending on whether a ``group_training_examples_by`` column is set.

    The holdout size is resolved as follows:

      - If ``holdout < 1.0``, it is treated as a fraction of the input rows.
      - If ``holdout >= 1.0``, it is treated as an absolute row count.
      - The result is clamped to ``max_holdout`` and must be at least
        ``MIN_HOLDOUT`` rows.

    Args:
        config: Pipeline parameters providing ``holdout``, ``max_holdout``,
            ``group_training_examples_by``, and ``random_state``.
    """

    def __init__(self, config: SafeSynthesizerParameters):
        self.holdout = config.get("holdout")
        self.max_holdout = config.get("max_holdout")
        self.group_by = config.get("group_training_examples_by")
        self.random_state = config.get("random_state")

    def train_test_split(self, input_df: pd.DataFrame) -> DataFrameOptionalTuple:
        """Split the input dataframe into training and holdout test sets.

        Returns the full dataframe with no test set when holdout is disabled
        (``holdout == 0`` or ``max_holdout == 0``).

        Args:
            input_df: The full input dataframe to split.

        Returns:
            Tuple of ``(training_df, test_df)``, or ``(training_df, None)`` when
            holdout is disabled or the grouped split fails.

        Raises:
            ValueError: If the input dataframe has fewer than
                ``MIN_RECORDS_FOR_TEXT_AND_PRIVACY_METRICS`` rows, if the
                computed holdout is smaller than ``MIN_HOLDOUT``, or if
                the ``group_by`` column contains missing values.
        """
        if self.holdout == 0 or self.max_holdout == 0:
            return input_df, None

        # Check if the input dataset is large enough to hold out
        if len(input_df) < MIN_RECORDS_FOR_TEXT_AND_PRIVACY_METRICS:
            raise ValueError(
                INPUT_DATA_TOO_SMALL_ERROR,
            )

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
            training_df, test_df = grouped_train_test_split(
                input_df=input_df,
                test_size=final_holdout,
                group_by=self.group_by,
                random_state=self.random_state,
            )
        else:
            training_df, test_df = naive_train_test_split(
                input_df=input_df,
                test_size=final_holdout,
                random_state=self.random_state,
            )

        return training_df, test_df
