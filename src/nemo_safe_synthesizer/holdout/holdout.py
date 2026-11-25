# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train/test splitting for evaluation holdout.

Provides two splitting strategies -- naive (random) and grouped (preserving
group membership) -- and a ``Holdout`` class that selects the appropriate
strategy based on pipeline configuration.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ..config.data import MIN_HOLDOUT
from ..config.parameters import SafeSynthesizerParameters
from ..data_processing.validation import check_groupby_column
from ..errors import ParameterError
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


def naive_train_test_split(
    input_df: pd.DataFrame, test_size: float | int, random_state: int | None = None
) -> DataFrameOptionalTuple:
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


def grouped_train_test_split(
    input_df: pd.DataFrame, test_size: float, group_by: str, random_state: int | None = None
) -> DataFrameOptionalTuple:
    """Split the input dataframe so that all rows sharing a group stay in the same fold.

    ``GroupShuffleSplit`` holds out ``test_size`` of the groups, rounded up to
    whole groups. Of 20 candidate splits, the one whose row count is closest to
    ``test_size`` of the rows is kept, which matters when groups differ in size.

    Args:
        input_df: Input dataframe to split.
        test_size: Fraction of groups to hold out, in ``(0, 1)``.
        group_by: Column name whose values define the groups.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of ``(training_df, test_df)``, or ``(input_df, None)`` if no valid
        grouped split could be produced.

    Raises:
        ParameterError: If ``test_size`` is not in ``(0, 1)`` or the
            ``group_by`` column is not present in ``df``.
        DataError: If the ``group_by`` column contains missing values.
    """
    if not 0 < test_size < 1:
        raise ParameterError(f"Grouped test_size must be a fraction in (0, 1), got {test_size}.")
    check_groupby_column(input_df, group_by)

    splitter = GroupShuffleSplit(test_size=test_size, n_splits=20, random_state=random_state)
    split = splitter.split(input_df, groups=input_df[group_by])
    training_df, test_df = pd.DataFrame(), pd.DataFrame()
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

    ``holdout`` is a fraction in ``[0, 1)``. Ungrouped data holds out that
    fraction of the rows, rounded down. Grouped data holds out that fraction of
    the groups, rounded up to whole groups. Either way the holdout must cover at
    least ``MIN_HOLDOUT`` rows.

    Args:
        config: Pipeline parameters providing ``holdout``,
            ``group_training_examples_by``, and ``random_state``.
    """

    def __init__(self, config: SafeSynthesizerParameters):
        self.holdout = config.get("holdout")
        self.group_by = config.get("group_training_examples_by")
        self.random_state = config.get("random_state")

    def train_test_split(self, input_df: pd.DataFrame) -> DataFrameOptionalTuple:
        """Split the input dataframe into training and holdout test sets.

        Returns the full dataframe with no test set when ``holdout`` is 0.

        Args:
            input_df: The full input dataframe to split.

        Returns:
            Tuple of ``(training_df, test_df)``, or ``(training_df, None)`` when
            holdout is disabled or the grouped split fails.

        Raises:
            ValueError: If the input dataframe has fewer than
                ``MIN_RECORDS_FOR_TEXT_AND_PRIVACY_METRICS`` rows or if the
                computed holdout is smaller than ``MIN_HOLDOUT``.
            ParameterError: If the configured ``group_by`` column is missing.
            DataError: If the configured ``group_by`` column contains missing
                values.
        """
        holdout_val = self.holdout
        if not holdout_val:
            return input_df, None

        # Check if the input dataset is large enough to hold out
        if len(input_df) < MIN_RECORDS_FOR_TEXT_AND_PRIVACY_METRICS:
            raise ValueError(
                INPUT_DATA_TOO_SMALL_ERROR,
            )

        # Narrow from Unknown (Pydantic descriptor) so ty can check arithmetic.
        holdout: float = holdout_val
        holdout_rows = int(len(input_df) * holdout)
        if holdout_rows < MIN_HOLDOUT:
            raise ValueError(
                HOLDOUT_TOO_SMALL_ERROR,
            )

        check_groupby_column(input_df, self.group_by)

        if self.group_by:
            training_df, test_df = grouped_train_test_split(
                input_df=input_df,
                test_size=holdout,
                group_by=self.group_by,
                random_state=self.random_state,
            )
        else:
            training_df, test_df = naive_train_test_split(
                input_df=input_df,
                test_size=holdout_rows,
                random_state=self.random_state,
            )

        return (training_df, test_df)
