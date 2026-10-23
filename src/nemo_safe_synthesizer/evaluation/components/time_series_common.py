# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared sequence and column handling for time-series evaluation metrics.

Every time-series metric resolves the same grouping column, selects the same
numeric value columns, and walks the same shared groups in the same order.
Keeping that logic here means a fix applies to every metric at once.

Functions:
    resolve_group_column: Return the grouping column evaluation should use.
    select_value_columns: Select numeric value columns to evaluate.
    invalid_value_columns: Validate explicitly configured value columns.
    shared_groups: Find a deterministic, bounded set of shared groups.
    group_sort_key: Return a stable ordering key for group values.
    group_frame: Return one sequence in timestamp order.
    summaries: Average similarities by group or column.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from ...artifacts.analyzers.field_features import FieldType
from ...config.parameters import SafeSynthesizerParameters
from ...defaults import PSEUDO_GROUP_COLUMN
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets


def resolve_group_column(
    datasets: EvaluationDatasets,
    config: SafeSynthesizerParameters | None,
) -> tuple[str | None, str | None]:
    """Return the grouping column evaluation should use.

    Args:
        datasets: Training and synthetic datasets to compare.
        config: Optional top-level configuration naming the group column.

    Returns:
        The group column, or ``None`` for one global sequence, and an error
        note when a configured group column is missing from a dataset.
    """
    group_column = config.data.group_training_examples_by if config is not None else None
    if (
        group_column == PSEUDO_GROUP_COLUMN
        and group_column not in datasets.training
        and group_column not in datasets.synthetic
    ):
        # Training injects this reserved column to reuse grouped sequence
        # infrastructure, but evaluation receives frames with it removed.
        return None, None
    if group_column is not None and (group_column not in datasets.training or group_column not in datasets.synthetic):
        return None, f"Configured group column {group_column!r} is missing from a dataset."
    return group_column, None


def select_value_columns(
    datasets: EvaluationDatasets,
    value_columns: list[str] | None,
    timestamp_column: str | None,
    group_column: str | None,
    label: str,
) -> tuple[list[str], str | None, list[str]]:
    """Select numeric value columns eligible for evaluation.

    Explicit ``value_columns`` retain their configured order. Automatic
    selection is sorted for deterministic output and follows the training
    field types, so a synthetic column that collapsed to a constant or a few
    integers is still evaluated (and penalized) instead of silently dropping
    out. Timestamp and grouping columns are excluded even when their storage
    dtype is numeric.

    Args:
        datasets: Training and synthetic datasets with inferred field types.
        value_columns: Explicitly configured columns, or ``None`` for automatic
            selection.
        timestamp_column: Column used only to order observations.
        group_column: Column used only to separate sequences.
        label: Metric name used in the validation error, such as
            ``"autocorrelation"``.

    Returns:
        Selected column names, an optional validation error, and the numeric
        training columns left out because synthetic data lacks them or stores
        them as non-numeric values.
    """
    if value_columns is not None:
        invalid = invalid_value_columns(datasets, value_columns, timestamp_column, group_column)
        if invalid:
            return [], f"Invalid {label} value columns: " + "; ".join(invalid), []
        return list(dict.fromkeys(value_columns)), None, []
    reserved = {timestamp_column, group_column, PSEUDO_GROUP_COLUMN}
    candidates = sorted(
        column
        for column in datasets.get_columns_of_type({FieldType.NUMERIC}, based_on="training")
        if column not in reserved
    )
    columns: list[str] = []
    excluded: list[str] = []
    for column in candidates:
        if column in datasets.synthetic and pd.api.types.is_numeric_dtype(datasets.synthetic[column]):
            columns.append(column)
        else:
            excluded.append(column)
    return columns, None, excluded


def invalid_value_columns(
    datasets: EvaluationDatasets,
    columns: list[str],
    timestamp_column: str | None,
    group_column: str | None,
) -> list[str]:
    """Return actionable validation failures for explicit value columns."""
    invalid: list[str] = []
    reserved = {timestamp_column, group_column, PSEUDO_GROUP_COLUMN}
    for column in columns:
        if column in reserved:
            invalid.append(f"{column!r} is used for timestamp or sequence grouping")
            continue
        missing_from = [
            name
            for name, frame in (("training", datasets.training), ("synthetic", datasets.synthetic))
            if column not in frame
        ]
        if missing_from:
            invalid.append(f"{column!r} is missing from {' and '.join(missing_from)} data")
            continue
        if not pd.api.types.is_numeric_dtype(datasets.training[column]):
            invalid.append(f"{column!r} is not numeric in training data")
            continue
        if not pd.api.types.is_numeric_dtype(datasets.synthetic[column]):
            invalid.append(f"{column!r} is not numeric in synthetic data")
    return invalid


def shared_groups(
    training: pd.DataFrame,
    synthetic: pd.DataFrame,
    group_column: str | None,
    max_groups: int,
    seed: int,
) -> tuple[list[Any], list[str], list[str], int]:
    """Find a deterministic, bounded set of groups shared by both datasets.

    A ``None`` sentinel represents one global sequence when grouping is not
    configured. Group labels present in only one dataset are returned for
    diagnostics but do not contribute to the score.

    Args:
        training: Training records containing candidate group labels.
        synthetic: Synthetic records containing candidate group labels.
        group_column: Optional column that identifies independent sequences.
        max_groups: Maximum number of shared groups to evaluate.
        seed: Seed for the reproducible sample taken above ``max_groups``.

    Returns:
        Shared group keys, training-only labels, synthetic-only labels,
        and the total number of shared groups before limiting.
    """
    if group_column is None:
        return [None], [], [], 1
    if group_column not in training or group_column not in synthetic:
        return [], [], [], 0
    training_groups = set(training[group_column].dropna().unique())
    synthetic_groups = set(synthetic[group_column].dropna().unique())
    ordered_shared = sorted(training_groups & synthetic_groups, key=group_sort_key)
    shared = ordered_shared
    if len(ordered_shared) > max_groups:
        shared = random.Random(seed).sample(ordered_shared, k=max_groups)
        shared.sort(key=group_sort_key)
    only_training = [str(value) for value in sorted(training_groups - synthetic_groups, key=str)]
    only_synthetic = [str(value) for value in sorted(synthetic_groups - training_groups, key=str)]
    return shared, only_training, only_synthetic, len(ordered_shared)


def group_sort_key(value: Any) -> tuple[str, str]:
    """Return a stable ordering key for heterogeneous group values."""
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    return type_name, str(value)


def group_frame(
    df: pd.DataFrame,
    group_column: str | None,
    group_value: Any,
    timestamp_column: str | None,
) -> pd.DataFrame:
    """Return one sequence in deterministic timestamp order.

    ``mergesort`` preserves input order for equal timestamps, which makes
    repeated runs stable without inventing a secondary ordering key.

    Args:
        df: Dataset containing one or more sequences.
        group_column: Optional sequence identifier column.
        group_value: Group to select, or ``None`` for the full dataset.
        timestamp_column: Optional column used to order the selected rows.

    Returns:
        A filtered and time-ordered DataFrame view or copy.
    """
    frame = df if group_column is None else df[df[group_column] == group_value]
    if timestamp_column and timestamp_column in frame:
        frame = frame.sort_values(timestamp_column, kind="mergesort")
    return frame


def summaries(profiles: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Average profile similarities by a diagnostic key.

    Args:
        profiles: Successful group-and-column comparison details.
        key: Detail key to group by, such as ``group`` or ``column``.

    Returns:
        Deterministically ordered summaries with similarity and count.
    """
    scores: defaultdict[Any, list[float]] = defaultdict(list)
    for item in profiles:
        scores[item[key]].append(item["similarity"])
    return [
        {key: value, "similarity": float(np.mean(values)), "count": len(values)}
        for value, values in sorted(scores.items(), key=lambda item: str(item[0]))
    ]
