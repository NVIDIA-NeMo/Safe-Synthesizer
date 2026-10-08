# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time series preprocessing utilities for Safe Synthesizer training."""

from __future__ import annotations

import pandas as pd

from ..config import SafeSynthesizerParameters
from ..data_processing.flexible_timeseries import prepare_flexible_timeseries_data
from ..data_processing.timeseries_validation import (
    resolve_timeseries_routing,
    validate_deterministic_inspection,
)
from ..llm.metadata import TimeseriesMetadata
from ..observability import get_logger

logger = get_logger(__name__)


def _reorder_timeseries_columns(
    dataframe: pd.DataFrame,
    group_by_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """Put time-series identity columns first in the persisted schema order.

    The resulting order is shared by the prompt schema, training JSONL, and
    partial-record generation prefix. The pseudo-group remains internal and is
    excluded later when the persisted schema is built.

    Args:
        dataframe: Validated and chronologically sorted training data.
        group_by_column: Real or pseudo group column.
        timestamp_column: Resolved timestamp column.

    Returns:
        A view with group and timestamp columns first, followed by all remaining
        columns in their original relative order.
    """
    leading_columns = [group_by_column, timestamp_column]
    remaining_columns = [column for column in dataframe.columns if column not in leading_columns]
    return dataframe.loc[:, [*leading_columns, *remaining_columns]]


def process_timeseries_data(
    training_df: pd.DataFrame,
    config: SafeSynthesizerParameters,
) -> tuple[pd.DataFrame, SafeSynthesizerParameters, TimeseriesMetadata | None]:
    """Resolve and prepare deterministic or flexible time-series training data.

    Normalizes grouped and ungrouped time series into the same training path.
    When no group column is configured, a reserved pseudo-group column
    (``PSEUDO_GROUP_COLUMN``) is added so the whole dataset is treated as one
    sequence. Fixed-shape groups retain deterministic time-range processing.
    Groups with different lengths, ranges, or unasserted intervals are
    automatically transformed to use a generated sequence index and final-row
    marker. The passed configuration is updated with the selected
    representation and resolved timestamp metadata.

    Args:
        training_df: The training DataFrame.
        config: Configuration containing time-series and data settings.

    Returns:
        Processed training data, the resolved configuration, and the
        time-series metadata to persist with the model, or ``None`` for
        non-time-series data.

    Raises:
        ParameterError: If a configured timestamp or ordering column is missing,
            or if the timestamp format is incompatible with the source data.
        DataError: If required source values are null, timestamps cannot be
            parsed, or timestamps do not follow an asserted interval.
    """
    routing = resolve_timeseries_routing(training_df, config)
    if routing is None:
        return training_df, config, None

    ts_config = config.time_series
    metadata = routing.timeseries_metadata
    if metadata is not None and metadata.flexible is not None:
        training_df, group_column = prepare_flexible_timeseries_data(
            training_df,
            config,
            metadata.flexible,
            routing.timestamp_format,
        )
        logger.info(
            "Time-series groups differ in shape; using flexible time-series processing.",
            extra={
                "group_column": group_column,
                "failed_constraints": list(routing.failed_constraints),
                "sequence_max_records": metadata.flexible.max_records,
            },
        )
        return training_df, config, metadata

    logger.info("Time-series groups share one shape; using deterministic time-range processing.")

    original_group_column = config.data.group_training_examples_by
    original_timestamp_column = ts_config.timestamp_column
    validation = validate_deterministic_inspection(
        routing.inspection,
        ts_config.timestamp_interval_seconds,
    )
    training_df = validation.data
    if original_group_column is None:
        logger.info("No group column specified, treating entire dataset as a single sequence")
    if original_timestamp_column is None:
        logger.info(f"Added timestamp column '{validation.timestamp_column}' with elapsed seconds")
    config.data.group_training_examples_by = validation.group_by_column
    config.data.order_training_examples_by = validation.timestamp_column
    ts_config.timestamp_column = validation.timestamp_column
    ts_config.timestamp_format = validation.timestamp_format
    ts_config.timestamp_interval_seconds = validation.timestamp_interval_seconds
    ts_config.start_timestamp = validation.start_timestamp
    ts_config.stop_timestamp = validation.stop_timestamp
    is_elapsed_time = validation.is_elapsed_time
    logger.info(f"Resolved time-series timestamp format: {validation.timestamp_format}")
    if validation.timestamp_interval_seconds is not None:
        logger.info(f"Resolved timestamp_interval_seconds: {validation.timestamp_interval_seconds}s")
    logger.info(
        f"Time series range (consistent across {len(validation.group_stats)} groups): "
        f"{validation.start_timestamp} to {validation.stop_timestamp}",
    )

    # Step 7: Convert timestamp back to string format
    # Skip string conversion for elapsed_seconds format (values are already numeric)
    if (
        not is_elapsed_time
        and ts_config.timestamp_format is not None
        and ts_config.timestamp_format != "elapsed_seconds"
    ):
        training_df[ts_config.timestamp_column] = training_df[ts_config.timestamp_column].dt.strftime(
            ts_config.timestamp_format
        )

    training_df = _reorder_timeseries_columns(
        training_df,
        validation.group_by_column,
        validation.timestamp_column,
    )
    return training_df, config, metadata
