# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import (
    Annotated,
)

from pydantic import (
    Field,
)

from ..configurator.parameters import (
    Parameters,
)

__all__ = [
    "AutocorrelationSimilarityParameters",
    "EvaluationParameters",
    "TimeSeriesEvaluationParameters",
]

DEFAULT_SQS_REPORT_COLUMNS: int = 250
DEFAULT_RECORD_COUNT = 5000
QUASI_IDENTIFIER_COUNT = 3


class AutocorrelationSimilarityParameters(Parameters):
    """Control autocorrelation similarity evaluation.

    Timestamp ordering always uses the top-level time-series setting. The
    optional grouping override inherits the top-level data grouping when left
    unset. The requested lag is capped automatically for short sequences, and
    undersized or constant training profiles are skipped. A constant synthetic
    profile paired with varying training data receives zero similarity.
    """

    enabled: bool | None = Field(
        default=None,
        description="Enable this metric; None enables it automatically for time-series data.",
    )
    value_columns: list[str] | None = Field(
        default=None,
        description="Numeric value columns to evaluate. Defaults to all shared numeric columns.",
    )
    group_column: str | None = Field(
        default=None,
        description="Optional group column overriding the top-level data grouping column.",
    )
    max_lag: int = Field(
        default=20,
        ge=1,
        description="Maximum requested lag; short sequences use a smaller stable lag cap.",
    )
    min_points: int = Field(
        default=4,
        ge=4,
        description="Minimum finite observations required in each sequence.",
    )
    max_groups: int = Field(
        default=128,
        ge=1,
        description="Maximum shared groups to evaluate in deterministic label order.",
    )


class TimeSeriesEvaluationParameters(Parameters):
    """Metric-specific time-series evaluation configuration."""

    autocorrelation: AutocorrelationSimilarityParameters = Field(
        default_factory=AutocorrelationSimilarityParameters,
        description="Autocorrelation similarity metric parameters.",
    )


class EvaluationParameters(Parameters):
    """Configuration for evaluating synthetic data quality and privacy.

    This class controls which evaluation metrics are computed and how they are configured.
    It includes privacy attack evaluations, statistical quality metrics, and downstream
    machine learning performance assessments.
    """

    mia_enabled: Annotated[
        bool,
        Field(
            title="mia_enabled",
            description="Enable membership inference attack evaluation for privacy assessment.",
        ),
    ] = True

    aia_enabled: Annotated[
        bool,
        Field(
            title="aia_enabled",
            description="Enable attribute inference attack evaluation for privacy assessment.",
        ),
    ] = True

    sqs_report_columns: int = Field(
        default=DEFAULT_SQS_REPORT_COLUMNS,
        description="Number of columns to include in statistical quality reports.",
    )

    sqs_report_rows: int = Field(
        default=DEFAULT_RECORD_COUNT,
        description="Number of rows to include in statistical quality reports.",
    )

    mandatory_columns: Annotated[
        int | None,
        Field(title="mandatory_columns", description="Number of mandatory columns that must be used in evaluation."),
    ] = None

    enabled: Annotated[
        bool,
        Field(
            title="enabled",
            description="Enable or disable evaluation.",
        ),
    ] = True

    quasi_identifier_count: Annotated[
        int,
        Field(
            description="Number of quasi-identifiers to sample for privacy attacks.",
        ),
    ] = QUASI_IDENTIFIER_COUNT

    pii_replay_enabled: Annotated[
        bool,
        Field(
            title="pii_replay_enabled",
            description="Enable PII Replay detection.",
        ),
    ] = True

    pii_replay_entities: Annotated[
        list[str] | None,
        Field(
            description="List of entities for PII Replay. If not provided, default entities will be used.",
        ),
    ] = None

    pii_replay_columns: Annotated[
        list[str] | None,
        Field(
            description="List of columns for PII Replay. If not provided, only entities will be used.",
        ),
    ] = None

    time_series: TimeSeriesEvaluationParameters = Field(
        default_factory=TimeSeriesEvaluationParameters,
        description="Time-series-specific evaluation settings.",
    )
