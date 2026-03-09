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

__all__ = ["EvaluationParameters"]

DEFAULT_SQS_REPORT_COLUMNS: int = 250
DEFAULT_RECORD_COUNT = 5000
QUASI_IDENTIFIER_COUNT = 3


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
