# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import (
    Annotated,
)

from pydantic import (
    Field,
    field_validator,
)

from ..configurator.parameters import (
    Parameters,
)
from ..configurator.validators import (
    DependsOnValidator,
)
from .types import (
    AUTO_STR,
    OptionalAutoInt,
)

__all__ = [
    "DataParameters",
]

# Holdout constants
DEFAULT_HOLDOUT = 0.05
MIN_HOLDOUT = 10


class DataParameters(Parameters):
    """Configuration for grouping, ordering, and splitting input data for training and evaluation."""

    group_training_examples_by: Annotated[
        str | None,
        Field(
            description=(
                "Column to group training examples by. This is useful when you want the model to "
                "learn inter-record correlations for a given grouping of records."
            ),
        ),
    ] = None

    order_training_examples_by: Annotated[
        str | None,
        DependsOnValidator(
            depends_on="group_training_examples_by",
            depends_on_func=lambda v: v is not None,
            value_func=lambda v: v is not None,
            message=(
                "order_training_examples_by requires group_training_examples_by. "
                "In time-series mode (time_series.is_timeseries=true) with a single series, "
                "set time_series.timestamp_column instead; records are grouped automatically."
            ),
        ),
        Field(
            description=(
                "Column to order training examples by. This is useful when you want the model to "
                "learn sequential relationships for a given ordering of records. If you provide this "
                "parameter, you must also provide ``group_training_examples_by``."
            ),
        ),
    ] = None

    max_sequences_per_example: Annotated[
        OptionalAutoInt,
        Field(
            description=(
                "If specified, adds at most this number of sequences per example. "
                "Supports 'auto', which resolves to 1 when differential privacy is "
                "enabled, None for time-series mode (each example fills the context "
                "window), and 10 otherwise. If set to None, fills up the context window. "
                "Required for DP to limit contribution of each example."
            ),
        ),
    ] = AUTO_STR

    holdout: float = Field(
        default=DEFAULT_HOLDOUT,
        description=(
            "Fraction of records to hold out for evaluation. With grouped data, the same fraction of groups is "
            "held out, rounded up to whole groups. Set to 0 to turn off the holdout. Must be in [0, 1)."
        ),
    )

    random_state: Annotated[
        int | None,
        Field(
            description="Random state for holdout split to ensure reproducibility.",
        ),
    ] = None

    @field_validator("holdout", mode="after")
    @classmethod
    def check_holdout_fraction(cls, v: float) -> float:
        """Require a holdout fraction in [0, 1)."""
        if not 0 <= v < 1:
            raise ValueError(
                f"data.holdout must be a fraction in [0, 1), got {v}. Holdout row counts are no longer supported. "
                "Use a fraction such as 0.05, or 0 to turn off the holdout."
            )
        return v

    @field_validator("random_state", mode="after", check_fields=False)
    def set_random_state_if_none(cls, v: int | int | None) -> int | None:
        """Generate a random state if none was provided."""
        import random

        if v is None:
            return random.randint(0, 1000000)
        return v
