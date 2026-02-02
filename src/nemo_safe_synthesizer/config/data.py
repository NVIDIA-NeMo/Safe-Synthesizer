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
    ValueValidator,
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
DEFAULT_MAX_HOLDOUT = 2000
MIN_HOLDOUT = 10


class DataParameters(Parameters):
    """Configuration for parameters related to how to shape or use the data being passed in.

    Attributes:
        group_training_examples_by: Column to group training examples by.
        order_training_examples_by: Column to order training examples by.
        max_sequences_per_example: Maximum number of sequences per example for training.
        holdout: Amount of records to holdout for evaluation.
        max_holdout: Maximum number of records to hold out. Overrides any behavior set by holdout parameter.
        random_state: Random state for holdout split to ensure reproducibility.
    """

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
        ),
        Field(
            description=(
                "Column to order training examples by. This is useful when you want the model to "
                "learn sequential relationships for a given ordering of records. If you provide this "
                "parameter, you must also provide `group_training_examples_by`."
            ),
        ),
    ] = None

    max_sequences_per_example: Annotated[
        OptionalAutoInt,
        Field(
            description=(
                "If specified, adds at most this number of sequences per example; "
                "otherwise, fills up context. Supports 'auto' where a value of 1 is "
                "chosen if differential privacy is enabled, and None otherwise. "
                "Required for DP to limit contribution of each example."
            ),
        ),
    ] = AUTO_STR

    holdout: Annotated[
        float,
        ValueValidator(value_func=lambda v: v >= 0),
        Field(
            description=(
                "Amount of records to holdout. If this is a float between 0 and 1, that ratio of "
                "records is held out. If an integer greater than 1, that number of records is held out."
                "If the value is equal to zero, no holdout will be performed."
            ),
        ),
    ] = DEFAULT_HOLDOUT

    max_holdout: Annotated[
        int,
        ValueValidator(value_func=lambda v: v >= 0),
        Field(
            description="Maximum number of records to hold out. Overrides any behavior set by holdout parameter.",
        ),
    ] = DEFAULT_MAX_HOLDOUT

    random_state: Annotated[
        int | None,
        Field(
            description="Use this random state for holdout split to ensure reproducibility.",
        ),
    ] = None

    @field_validator("random_state", mode="after", check_fields=False)
    def set_random_state_if_none(cls, v: int | int | None) -> int | None:
        import random

        if v is None:
            return random.randint(0, 1000000)
        return v
