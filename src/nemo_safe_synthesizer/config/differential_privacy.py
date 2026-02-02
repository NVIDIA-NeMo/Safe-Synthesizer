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
from ..configurator.validators import (
    ValueValidator,
)
from .types import (
    AUTO_STR,
    AutoFloatParam,
)

__all__ = [
    "DifferentialPrivacyHyperparams",
]


class DifferentialPrivacyHyperparams(Parameters):
    """Hyperparameters for differential privacy during training.

    These parameters configure differential privacy (DP) training using DP-SGD algorithm.
    When enabled, they provide formal privacy guarantees by adding calibrated noise
    during training.

    Attributes:
        dp_enabled: Enable differential privacy training with DP-SGD algorithm.
        epsilon: Target privacy budget (ε) - lower values provide stronger privacy.
        delta: Probability of privacy failure (δ) - should be much smaller than 1/n
            where n is the number of training records.
        per_sample_max_grad_norm: Maximum L2 norm for gradient clipping per sample.

    """

    dp_enabled: Annotated[
        bool,
        Field(
            title="dp_enabled",
            description="Enable differentially-private training with DP-SGD.",
        ),
    ] = False

    epsilon: Annotated[
        float,
        ValueValidator(value_func=lambda v: v > 0),
        Field(
            title="epsilon",
            description="Target for epsilon when training completes.",
        ),
    ] = 8.0

    delta: Annotated[
        AutoFloatParam,
        ValueValidator(value_func=lambda v: True if v == AUTO_STR else (0 <= v < 1)),
        Field(
            title="delta",
            description=(
                "Probability of accidentally leaking information. Setting to 'auto' uses"
                "delta of 1/n^1.2, where n is the number of training records"
            ),
        ),
    ] = AUTO_STR

    per_sample_max_grad_norm: Annotated[
        float,
        ValueValidator(value_func=lambda v: v > 0),
        Field(
            title="per_sample_max_grad_norm",
            description="Maximum L2 norm of per sample gradients.",
        ),
    ] = 1.0
