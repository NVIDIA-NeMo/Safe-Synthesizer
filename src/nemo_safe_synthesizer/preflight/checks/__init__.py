# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core pre-flight check implementations.

The aggregated ``_CORE_CHECKS`` tuple is what ``registry.build_registry``
uses to seed the registry returned by ``get_registry()``. Add a new core check by
implementing it in the stage-matching submodule and appending it here in
the order you want it to run within its stage block.
"""

from __future__ import annotations

from ..base import PreflightCheck
from .advisory import (
    OversamplingCheck,
    SmallDatasetCheck,
)
from .dataframe import (
    ConstantColumnCheck,
    DatasetSizeCheck,
    GroupbyColumnCheck,
    OrderbyColumnCheck,
    PseudoColumnCheck,
    TimeSeriesDataShapeCheck,
    TimestampColumnCheck,
)
from .environment import (
    CUDAAvailabilityCheck,
    HFModelAvailabilityCheck,
    InferenceModelCheck,
    VRAMHeadroomCheck,
)
from .metadata import TokenBudgetCheck

__all__ = [
    "CUDAAvailabilityCheck",
    "ConstantColumnCheck",
    "SmallDatasetCheck",
    "DatasetSizeCheck",
    "GroupbyColumnCheck",
    "HFModelAvailabilityCheck",
    "InferenceModelCheck",
    "OrderbyColumnCheck",
    "OversamplingCheck",
    "PseudoColumnCheck",
    "TimeSeriesDataShapeCheck",
    "TimestampColumnCheck",
    "TokenBudgetCheck",
    "VRAMHeadroomCheck",
    "_CORE_CHECKS",
]


# Tuple (not list) because the core check set is intentionally immutable:
# plugin registration extends the built ``PreflightRegistry``, not this
# constant. Mutating it in-process would silently desync the registry from
# what ``registry.build_registry`` originally seeded.
_CORE_CHECKS: tuple[PreflightCheck, ...] = (
    # CONFIG
    CUDAAvailabilityCheck(),
    InferenceModelCheck(),
    HFModelAvailabilityCheck(),
    # DATAFRAME
    DatasetSizeCheck(),
    GroupbyColumnCheck(),
    OrderbyColumnCheck(),
    PseudoColumnCheck(),
    ConstantColumnCheck(),
    TimestampColumnCheck(),
    TimeSeriesDataShapeCheck(),
    # METADATA
    VRAMHeadroomCheck(),
    TokenBudgetCheck(),
    # ADVISORY
    SmallDatasetCheck(),
    OversamplingCheck(),
)
