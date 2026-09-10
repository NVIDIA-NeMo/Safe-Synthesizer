# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular PII replacement interface."""

from __future__ import annotations

import pandas as pd

from ..config.data import DataParameters
from ..config.replace_pii import ReplacePiiConfig
from ..config.time_series import TimeSeriesParameters
from .transform_result import TransformResult

__all__ = ["TabularPiiReplacer"]


class TabularPiiReplacer:
    """Replace PII in a dataframe through one plan-driven interface.

    The replacement module owns plan resolution and DAG execution, positional
    row identity, scope keys, synthetic value generation, free-text detection,
    overlap handling, and statistics. ``replace`` returns a new dataframe and
    never mutates the caller's frame or writes artifacts. The pipeline remains
    responsible for deciding whether and where to persist the resolved plan.

    Replacement execution is intentionally deferred from this interface-only
    implementation.
    """

    def __init__(
        self,
        config: ReplacePiiConfig,
        *,
        data_config: DataParameters,
        time_series: TimeSeriesParameters | None = None,
    ) -> None:
        self._config = config
        self._data_config = data_config
        self._time_series = time_series

    def replace(self, df: pd.DataFrame) -> TransformResult:
        """Return a replacement result for ``df`` without mutating ``df``.

        Raises:
            NotImplementedError: Always in the interface-definition PR because
                replacement execution is introduced by a follow-up change.
        """
        raise NotImplementedError("TabularPiiReplacer execution is not implemented")
