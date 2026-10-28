# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular PII replacement interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.data import DataParameters
from ..config.replace_pii import ReplacePiiConfig
from ..config.time_series import TimeSeriesParameters
from .transform_result import TransformResult

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["TabularPiiReplacer"]


class TabularPiiReplacer:
    """Replace PII in a dataframe through one plan-driven interface.

    The replacement module owns plan resolution, DAG execution, free-text span
    detection and rewriting, positional row identity, scope keys, synthetic
    value generation, and statistics. ``replace`` returns a new dataframe and
    never mutates the caller's frame or writes artifacts. The pipeline remains
    responsible for deciding whether and where to persist the resolved
    configuration.

    Replacement execution is intentionally deferred from this interface-only
    implementation.

    Args:
        config: PII replacement configuration, including the replacement plan.
        data_config: Input data configuration used to validate and execute the
            resolved plan.
        time_series: Optional time-series configuration used to protect ordering
            and grouping columns.
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

    def replace(self, df: pd.DataFrame, *, capture_replacement_map: bool = False) -> TransformResult:
        """Return a replacement result for ``df`` without mutating ``df``.

        Args:
            df: Dataframe whose planned PII values should be replaced.
            capture_replacement_map: Include sensitive per-occurrence replacement
                provenance and accepted free-text span traces in the result. The
                default leaves this data unavailable so ordinary calls do not
                accidentally persist original PII.

        Raises:
            NotImplementedError: Always in the interface-definition PR because
                replacement execution is introduced by a follow-up change.
        """
        raise NotImplementedError("TabularPiiReplacer execution is not implemented")
