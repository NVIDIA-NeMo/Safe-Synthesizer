# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular PII replacement interface."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..config.data import DataParameters
from ..config.replace_pii import PiiSamplerBackend, ReplacePiiConfig
from ..config.time_series import TimeSeriesParameters
from ..errors import InternalError
from .planning import resolve_replacement_config
from .replacement.executor import StructuredReplacementExecutor, resolve_base_seed
from .replacement.generation import ReplacementGenerator
from .replacement.generators import FakerReplacementGenerator, ManagedReplacementGenerator
from .transform_result import TransformResult

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["TabularPiiReplacer"]


class TabularPiiReplacer:
    """Replace PII in a dataframe through one plan-driven interface.

    The replacement module owns plan resolution and structured DAG execution,
    positional row identity, scope keys, synthetic value generation, and
    statistics. Free-text detection and span replacement are deferred.
    ``replace`` returns a new dataframe and never mutates the caller's frame or
    writes artifacts. The pipeline remains responsible for deciding whether
    and where to persist the resolved configuration.

    Args:
        config: PII replacement configuration, including the plan and mapping sources.
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

    def replace(self, df: pd.DataFrame) -> TransformResult:
        """Return a replacement result for ``df`` without mutating ``df``.

        Raises:
            ParameterError: If the configured plan is invalid for ``df``.
            GenerationError: If a replacement cannot be generated.
        """
        started = time.perf_counter()
        resolved_config = resolve_replacement_config(
            df,
            self._config,
            self._data_config,
            self._time_series,
        )
        plan = resolved_config.inline_plan
        mappings = resolved_config.sampler.inline_dependency_value_mappings
        if plan is None or mappings is None:
            raise InternalError("PII replacement configuration was not fully resolved")
        executor = StructuredReplacementExecutor(
            plan,
            self._replacement_generator(resolved_config),
            group_column=self._data_config.group_training_examples_by,
            base_seed=resolve_base_seed(self._config.replacement.seed),
            dependency_value_mappings=mappings,
        )
        execution = executor.execute(df)
        return TransformResult(
            transformed_df=execution.dataframe,
            column_statistics=execution.column_statistics,
            replacement_plan=plan,
            resolved_config=resolved_config,
            generation_statistics=execution.generation_statistics,
            elapsed_time_seconds=time.perf_counter() - started,
        )

    def _replacement_generator(self, config: ReplacePiiConfig) -> ReplacementGenerator:
        settings = config.replacement
        sampler = config.sampler
        match sampler.backend:
            case PiiSamplerBackend.MANAGED:
                return ManagedReplacementGenerator(settings=settings, sampler=sampler)
            case PiiSamplerBackend.FAKER:
                return FakerReplacementGenerator(settings=settings, sampler=sampler)
        raise InternalError(f"Unsupported PII sampler backend: {sampler.backend!r}")
