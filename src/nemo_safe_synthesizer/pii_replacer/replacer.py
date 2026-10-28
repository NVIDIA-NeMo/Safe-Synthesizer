# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular PII replacement interface."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..config.data import DataParameters
from ..config.replace_pii import (
    EntityType,
    PiiReplacementPlan,
    PiiSamplerBackend,
    PiiSamplerConfig,
    ReplacePiiConfig,
)
from ..config.time_series import TimeSeriesParameters
from ..errors import InternalError
from ..observability import get_logger
from .planning import resolve_replacement_config
from .replacement.detection import CompositeFreeTextDetector
from .replacement.executor import StructuredReplacementExecutor, resolve_base_seed
from .replacement.generation import ReplacementGenerator
from .transform_result import TransformResult

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["TabularPiiReplacer"]

logger = get_logger(__name__)

# find_and_replace skips known values that are common English words, so it runs only for this locale.
_FIND_AND_REPLACE_LOCALE = "en_US"


class TabularPiiReplacer:
    """Replace PII in a dataframe through one plan-driven interface.

    The replacement module owns plan resolution, DAG execution, free-text span
    detection and rewriting, positional row identity, scope keys, synthetic
    value generation, and statistics. ``replace`` returns a new dataframe and
    never mutates the caller's frame or writes artifacts. The pipeline remains
    responsible for deciding whether and where to persist the resolved
    configuration.

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

    def replace(
        self,
        df: pd.DataFrame,
        *,
        capture_replacement_map: bool = False,
        check_pattern_coverage: bool = True,
    ) -> TransformResult:
        """Return a replacement result for ``df`` without mutating ``df``.

        Args:
            df: Dataframe whose planned PII values should be replaced.
            capture_replacement_map: Include sensitive per-occurrence replacement
                provenance and accepted free-text span traces in the result. The
                default leaves this data unavailable so ordinary calls do not
                accidentally persist original PII.
            check_pattern_coverage: Require each pattern to cover at least 99%
                of ``df``'s values. Pass ``False`` when ``df`` is a subset of
                data the plan was already resolved against, such as the
                pipeline's training split, so a subset that falls just below
                the threshold does not reject a plan the full input accepted.

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
            check_pattern_coverage=check_pattern_coverage,
        )
        plan = resolved_config.inline_plan
        if plan is None:
            raise InternalError("PII replacement configuration was not fully resolved")
        # Street-address columns also use the detector, to find the street inside a full address.
        free_text_detector = (
            CompositeFreeTextDetector(resolved_config.free_text_detection)
            if any(
                spec.entity_type in {EntityType.FREE_TEXT, EntityType.STREET_ADDRESS}
                for spec in plan.columns_to_replace
            )
            else None
        )
        executor = StructuredReplacementExecutor(
            plan,
            self._replacement_generator(resolved_config.sampler, resolved_config.locale),
            group_column=self._data_config.group_training_examples_by,
            base_seed=resolve_base_seed(resolved_config.seed),
            free_text_detector=free_text_detector,
            capture_replacement_map=capture_replacement_map,
            find_and_replace=_find_and_replace_enabled(resolved_config, plan),
        )
        execution = executor.execute(df)
        return TransformResult(
            transformed_df=execution.dataframe,
            column_statistics=execution.column_statistics,
            resolved_config=resolved_config,
            generation_statistics=execution.generation_statistics,
            replacement_time_seconds=time.perf_counter() - started,
            replacement_map=execution.replacement_map,
        )

    def _replacement_generator(self, sampler: PiiSamplerConfig, locale: str) -> ReplacementGenerator:
        # Imported here so loading the package does not load generator implementations.
        from .replacement.generators import FakerReplacementGenerator, NemotronPersonasReplacementGenerator

        match sampler.backend:
            case PiiSamplerBackend.NEMOTRON_PERSONAS:
                return NemotronPersonasReplacementGenerator(sampler=sampler, locale=locale)
            case PiiSamplerBackend.FAKER:
                return FakerReplacementGenerator(sampler=sampler, locale=locale)
        raise InternalError(f"Unsupported PII sampler backend: {sampler.backend!r}")


def _find_and_replace_enabled(config: ReplacePiiConfig, plan: PiiReplacementPlan) -> bool:
    """Return whether to search free text for known values, warning when the locale turns the search off."""
    if not config.free_text_detection.find_and_replace:
        return False
    if config.locale == _FIND_AND_REPLACE_LOCALE:
        return True
    if any(spec.entity_type is EntityType.FREE_TEXT for spec in plan.columns_to_replace):
        logger.user.warning(
            f"Skipping free_text_detection.find_and_replace for locale {config.locale!r}: it relies on a list of "
            f"common English words and runs only for {_FIND_AND_REPLACE_LOCALE}. Detected free-text spans are "
            "still replaced."
        )
    return False
