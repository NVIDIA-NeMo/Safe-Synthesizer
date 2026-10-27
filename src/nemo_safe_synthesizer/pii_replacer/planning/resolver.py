# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve one validated PII replacement plan from configuration and data."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from ...config.data import DataParameters
from ...config.replace_pii import (
    DataToSamplerValueMapping,
    EntityType,
    LLMConfig,
    PiiReplacementPlan,
    PiiSamplerBackend,
    ReplacePiiConfig,
)
from ...config.time_series import TimeSeriesParameters
from ...errors import ParameterError
from ...observability import get_logger
from .dependency_mappings import (
    built_in_value_mappings,
    conditioner_columns,
    mapping_inputs,
    unmatched_dependency_values,
    validate_data_to_sampler_value_mapping,
)
from .validation import get_protected_columns, validate_plan

logger = get_logger(__name__)

__all__ = [
    "ColumnProfile",
    "HeuristicPlanDiscoverer",
    "PlanDiscoverer",
    "PlanDiscoveryInput",
    "PlanEnhancer",
    "resolve_replacement_config",
    "resolve_plan",
]

MAX_PROFILE_SAMPLES = 8
MAX_PROFILE_SAMPLE_LENGTH = 128


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Bounded descriptive evidence about one column for plan discovery."""

    column_name: str
    dtype: str
    non_null_count: int
    unique_count: int
    unique_ratio: float
    samples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanDiscoveryInput:
    """Data and deterministic preparation supplied to discovery adapters.

    ``group_column`` is the runtime consistency signal: a configured column
    means replacements remain consistent within each group; ``None`` means
    each record is replaced independently. This derived choice is intentionally
    not part of the reusable replacement plan.
    """

    dataframe: pd.DataFrame
    group_column: str | None
    protected_columns: frozenset[str]
    column_profiles: tuple[ColumnProfile, ...]


class PlanDiscoverer(ABC):
    """Internal seam for producing the heuristic baseline plan."""

    @abstractmethod
    def discover(self, discovery_input: PlanDiscoveryInput) -> PiiReplacementPlan:
        """Return a context-free valid baseline plan."""


class PlanEnhancer(ABC):
    """Internal seam for revising a heuristic baseline plan."""

    @abstractmethod
    def enhance(
        self,
        discovery_input: PlanDiscoveryInput,
        baseline: PiiReplacementPlan,
    ) -> PiiReplacementPlan:
        """Return a context-free valid replacement for ``baseline``."""


class DependencyMappingDiscoverer(Protocol):
    """Discover sampler-specific mappings for dependency-column values."""

    def discover_data_to_sampler_value_mapping(
        self,
        dataframe: pd.DataFrame,
        plan: PiiReplacementPlan,
        catalog: Mapping[EntityType, tuple[str, ...]],
    ) -> DataToSamplerValueMapping:
        """Return sparse mappings for non-identity dependency values."""


class HeuristicPlanDiscoverer(PlanDiscoverer):
    """Initial no-op heuristic adapter, ready for later rule discovery."""

    def discover(self, discovery_input: PlanDiscoveryInput) -> PiiReplacementPlan:
        """Return an empty plan for later heuristic discovery."""
        return PiiReplacementPlan()


def _stable_samples(series: pd.Series) -> tuple[str, ...]:
    """Pick a bounded set of distinct raw column values for discovery.

    Nulls are removed, values are converted to truncated strings, and duplicate
    strings are collapsed. Content hashes assign each distinct value a stable
    priority, so reordering dataframe rows does not change which values are
    selected. The selected values are returned in dataframe order.
    """
    values = list(dict.fromkeys(str(value)[:MAX_PROFILE_SAMPLE_LENGTH] for value in series.dropna().tolist()))
    # Rank distinct values by content to choose a row-order-independent subset,
    # then present that subset in first-occurrence dataframe order. The hashes
    # select values deterministically; they do not anonymize the returned text.
    selected = set(sorted(values, key=lambda value: hashlib.sha256(value.encode()).digest())[:MAX_PROFILE_SAMPLES])
    return tuple(value for value in values if value in selected)


def _profile_columns(df: pd.DataFrame) -> tuple[ColumnProfile, ...]:
    profiles: list[ColumnProfile] = []
    for column in df.columns:
        non_null = df[column].dropna().astype(str)
        non_null_count = len(non_null)
        unique_count = int(non_null.nunique(dropna=True))
        profiles.append(
            ColumnProfile(
                column_name=column,
                dtype=str(df[column].dtype),
                non_null_count=non_null_count,
                unique_count=unique_count,
                unique_ratio=unique_count / non_null_count if non_null_count else 0.0,
                samples=_stable_samples(df[column]),
            )
        )
    return tuple(profiles)


def _prepare_discovery_input(
    df: pd.DataFrame,
    data_config: DataParameters,
    time_series: TimeSeriesParameters | None,
) -> PlanDiscoveryInput:
    group_column = data_config.group_training_examples_by
    protected_columns = get_protected_columns(data_config, time_series)
    return PlanDiscoveryInput(
        dataframe=df,
        group_column=group_column,
        protected_columns=protected_columns,
        column_profiles=_profile_columns(df),
    )


def _enhance_with_default_llm(
    llm_config: LLMConfig,
    discovery_input: PlanDiscoveryInput,
    baseline: PiiReplacementPlan,
) -> PiiReplacementPlan:
    """Run the default LLM enhancer inside its inference environment.

    A managed local server, the default without ``NSS_INFERENCE_ENDPOINT``,
    stops before this returns, so validation, persistence, replacement, and
    training never run while it holds the GPU.
    """
    from ..local_inference import planning_inference_settings
    from .llm import LLMPlanEnhancer

    with planning_inference_settings(llm_config) as settings:
        return LLMPlanEnhancer(llm_config, settings=settings).enhance(discovery_input, baseline)


def resolve_plan(
    df: pd.DataFrame,
    config: ReplacePiiConfig,
    data_config: DataParameters,
    time_series: TimeSeriesParameters | None = None,
    *,
    discoverer: PlanDiscoverer | None = None,
    enhancer: PlanEnhancer | None = None,
    check_pattern_coverage: bool = True,
) -> PiiReplacementPlan:
    """Resolve and validate one replacement plan.

    An inline plan is authoritative and bypasses discovery.
    Auto-discovery runs the heuristic adapter first, then the LLM enhancer
    configured by ``config.llm``. When no ``enhancer`` is
    supplied, the default enhancer runs against a managed local vLLM server
    unless ``NSS_INFERENCE_ENDPOINT`` selects another service; that server
    stops as soon as enhancement ends. Dataframe-aware validation occurs once, after the final plan has
    been selected.
    """
    if (inline_plan := config.inline_plan) is not None:
        plan = inline_plan
    else:
        discovery_input = _prepare_discovery_input(df, data_config, time_series)
        baseline = (discoverer or HeuristicPlanDiscoverer()).discover(discovery_input)
        if enhancer is not None:
            plan = enhancer.enhance(discovery_input, baseline)
        else:
            plan = _enhance_with_default_llm(config.llm, discovery_input, baseline)
        plan = _without_row_counters(df, plan, discovery_input.group_column)

    validate_plan(
        df,
        plan,
        data_config=data_config,
        time_series=time_series,
        check_pattern_coverage=check_pattern_coverage,
    )
    return plan


def resolve_replacement_config(
    df: pd.DataFrame,
    config: ReplacePiiConfig,
    data_config: DataParameters,
    time_series: TimeSeriesParameters | None = None,
    *,
    discoverer: PlanDiscoverer | None = None,
    enhancer: PlanEnhancer | None = None,
    mapping_discoverer: DependencyMappingDiscoverer | None = None,
    sampler_value_catalog: Mapping[EntityType, tuple[str, ...]] | None = None,
    check_pattern_coverage: bool = True,
) -> ReplacePiiConfig:
    """Resolve the replacement plan and its data-to-sampler value mapping for ``df``.

    The plan comes from ``config`` or, for ``auto_discovery``, from the
    discoverer and the LLM enhancer. The value
    mapping then combines built-in entries for common gender spellings, the
    plan's own entries, and, for ``auto_discovery``, discovered entries for any
    remaining conditioner values without a sampler match.

    Args:
        df: Data the plan is validated against and whose conditioner values are mapped.
        config: PII replacement configuration to resolve.
        data_config: Data parameters, used for grouping and protected columns.
        time_series: Time-series parameters, used for protected columns.
        discoverer: Plan discoverer for ``auto_discovery``; defaults to the heuristic discoverer.
        enhancer: Plan enhancer; defaults to the LLM enhancer configured by ``config.llm``.
        mapping_discoverer: Value-mapping discoverer; defaults to the LLM enhancer configured by ``config.llm``.
        sampler_value_catalog: Sampler values per entity type; loaded from the configured sampler when omitted.
        check_pattern_coverage: Require each pattern to cover at least 99% of ``df``'s values; see
            ``validate_plan``.

    Returns:
        A copy of ``config`` whose ``replacement_plan`` is the resolved inline plan,
        including its complete ``data_to_sampler_value_mapping``.

    When both plan enhancement and mapping discovery use the default LLM, they share one inference
    session, so a managed local vLLM server starts once and stops before this returns. Plan
    validation runs inside that session, between the two LLM steps.

    Raises:
        ParameterError: If the plan or mapping is invalid, a conditioner value has
            no sampler match or mapping entry and cannot be discovered, or the
            selected Nemotron-Personas dataset is missing.

    Example:
        With a plan where ``first_name`` depends on ``sex``, values ``"F"`` and
        ``"Woman"`` resolve to ``{"sex": {"F": ["female"], "Woman": ["female"]}}``
        without an LLM, while a value such as ``"X"`` needs a mapping entry.
    """
    if config.is_auto_discovery and enhancer is None and mapping_discoverer is None:
        from ..local_inference import planning_inference_settings
        from .llm import LLMPlanEnhancer

        with planning_inference_settings(config.llm) as settings:
            llm = LLMPlanEnhancer(config.llm, settings=settings)
            return _resolve_replacement_config(
                df,
                config,
                data_config,
                time_series,
                discoverer=discoverer,
                enhancer=llm,
                mapping_discoverer=llm,
                sampler_value_catalog=sampler_value_catalog,
                check_pattern_coverage=check_pattern_coverage,
            )
    return _resolve_replacement_config(
        df,
        config,
        data_config,
        time_series,
        discoverer=discoverer,
        enhancer=enhancer,
        mapping_discoverer=mapping_discoverer,
        sampler_value_catalog=sampler_value_catalog,
        check_pattern_coverage=check_pattern_coverage,
    )


def _resolve_replacement_config(
    df: pd.DataFrame,
    config: ReplacePiiConfig,
    data_config: DataParameters,
    time_series: TimeSeriesParameters | None = None,
    *,
    discoverer: PlanDiscoverer | None = None,
    enhancer: PlanEnhancer | None = None,
    mapping_discoverer: DependencyMappingDiscoverer | None = None,
    sampler_value_catalog: Mapping[EntityType, tuple[str, ...]] | None = None,
    check_pattern_coverage: bool = True,
) -> ReplacePiiConfig:
    """Resolve the plan and value mapping with the given adapters; see ``resolve_replacement_config``."""
    discover_plan = config.is_auto_discovery
    plan = resolve_plan(
        df,
        config,
        data_config,
        time_series,
        discoverer=discoverer,
        enhancer=enhancer,
        check_pattern_coverage=check_pattern_coverage,
    )
    if discover_plan or "sampler" not in plan.model_fields_set:
        plan = _with_available_sampler(plan, config.locale)

    if sampler_value_catalog is None and conditioner_columns(plan):
        from ..sampler_catalog import load_sampler_value_catalog

        sampler_value_catalog = load_sampler_value_catalog(plan.sampler, config.locale)
    catalog = dict(sampler_value_catalog or {})

    # Built-in entries for common spellings come first so neither the LLM nor the user has to map them;
    # the plan's own entries are kept as written.
    plan = _with_value_mapping(
        plan, _merged_value_mappings(built_in_value_mappings(df, plan, catalog), plan.data_to_sampler_value_mapping)
    )
    mappings = plan.data_to_sampler_value_mapping
    if discover_plan and mapping_inputs(df, plan, catalog):
        if mapping_discoverer is not None:
            discovered = mapping_discoverer.discover_data_to_sampler_value_mapping(df, plan, catalog)
        else:
            discovered = _discover_mappings_with_default_llm(config.llm, df, plan, catalog)
        mappings = _merged_value_mappings(mappings, discovered)

    resolved_plan = _with_value_mapping(plan, mappings)
    validate_data_to_sampler_value_mapping(resolved_plan, catalog)
    if unmatched := unmatched_dependency_values(df, resolved_plan, catalog):
        columns = ", ".join(f"{column!r} ({count})" for column, count in sorted(unmatched.items()))
        raise ParameterError(
            "data_to_sampler_value_mapping has no entry for dataset values that match no sampler value, "
            f"counted by depends_on column: {columns}. Add an entry listing the sampler values each one may "
            "match, or list every sampler value to allow any match"
        )
    return config.with_plan(resolved_plan)


def _with_available_sampler(plan: PiiReplacementPlan, locale: str) -> PiiReplacementPlan:
    """Return ``plan`` using Faker when its default Nemotron-Personas sampler has no data for ``locale``.

    Called for a discovered plan and for a written plan without a ``sampler``: the user has not chosen
    the sampler, so a missing dataset switches it to Faker with a warning instead of failing. A plan that
    names a sampler keeps it and fails later if its data is missing. The resolved plan records the switch
    for review.
    """
    from ..sampler_catalog import nemotron_personas_dataset_path

    sampler = plan.sampler
    if sampler.backend is not PiiSamplerBackend.NEMOTRON_PERSONAS:
        return plan
    path = nemotron_personas_dataset_path(sampler, locale)
    if path.is_file():
        return plan
    logger.user.warning(
        f"Using the Faker sampler because no Nemotron-Personas data for locale {locale!r} was found at {path}. "
        "To use Nemotron-Personas, download its dataset and point the plan's sampler.nemotron_personas_path or "
        "NSS_NEMOTRON_PERSONAS_PATH at it."
    )
    return plan.model_copy(update={"sampler": sampler.model_copy(update={"backend": PiiSamplerBackend.FAKER})})


def _with_value_mapping(plan: PiiReplacementPlan, mapping: DataToSamplerValueMapping) -> PiiReplacementPlan:
    """Return ``plan`` with ``mapping`` as its value mapping, validated, keeping its other fields."""
    fields = {name: getattr(plan, name) for name in PiiReplacementPlan.model_fields}
    return PiiReplacementPlan.model_validate({**fields, "data_to_sampler_value_mapping": mapping})


def _merged_value_mappings(*mappings: DataToSamplerValueMapping) -> DataToSamplerValueMapping:
    """Combine value mappings column by column; a later mapping's entry replaces an earlier one for the same key."""
    merged: DataToSamplerValueMapping = {}
    for mapping in mappings:
        for column_name, value_mappings in mapping.items():
            merged.setdefault(column_name, {}).update(value_mappings)
    return merged


def _discover_mappings_with_default_llm(
    llm_config: LLMConfig,
    df: pd.DataFrame,
    plan: PiiReplacementPlan,
    catalog: Mapping[EntityType, tuple[str, ...]],
) -> DataToSamplerValueMapping:
    """Run value-mapping discovery with the default LLM inside its own inference session."""
    from ..local_inference import planning_inference_settings
    from .llm import LLMPlanEnhancer

    with planning_inference_settings(llm_config) as settings:
        return LLMPlanEnhancer(llm_config, settings=settings).discover_data_to_sampler_value_mapping(df, plan, catalog)


def row_counter_columns(df: pd.DataFrame, *, group_column: str | None = None) -> frozenset[str]:
    """Return columns whose values number the rows, so automatic discovery does not treat them as PII.

    A row counter holds unique, non-missing integers that are exactly ``0`` to
    ``n - 1`` or ``1`` to ``n`` for ``n`` rows, in any order. Examples include
    the ``Unnamed: 0`` column pandas writes for a saved index, and plain
    ``row_id`` or ``event_id`` sequences. Detection uses the values only, not
    the column name. The grouping column is never a row counter: it identifies
    the entity each row belongs to.

    Example:
        A column holding ``3, 0, 2, 1`` in a four-row dataframe is a row
        counter; ``0, 1, 3, 4`` and ``17, 18, 19, 20`` are not.
    """
    return frozenset(str(column) for column in df.columns if column != group_column and _is_row_counter(df[column]))


def _is_row_counter(values: pd.Series) -> bool:
    if len(values) < 2 or values.isna().any():
        return False
    if pd.api.types.is_bool_dtype(values) or not pd.api.types.is_numeric_dtype(values):
        return False
    numbers = values.to_numpy(dtype="float64")
    if not np.array_equal(numbers, np.floor(numbers)):
        return False
    start = numbers.min()
    return start in (0.0, 1.0) and numbers.max() - start + 1 == len(numbers) and len(np.unique(numbers)) == len(numbers)


def _without_row_counters(df: pd.DataFrame, plan: PiiReplacementPlan, group_column: str | None) -> PiiReplacementPlan:
    """Drop automatically discovered replacements for row-counter columns; explicit plans are never changed."""
    counters = row_counter_columns(df, group_column=group_column)
    kept = [spec for spec in plan.columns_to_replace if spec.column_name not in counters]
    if len(kept) == len(plan.columns_to_replace):
        return plan
    skipped = sorted(spec.column_name for spec in plan.columns_to_replace if spec.column_name in counters)
    logger.user.info(
        "Skipping row-counter columns in automatic PII discovery",
        extra={"columns": skipped},
    )
    return plan.model_copy(update={"columns_to_replace": kept})
