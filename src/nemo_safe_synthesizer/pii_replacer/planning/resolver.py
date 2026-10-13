# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve one validated PII replacement plan from configuration and data."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pandas as pd

from ...config.data import DataParameters
from ...config.replace_pii import DataToSamplerValueMapping, EntityType, LLMConfig, PiiReplacementPlan, ReplacePiiConfig
from ...config.time_series import TimeSeriesParameters
from ...errors import InternalError, ParameterError
from .dependency_mappings import (
    built_in_value_mappings,
    conditioner_columns,
    mapping_inputs,
    unmatched_dependency_values,
    validate_data_to_sampler_value_mapping,
)
from .io import load_plan, save_plan
from .validation import get_protected_columns, validate_plan

if TYPE_CHECKING:
    from .llm import LLMPlanEnhancer

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


def _configured_plan(config: ReplacePiiConfig) -> PiiReplacementPlan:
    if config.plan_path is not None:
        return load_plan(config.plan_path)
    if config.inline_plan is not None:
        return config.inline_plan
    raise ParameterError("replacement_plan must be auto_discovery, an inline plan, or a path to a plan file")


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
    output_path: str | Path | None = None,
) -> PiiReplacementPlan:
    """Resolve, validate, and optionally persist one replacement plan.

    Inline plans and plan files are authoritative and bypass discovery.
    Auto-discovery always runs the heuristic adapter first, then runs an LLM
    enhancer only when ``config.llm`` is configured. When no ``enhancer`` is
    supplied, the default enhancer runs against a managed local vLLM server
    unless ``NSS_INFERENCE_ENDPOINT`` selects another service; that server
    stops as soon as enhancement ends. Dataframe-aware validation occurs once, after the final plan has
    been selected.
    """
    if not config.is_auto_discovery:
        plan = _configured_plan(config)
    else:
        discovery_input = _prepare_discovery_input(df, data_config, time_series)
        baseline = (discoverer or HeuristicPlanDiscoverer()).discover(discovery_input)
        if config.llm is None:
            plan = baseline
        elif enhancer is not None:
            plan = enhancer.enhance(discovery_input, baseline)
        else:
            plan = _enhance_with_default_llm(config.llm, discovery_input, baseline)

    validate_plan(
        df,
        plan,
        data_config=data_config,
        time_series=time_series,
    )
    if output_path is not None:
        save_plan(plan, output_path)
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
) -> ReplacePiiConfig:
    """Resolve the replacement plan and its data-to-sampler value mapping for ``df``.

    The plan comes from ``config`` or, for ``auto_discovery``, from the
    discoverer and, when ``config.llm`` is set, the LLM enhancer. The value
    mapping then combines built-in entries for common gender spellings, the
    plan's own entries, and, for ``auto_discovery``, discovered entries for any
    remaining conditioner values without a sampler match.

    Args:
        df: Data the plan is validated against and whose conditioner values are mapped.
        config: PII replacement configuration to resolve.
        data_config: Data parameters, used for grouping and protected columns.
        time_series: Time-series parameters, used for protected columns.
        discoverer: Plan discoverer for ``auto_discovery``; defaults to the heuristic discoverer.
        enhancer: Plan enhancer; defaults to the LLM enhancer when ``config.llm`` is set.
        mapping_discoverer: Value-mapping discoverer; defaults to the LLM enhancer when ``config.llm`` is set.
        sampler_value_catalog: Sampler values per entity type; loaded from the configured sampler when omitted.

    Returns:
        A copy of ``config`` whose ``replacement_plan`` is the resolved inline plan,
        including its complete ``data_to_sampler_value_mapping``.

    Raises:
        ParameterError: If the plan or mapping is invalid, a conditioner value has
            no sampler match or mapping entry and cannot be discovered, or the
            selected Nemotron-Personas dataset is missing.

    Example:
        With a plan where ``first_name`` depends on ``sex``, values ``"F"`` and
        ``"Woman"`` resolve to ``{"sex": {"F": ["female"], "Woman": ["female"]}}``
        without an LLM, while a value such as ``"X"`` needs a mapping entry.
    """
    discover_plan = config.is_auto_discovery
    default_enhancer: PlanEnhancer | None = None
    default_mapping_discoverer: DependencyMappingDiscoverer | None = None
    if config.llm is not None and discover_plan and enhancer is None:
        llm_adapter = _llm_plan_enhancer(config)
        default_enhancer = llm_adapter
        default_mapping_discoverer = llm_adapter

    plan = resolve_plan(
        df,
        config,
        data_config,
        time_series,
        discoverer=discoverer,
        enhancer=enhancer or default_enhancer,
    )

    if sampler_value_catalog is None and conditioner_columns(plan):
        from ..sampler_catalog import load_sampler_value_catalog

        sampler_value_catalog = load_sampler_value_catalog(config.replacement, config.sampler)
    catalog = dict(sampler_value_catalog or {})

    # Built-in entries for common spellings come first so neither the LLM nor the user has to map them;
    # the plan's own entries are kept as written.
    plan = PiiReplacementPlan(
        columns_to_replace=plan.columns_to_replace,
        data_to_sampler_value_mapping=_merged_value_mappings(
            built_in_value_mappings(df, plan, catalog), plan.data_to_sampler_value_mapping
        ),
    )
    mappings = plan.data_to_sampler_value_mapping
    if discover_plan and mapping_inputs(df, plan, catalog):
        active_discoverer = mapping_discoverer or default_mapping_discoverer
        if active_discoverer is None and config.llm is not None:
            active_discoverer = _llm_plan_enhancer(config)
        if active_discoverer is None:
            raise ParameterError(
                "automatic data-to-sampler value mapping found dataset values without case-insensitive sampler "
                "matches; configure replace_pii.llm, or generate and edit an explicit replacement plan"
            )
        discovered = active_discoverer.discover_data_to_sampler_value_mapping(df, plan, catalog)
        mappings = _merged_value_mappings(mappings, discovered)

    resolved_plan = PiiReplacementPlan(
        columns_to_replace=plan.columns_to_replace,
        data_to_sampler_value_mapping=mappings,
    )
    validate_data_to_sampler_value_mapping(resolved_plan, catalog)
    if unmatched := unmatched_dependency_values(df, resolved_plan, catalog):
        columns = ", ".join(f"{column!r} ({count})" for column, count in sorted(unmatched.items()))
        raise ParameterError(
            "data_to_sampler_value_mapping has no entry for dataset values that match no sampler value, "
            f"counted by depends_on column: {columns}. Add an entry listing the sampler values each one may "
            "match, or list every sampler value to allow any match"
        )
    return config.model_copy(update={"replacement_plan": resolved_plan})


def _merged_value_mappings(*mappings: DataToSamplerValueMapping) -> DataToSamplerValueMapping:
    """Combine value mappings column by column; a later mapping's entry replaces an earlier one for the same key."""
    merged: DataToSamplerValueMapping = {}
    for mapping in mappings:
        for column_name, value_mappings in mapping.items():
            merged.setdefault(column_name, {}).update(value_mappings)
    return merged


def _llm_plan_enhancer(config: ReplacePiiConfig) -> LLMPlanEnhancer:
    """Build the LLM plan enhancer for ``config.llm``, importing the LLM client only when needed."""
    if config.llm is None:
        raise InternalError("the LLM plan enhancer requires replace_pii.llm")
    from .llm import LLMPlanEnhancer

    return LLMPlanEnhancer(config.llm)
