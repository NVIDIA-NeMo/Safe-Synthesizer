# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve one validated PII replacement plan from configuration and data."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from ...config.data import DataParameters
from ...config.replace_pii import DependencyValueMappings, EntityType, PiiReplacementPlan, ReplacePiiConfig
from ...config.time_series import TimeSeriesParameters
from ...errors import ParameterError
from .dependency_mappings import dependency_columns, mapping_inputs, validate_dependency_value_mappings
from .io import load_plan, save_plan
from .validation import get_protected_columns, validate_plan

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

    def discover_dependency_value_mappings(
        self,
        dataframe: object,
        plan: PiiReplacementPlan,
        catalog: Mapping[EntityType, tuple[str, ...]],
    ) -> DependencyValueMappings:
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
    enhancer only when ``config.llm`` is configured. Dataframe-aware validation
    occurs once, after the final plan has been selected.
    """
    if not config.is_auto_discovery:
        plan = _configured_plan(config)
    else:
        discovery_input = _prepare_discovery_input(df, data_config, time_series)
        baseline = (discoverer or HeuristicPlanDiscoverer()).discover(discovery_input)
        if config.llm is None:
            plan = baseline
        else:
            if enhancer is None:
                from .llm import LLMPlanEnhancer

                enhancer = LLMPlanEnhancer(config.llm)
            plan = enhancer.enhance(discovery_input, baseline)

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
    dependency_labels: Mapping[EntityType, tuple[str, ...]] | None = None,
) -> ReplacePiiConfig:
    """Resolve the semantic plan and sampler-specific dependency mappings."""
    default_enhancer: PlanEnhancer | None = None
    default_mapping_discoverer: DependencyMappingDiscoverer | None = None
    if config.llm is not None and config.is_auto_discovery and enhancer is None:
        from .llm import LLMPlanEnhancer

        llm_adapter = LLMPlanEnhancer(config.llm)
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

    if dependency_labels is None and dependency_columns(plan):
        from ..dependency_labels import dependency_label_catalog

        dependency_labels = dependency_label_catalog(config.replacement, config.sampler)
    catalog = dict(dependency_labels or {})

    mappings = config.sampler.inline_dependency_value_mappings
    if mappings is None:
        unresolved = mapping_inputs(df, plan, catalog)
        if not unresolved:
            mappings = {}
        else:
            active_discoverer = mapping_discoverer or default_mapping_discoverer
            if active_discoverer is None and config.llm is not None:
                from .llm import LLMPlanEnhancer

                active_discoverer = LLMPlanEnhancer(config.llm)
            if active_discoverer is None:
                raise ParameterError(
                    "automatic dependency value mapping found dataset labels without case-insensitive sampler "
                    "matches; configure replace_pii.llm or provide manual dependency_value_mappings"
                )
            mappings = active_discoverer.discover_dependency_value_mappings(df, plan, catalog)

    validate_dependency_value_mappings(plan, mappings, catalog)
    sampler = config.sampler.model_copy(update={"dependency_value_mappings": mappings})
    return config.model_copy(update={"replacement_plan": plan, "sampler": sampler})
