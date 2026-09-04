# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve one validated PII replacement plan from configuration and data."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...config.data import DataParameters
from ...config.replace_pii import PiiReplacementPlan, PiiReplacementScope, ReplacePiiConfig
from ...config.time_series import TimeSeriesParameters
from ...errors import ParameterError
from .io import load_plan, save_plan
from .validation import protected_columns, validate_plan

__all__ = [
    "ColumnProfile",
    "HeuristicPlanDiscoverer",
    "PlanDiscoverer",
    "PlanDiscoveryInput",
    "PlanEnhancer",
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
    """Data and deterministic preparation supplied to discovery adapters."""

    dataframe: pd.DataFrame
    scope: PiiReplacementScope
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


class HeuristicPlanDiscoverer(PlanDiscoverer):
    """Initial no-op heuristic adapter, ready for later rule discovery."""

    def discover(self, discovery_input: PlanDiscoveryInput) -> PiiReplacementPlan:
        """Return an empty plan while preserving the deterministically resolved scope."""
        return PiiReplacementPlan(scope=discovery_input.scope)


def _stable_samples(series: pd.Series) -> tuple[str, ...]:
    """Pick a bounded set of distinct raw column values for discovery.

    Nulls are removed, values are converted to truncated strings, and duplicate
    strings are collapsed. Hash ordering assigns each distinct value a stable,
    content-based priority, so reordering dataframe rows does not change which
    values are selected.
    """
    values = {str(value)[:MAX_PROFILE_SAMPLE_LENGTH] for value in series.dropna().tolist()}
    # Rank distinct values by their content rather than sampling row positions.
    ordered = sorted(values, key=lambda value: hashlib.sha256(value.encode()).digest())
    return tuple(ordered[:MAX_PROFILE_SAMPLES])


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
    scope = PiiReplacementScope.GROUP if group_column is not None else PiiReplacementScope.DATAFRAME
    structural_columns = protected_columns(data_config, time_series)
    return PlanDiscoveryInput(
        dataframe=df,
        scope=scope,
        group_column=group_column,
        protected_columns=structural_columns,
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
        elif enhancer is None:
            raise ParameterError("replace_pii.llm is configured, but no LLM plan enhancer is available in this build")
        else:
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
