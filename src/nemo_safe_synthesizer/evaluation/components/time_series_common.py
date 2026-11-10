# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared component base and sequence handling for time-series evaluation metrics.

Every time-series metric resolves the same grouping column, selects the same
numeric value columns, walks the same shared groups in the same order, and
renders as one card in the report's Time-Series Metrics panel. Metrics that
score each group and column pair separately also share ``evaluate_pairs``.
Keeping that logic here means a fix applies to every metric at once.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from pydantic import Field

from ...artifacts.analyzers.field_features import FieldType
from ...config.parameters import SafeSynthesizerParameters
from ...defaults import PSEUDO_GROUP_COLUMN
from ...evaluation.assets.text.multi_modal_tooltips import tooltips
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from .component import Component

logger = get_logger(__name__)

_GROUP_SELECTION_SEED = 2112
_CONSTANT_TOLERANCE_FACTOR = 32.0

FigureSpec = tuple[str, go.Figure, bool, str]
"""Chart title, figure, whether it gets the column select, and its tooltip key."""

PairResult = tuple[dict[str, Any] | None, str | None]
"""Profile details and ``None`` for a scored pair, or ``None`` and the reason it was skipped."""

PairScorer = Callable[[pd.Series, pd.Series], PairResult]
"""Compare one ordered training and synthetic sequence for the same group and column."""


class PairMetricParameters(Protocol):
    """Metric parameters ``evaluate_pairs`` reads."""

    metric_label: ClassVar[str]
    value_columns: list[str] | None
    max_groups: int


class TimeSeriesComponent(Component):
    """Base for components shown in the report's Time-Series Metrics panel.

    The report runs every component in ``TIME_SERIES_COMPONENTS`` and renders
    each as a metric card from the class attributes below. Subclasses extend
    ``jinja_context`` with their figures, built with ``figure_rows``.

    Attributes:
        card_id: HTML id of the metric card.
        description: One-line description shown on the card.
        tooltip_key: Key of the card's info tooltip in ``tooltips``.
        template: Jinja template rendered inside the card.
        score_colored_traces: Whether chart traces are colored by score, so
            the report keeps their colors instead of applying dataset colors.
    """

    card_id: ClassVar[str]
    description: ClassVar[str]
    tooltip_key: ClassVar[str]
    template: ClassVar[str]
    score_colored_traces: ClassVar[bool] = False

    details: dict[str, Any] = Field(default_factory=dict, description="Metric-specific diagnostic details.")

    @cached_property
    def jinja_context(self) -> dict[str, Any]:
        """Return score, details, and card attributes for report rendering."""
        context = super().jinja_context
        context["details"] = self.details
        context["card_id"] = self.card_id
        context["description"] = self.description
        context["tooltip_key"] = self.tooltip_key
        context["template"] = self.template
        context["score_colored_traces"] = self.score_colored_traces
        return context

    @staticmethod
    def figure_rows(figures: Sequence[FigureSpec]) -> list[dict[str, Any]]:
        """Convert figure specs into the rows the chart grid template renders."""
        return [
            {
                "title": title,
                "html": figure.to_html(full_html=False, include_plotlyjs=False),
                "column_select": column_select,
                "tooltip_id": tooltip_key.replace("_", "-"),
                "tooltip": tooltips[tooltip_key],
            }
            for title, figure, column_select, tooltip_key in figures
        ]


def evaluate_pairs(
    datasets: EvaluationDatasets,
    config: SafeSynthesizerParameters | None,
    cfg: PairMetricParameters,
    score_pair: PairScorer,
) -> tuple[EvaluationScore, dict[str, Any]]:
    """Score every shared group and value column pair and average the similarities.

    Groups above ``cfg.max_groups`` are sampled reproducibly. Pairs that
    ``score_pair`` skips are reported in ``details`` and do not count toward
    the score. Evaluation failures are returned in the score notes instead of
    aborting the full evaluation pipeline.

    Args:
        datasets: Training and synthetic datasets to compare.
        config: Optional top-level configuration used for inherited columns.
        cfg: Resolved metric parameters.
        score_pair: Metric-specific comparison of one pair.

    Returns:
        The score on a 0-10 scale, or an unscored result with explanatory
        notes, and the diagnostic details the report renders.
    """
    # Optional metrics must fail independently so one diagnostic cannot
    # prevent the rest of the evaluation report from being produced.
    try:
        return _evaluate_pairs(datasets, config, cfg, score_pair)
    except Exception as exc:
        logger.exception(f"Failed to compute {cfg.metric_label} Similarity.")
        return EvaluationScore(notes=str(exc)), {}


def _evaluate_pairs(
    datasets: EvaluationDatasets,
    config: SafeSynthesizerParameters | None,
    cfg: PairMetricParameters,
    score_pair: PairScorer,
) -> tuple[EvaluationScore, dict[str, Any]]:
    """Run ``evaluate_pairs`` without failure isolation."""
    label = cfg.metric_label.lower()
    timestamp_column = config.time_series.timestamp_column if config is not None else None
    group_column, group_error = resolve_group_column(datasets, config)
    if group_error is not None:
        return EvaluationScore(notes=group_error), {}

    columns, column_error, excluded_columns = select_value_columns(
        datasets, cfg.value_columns, timestamp_column, group_column, label
    )
    if column_error is not None:
        return EvaluationScore(notes=column_error), {}
    if not columns:
        return _unavailable("No shared numeric value columns.", excluded_columns)

    groups, missing_training, missing_synthetic, shared_group_count = shared_groups(
        datasets.training, datasets.synthetic, group_column, cfg.max_groups, _GROUP_SELECTION_SEED
    )
    if not groups:
        return _unavailable("No shared groups to evaluate.", excluded_columns)
    omitted_groups = shared_group_count - len(groups)
    sample_note = None
    if omitted_groups:
        sample_note = f"Evaluated {len(groups)} of {shared_group_count} shared groups using a reproducible sample."

    profiles: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for group_value in groups:
        training_group = group_frame(datasets.training, group_column, group_value, timestamp_column)
        synthetic_group = group_frame(datasets.synthetic, group_column, group_value, timestamp_column)
        group_label = None if group_column is None else str(group_value)
        for column in columns:
            result, reason = score_pair(training_group[column], synthetic_group[column])
            if result is None:
                skipped.append({"group": group_label, "column": column, "reason": reason})
            else:
                profiles.append({"group": group_label, "column": column, **result})

    diagnostics = {
        "counts": {
            "groups": len(groups),
            "shared_groups": shared_group_count,
            "omitted_shared_groups": omitted_groups,
            "columns": len(columns),
            "evaluated_profiles": len(profiles),
            "skipped": len(skipped),
        },
        "group_selection": {
            "shared_groups": shared_group_count,
            "evaluated_groups": len(groups),
            "omitted_shared_groups": omitted_groups,
            "policy": "seeded_random_sample" if omitted_groups else "all_shared_groups",
        },
        "skipped": skipped,
        "groups_only_in_training": missing_training,
        "groups_only_in_synthetic": missing_synthetic,
    }
    if not profiles:
        note = f"No usable group/column {label} profiles."
        return _unavailable(note if sample_note is None else f"{note} {sample_note}", excluded_columns, diagnostics)

    similarity = float(np.mean([item["similarity"] for item in profiles]))
    score = EvaluationScore.finalize_grade(raw_score=similarity, score=10.0 * similarity)
    skipped_note = f"Skipped {len(skipped)} unusable group/column comparisons." if skipped else None
    notes = [note for note in (skipped_note, sample_note, excluded_columns_note(excluded_columns)) if note]
    if notes:
        score.notes = " ".join(notes)

    details = {
        "evaluation_mode": "per_group" if group_column else "global",
        "timestamp_column": timestamp_column,
        "group_column": group_column,
        **diagnostics,
        "per_group": summaries(profiles, "group"),
        "per_column": summaries(profiles, "column"),
        "profiles": profiles,
        "excluded_columns": excluded_columns,
    }
    return score, details


def _unavailable(
    note: str,
    excluded_columns: list[str],
    details: dict[str, Any] | None = None,
) -> tuple[EvaluationScore, dict[str, Any]]:
    """Return an unscored result whose notes and details still name left-out columns."""
    excluded_note = excluded_columns_note(excluded_columns)
    return (
        EvaluationScore(notes=note if excluded_note is None else f"{note} {excluded_note}"),
        {**(details or {}), "excluded_columns": excluded_columns},
    )


def prepare_values(values: pd.Series) -> NDArray[np.float64]:
    """Convert a series to floats while preserving missing and non-finite positions as ``NaN`` gaps."""
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return np.where(np.isfinite(numeric_values), numeric_values, np.nan)


def profile_details(values: NDArray[np.float64]) -> list[float | None]:
    """Convert a profile vector into JSON-safe full-precision details, with ``None`` for non-finite entries."""
    return [float(value) if np.isfinite(value) else None for value in values]


def is_effectively_constant(values: NDArray[np.float64]) -> bool:
    """Return whether finite variation is negligible relative to value scale."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return True
    scale = float(np.max(np.abs(finite_values)))
    if scale == 0.0:
        return True
    normalized_std = float(np.std(finite_values / scale))
    tolerance = _CONSTANT_TOLERANCE_FACTOR * np.finfo(float).eps
    return bool(normalized_std <= tolerance)


def excluded_columns_note(excluded: list[str]) -> str | None:
    """Return a report note naming numeric training columns left out of evaluation."""
    if not excluded:
        return None
    return (
        f"Left out {len(excluded)} numeric training columns that synthetic data lacks "
        f"or stores as non-numeric values: {', '.join(excluded)}."
    )


def resolve_group_column(
    datasets: EvaluationDatasets,
    config: SafeSynthesizerParameters | None,
) -> tuple[str | None, str | None]:
    """Return the grouping column evaluation should use.

    Args:
        datasets: Training and synthetic datasets to compare.
        config: Optional top-level configuration naming the group column.

    Returns:
        The group column, or ``None`` for one global sequence, and an error
        note when a configured group column is missing from a dataset.
    """
    group_column = config.data.group_training_examples_by if config is not None else None
    if (
        group_column == PSEUDO_GROUP_COLUMN
        and group_column not in datasets.training
        and group_column not in datasets.synthetic
    ):
        # Training injects this reserved column to reuse grouped sequence
        # infrastructure, but evaluation receives frames with it removed.
        return None, None
    if group_column is not None and (group_column not in datasets.training or group_column not in datasets.synthetic):
        return None, f"Configured group column {group_column!r} is missing from a dataset."
    return group_column, None


def select_value_columns(
    datasets: EvaluationDatasets,
    value_columns: list[str] | None,
    timestamp_column: str | None,
    group_column: str | None,
    label: str,
) -> tuple[list[str], str | None, list[str]]:
    """Select numeric value columns eligible for evaluation.

    Explicit ``value_columns`` retain their configured order. Automatic
    selection is sorted for deterministic output and follows the training
    field types, so a synthetic column that collapsed to a constant or a few
    integers is still evaluated (and penalized) instead of silently dropping
    out. Timestamp and grouping columns are excluded even when their storage
    dtype is numeric.

    Args:
        datasets: Training and synthetic datasets with inferred field types.
        value_columns: Explicitly configured columns, or ``None`` for automatic
            selection.
        timestamp_column: Column used only to order observations.
        group_column: Column used only to separate sequences.
        label: Metric name used in the validation error, such as
            ``"autocorrelation"``.

    Returns:
        Selected column names, an optional validation error, and the numeric
        training columns left out because synthetic data lacks them or stores
        them as non-numeric values.
    """
    if value_columns is not None:
        invalid = invalid_value_columns(datasets, value_columns, timestamp_column, group_column)
        if invalid:
            return [], f"Invalid {label} value columns: " + "; ".join(invalid), []
        return list(dict.fromkeys(value_columns)), None, []
    reserved = {timestamp_column, group_column, PSEUDO_GROUP_COLUMN}
    candidates = sorted(
        column
        for column in datasets.get_columns_of_type({FieldType.NUMERIC}, based_on="training")
        if column not in reserved
    )
    columns: list[str] = []
    excluded: list[str] = []
    for column in candidates:
        if column in datasets.synthetic and pd.api.types.is_numeric_dtype(datasets.synthetic[column]):
            columns.append(column)
        else:
            excluded.append(column)
    return columns, None, excluded


def invalid_value_columns(
    datasets: EvaluationDatasets,
    columns: list[str],
    timestamp_column: str | None,
    group_column: str | None,
) -> list[str]:
    """Return actionable validation failures for explicit value columns."""
    invalid: list[str] = []
    reserved = {timestamp_column, group_column, PSEUDO_GROUP_COLUMN}
    for column in columns:
        if column in reserved:
            invalid.append(f"{column!r} is used for timestamp or sequence grouping")
            continue
        missing_from = [
            name
            for name, frame in (("training", datasets.training), ("synthetic", datasets.synthetic))
            if column not in frame
        ]
        if missing_from:
            invalid.append(f"{column!r} is missing from {' and '.join(missing_from)} data")
            continue
        if not pd.api.types.is_numeric_dtype(datasets.training[column]):
            invalid.append(f"{column!r} is not numeric in training data")
            continue
        if not pd.api.types.is_numeric_dtype(datasets.synthetic[column]):
            invalid.append(f"{column!r} is not numeric in synthetic data")
    return invalid


def shared_groups(
    training: pd.DataFrame,
    synthetic: pd.DataFrame,
    group_column: str | None,
    max_groups: int,
    seed: int,
) -> tuple[list[Any], list[str], list[str], int]:
    """Find a deterministic, bounded set of groups shared by both datasets.

    A ``None`` sentinel represents one global sequence when grouping is not
    configured. Group labels present in only one dataset are returned for
    diagnostics but do not contribute to the score.

    Args:
        training: Training records containing candidate group labels.
        synthetic: Synthetic records containing candidate group labels.
        group_column: Optional column that identifies independent sequences.
        max_groups: Maximum number of shared groups to evaluate.
        seed: Seed for the reproducible sample taken above ``max_groups``.

    Returns:
        Shared group keys, training-only labels, synthetic-only labels,
        and the total number of shared groups before limiting.
    """
    if group_column is None:
        return [None], [], [], 1
    if group_column not in training or group_column not in synthetic:
        return [], [], [], 0
    training_groups = set(training[group_column].dropna().unique())
    synthetic_groups = set(synthetic[group_column].dropna().unique())
    ordered_shared = sorted(training_groups & synthetic_groups, key=group_sort_key)
    shared = ordered_shared
    if len(ordered_shared) > max_groups:
        shared = random.Random(seed).sample(ordered_shared, k=max_groups)
        shared.sort(key=group_sort_key)
    only_training = [str(value) for value in sorted(training_groups - synthetic_groups, key=str)]
    only_synthetic = [str(value) for value in sorted(synthetic_groups - training_groups, key=str)]
    return shared, only_training, only_synthetic, len(ordered_shared)


def group_sort_key(value: Any) -> tuple[str, str]:
    """Return a stable ordering key for heterogeneous group values."""
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    return type_name, str(value)


def group_frame(
    df: pd.DataFrame,
    group_column: str | None,
    group_value: Any,
    timestamp_column: str | None,
) -> pd.DataFrame:
    """Return one sequence in deterministic timestamp order.

    ``mergesort`` preserves input order for equal timestamps, which makes
    repeated runs stable without inventing a secondary ordering key.

    Args:
        df: Dataset containing one or more sequences.
        group_column: Optional sequence identifier column.
        group_value: Group to select, or ``None`` for the full dataset.
        timestamp_column: Optional column used to order the selected rows.

    Returns:
        A filtered and time-ordered DataFrame view or copy.
    """
    frame = df if group_column is None else df[df[group_column] == group_value]
    if timestamp_column and timestamp_column in frame:
        frame = frame.sort_values(timestamp_column, kind="mergesort")
    return frame


def summaries(profiles: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Average profile similarities by a diagnostic key.

    Args:
        profiles: Successful group-and-column comparison details.
        key: Detail key to group by, such as ``group`` or ``column``.

    Returns:
        Deterministically ordered summaries with similarity and count.
    """
    scores: defaultdict[Any, list[float]] = defaultdict(list)
    for item in profiles:
        scores[item[key]].append(item["similarity"])
    return [
        {key: value, "similarity": float(np.mean(values)), "count": len(values)}
        for value, values in sorted(scores.items(), key=lambda item: str(item[0]))
    ]
