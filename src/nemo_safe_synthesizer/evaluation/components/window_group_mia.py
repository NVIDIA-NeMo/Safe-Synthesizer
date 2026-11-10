# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Window membership inference protection for grouped numeric time-series data.

The metric asks whether synthetic data sits closer to training sequences than
to unseen holdout sequences from the same process, the trace a generator
leaves when it memorizes training data. Short windows of consecutive values,
across all evaluated numeric columns at once, are compared with their nearest
synthetic window. The score measures how much more often training windows
are flagged as close than holdout windows are, after removing the difference
expected from chance, on a 0--10 scale where 10 means no measurable
memorization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from ...config.evaluate import (
    WINDOW_GROUP_MIA_MIN_HOLDOUT_GROUPS,
    WINDOW_GROUP_MIA_MIN_WINDOWS,
    WindowGroupMIAParameters,
)
from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from .time_series_common import (
    FigureSpec,
    TimeSeriesComponent,
    excluded_columns_note,
    group_frame,
    group_sort_key,
    resolve_group_column,
    select_value_columns,
)
from .window_group_mia_figures import (
    generate_window_group_mia_closest_match_figure,
    generate_window_group_mia_distance_figure,
    generate_window_group_mia_group_figure,
    window_group_mia_closest_traces_per_column,
)

logger = get_logger(__name__)

WINDOW_GROUP_MIA_FLAG_RATE = 0.05
WINDOW_GROUP_MIA_MAX_SYNTHETIC_WINDOWS = 50_000
_OFFSET_SPLITS = 6
_SAMPLING_SEED = 2112
_DISTANCE_CHUNK = 256
_MIN_SCALE = 1e-12


@dataclass(frozen=True)
class _Sequence:
    """One group's numeric values in time order, with gaps filled by interpolation."""

    group: str | None
    values: NDArray[np.float64]


@dataclass(frozen=True)
class _Windows:
    """Flattened windows with the group and starting position each came from."""

    values: NDArray[np.float64]
    groups: list[str | None]
    starts: list[int]
    total: int

    def __len__(self) -> int:
        return len(self.groups)


@dataclass(frozen=True)
class _FlagCounts:
    """Training and holdout windows at or below the holdout's flag-rate distance quantile."""

    flagged_training: int
    training: int
    flagged_holdout: int
    holdout: int
    threshold: float

    @property
    def privacy_loss(self) -> float:
        """``log(TPR / FPR)``, with half a window added to each flagged count so it stays finite."""
        training_rate = (self.flagged_training + 0.5) / (self.training + 1)
        holdout_rate = (self.flagged_holdout + 0.5) / (self.holdout + 1)
        return float(np.log(training_rate) - np.log(holdout_rate))

    def binomial_noise(self, overlap: float) -> float:
        """Delta-method variance of ``privacy_loss`` for two binomial rates, times ``overlap``.

        Windows that start every half window share points, so their flags are
        correlated; ``overlap`` is the number of windows that share each point.
        """
        terms = (
            1 / (self.flagged_training + 0.5)
            - 1 / (self.training + 1)
            + 1 / (self.flagged_holdout + 0.5)
            - 1 / (self.holdout + 1)
        )
        return overlap * terms

    def exact_copy_loss(self, excess_exact_rate: float) -> float:
        """Privacy loss implied by exact copies alone, which chance can't explain.

        If ``excess_exact_rate`` of the training windows are certainly flagged
        and the rest are flagged at the holdout rate, the training flag rate is
        ``x + (1 - x) * alpha`` against ``alpha``. Exact copies of continuous
        values almost never arise by chance, so this loss is not reduced by
        the noise terms.
        """
        holdout_rate = (self.flagged_holdout + 0.5) / (self.holdout + 1)
        return float(np.log1p(excess_exact_rate * (1 - holdout_rate) / holdout_rate))


class WindowGroupMIA(TimeSeriesComponent):
    """Measure how well synthetic data protects training sequences from window membership inference.

    Training and holdout windows are compared with their nearest synthetic
    window. A window is flagged when it is closer than all but the closest 5%
    of holdout windows. The privacy loss is the log ratio of the training and
    holdout flag rates; chance variation from the binomial counts and from the
    holdout groups' own offset is removed before scoring
    ``10 * exp(-privacy_loss)``. Exact copies beyond the holdout's own
    exact-match rate set a floor on the privacy loss that chance can't remove,
    so a small holdout can't hide them. Without a group column or a usable
    holdout of at least two groups, the score is unavailable and only copy
    diagnostics are reported: with one holdout group, chance alone can
    cancel the privacy loss of blurred copies and report a perfect score.

    Attributes:
        name: Display name used in serialized evaluation results.
        details: Attack counts, noise terms, copy diagnostics, sampling counts, and per-group summaries.
    """

    card_id: ClassVar[str] = "window-group-mia"
    description: ClassVar[str] = (
        "Protection against attacks that determine whether a sequence was in training data, "
        "from short windows of consecutive values"
    )
    tooltip_key: ClassVar[str] = "window_group_mia_info"
    template: ClassVar[str] = "jinja/components/window_group_mia.j2"
    report_section: ClassVar[Literal["time_series", "data_privacy"]] = "data_privacy"

    name: str = Field(
        default="Window Membership Inference Protection",
        description="Display name used in serialized evaluation results.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Attack counts, noise terms, copy diagnostics, sampling counts, and per-group summaries.",
    )

    @cached_property
    def jinja_context(self) -> dict[str, Any]:
        """Return score, details, card attributes, and chart figures for report rendering."""
        context = super().jinja_context
        details = self.details
        figures: list[FigureSpec] = []
        if details.get("distances", {}).get("training"):
            figures.append(
                (
                    "Distance to synthetic data",
                    generate_window_group_mia_distance_figure(details),
                    False,
                    "window_group_mia_distance_info",
                )
            )
        if "attack" in details:
            figures.append(
                (
                    "Groups flagged most",
                    generate_window_group_mia_group_figure(
                        details["per_group"], details["attack"]["holdout_flag_rate"]
                    ),
                    False,
                    "window_group_mia_groups_info",
                )
            )
        if details.get("closest_match"):
            figures.append(
                (
                    "Closest match",
                    generate_window_group_mia_closest_match_figure(details),
                    True,
                    "window_group_mia_closest_info",
                )
            )
        context["figures"] = self.figure_rows(figures)
        match = details.get("closest_match") or {}
        context["summary_columns"] = details.get("columns", [])
        context["summary_traces_per_column"] = window_group_mia_closest_traces_per_column(match)
        if match:
            context["closest_match_distance_per_value"] = match["distance"] / np.sqrt(
                details["window_size"] * len(details["columns"])
            )
        return context

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets,
        config: SafeSynthesizerParameters | None = None,
    ) -> WindowGroupMIA:
        """Evaluate window membership inference protection for paired time-series datasets.

        Report orchestration controls whether the optional metric runs.
        Calling this component directly always computes it with the supplied
        configuration or isolated defaults. Evaluation failures are returned
        in ``score.notes`` instead of aborting the full evaluation pipeline.

        Args:
            evaluation_datasets: Training, synthetic, and holdout (``test``) datasets.
            config: Optional pipeline and metric configuration.

        Returns:
            A component containing the score, diagnostic details, and notes.
        """
        cfg = WindowGroupMIA._resolve_config(config)
        # Optional metrics must fail independently so one diagnostic cannot
        # prevent the rest of the evaluation report from being produced.
        try:
            return WindowGroupMIA._evaluate(evaluation_datasets, cfg, config)
        except Exception as exc:
            logger.exception("Failed to compute Window Membership Inference Protection.")
            return WindowGroupMIA(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def _resolve_config(config: SafeSynthesizerParameters | None) -> WindowGroupMIAParameters:
        """Return configured metric parameters or an isolated default model."""
        if config is None:
            return WindowGroupMIAParameters()
        return config.evaluation.time_series.window_group_mia

    @staticmethod
    def _evaluate(
        datasets: EvaluationDatasets,
        cfg: WindowGroupMIAParameters,
        config: SafeSynthesizerParameters | None,
    ) -> WindowGroupMIA:
        """Build windows for each split and score the attack, or report copy diagnostics only.

        Args:
            datasets: Training, synthetic, and holdout datasets.
            cfg: Resolved metric parameters.
            config: Optional top-level configuration used for inherited columns.

        Returns:
            A scored component, or an unavailable component with explanatory
            notes and whatever diagnostics could be computed.
        """
        timestamp_column = config.time_series.timestamp_column if config is not None else None
        group_column, group_error = resolve_group_column(datasets, config)
        if group_error is not None:
            return WindowGroupMIA(score=EvaluationScore(notes=group_error))

        columns, column_error, excluded_columns = select_value_columns(
            datasets, cfg.value_columns, timestamp_column, group_column, "window membership inference"
        )
        if column_error is not None:
            return WindowGroupMIA(score=EvaluationScore(notes=column_error))
        if not columns:
            return WindowGroupMIA(
                score=EvaluationScore(
                    notes=WindowGroupMIA._with_excluded_note("No shared numeric value columns.", excluded_columns)
                ),
                details={"excluded_columns": excluded_columns},
            )
        if cfg.value_columns is None and len(columns) > cfg.max_columns:
            columns = sorted(random.Random(_SAMPLING_SEED).sample(columns, k=cfg.max_columns))

        window = cfg.window_size
        stride = max(window // 2, 1)
        training = WindowGroupMIA._sequences(datasets.training, columns, group_column, timestamp_column)
        synthetic = WindowGroupMIA._sequences(datasets.synthetic, columns, group_column, timestamp_column)
        training_windows = WindowGroupMIA._balanced(training, window, stride, cfg.max_windows)
        synthetic_windows = WindowGroupMIA._every_step(synthetic, window)
        details: dict[str, Any] = {
            "timestamp_column": timestamp_column,
            "group_column": group_column,
            "columns": columns,
            "excluded_columns": excluded_columns,
            "window_size": window,
            "stride": stride,
        }
        if len(training_windows) == 0 or len(synthetic_windows) == 0:
            details["evaluation_mode"] = "unavailable"
            details["counts"] = WindowGroupMIA._counts(training_windows, synthetic_windows)
            return WindowGroupMIA(
                score=EvaluationScore(
                    notes=WindowGroupMIA._with_excluded_note(
                        f"Training and synthetic data each need at least one complete {window}-step window.",
                        excluded_columns,
                    )
                ),
                details=details,
            )

        mean, scale = WindowGroupMIA._standardizer(training_windows.values)
        support = (synthetic_windows.values - mean) / scale
        training_distances, matches = WindowGroupMIA._nearest_with_index(
            (training_windows.values - mean) / scale, support
        )
        details["exact_copy_rate"] = float(np.mean(training_distances == 0.0))
        details["closest_match"] = WindowGroupMIA._closest_match(
            training_windows, synthetic_windows, training_distances, matches, len(columns)
        )
        details["distances"] = {"training": training_distances.tolist()}

        holdout, holdout_reason = WindowGroupMIA._holdout_sequences(datasets, columns, group_column, timestamp_column)
        holdout_windows = (
            None if holdout is None else WindowGroupMIA._balanced(holdout, window, stride, cfg.max_windows)
        )
        details["counts"] = WindowGroupMIA._counts(training_windows, synthetic_windows, holdout_windows)
        if holdout_reason is None and holdout_windows is not None:
            holdout_reason = WindowGroupMIA._too_few_windows(
                training_windows, holdout_windows
            ) or WindowGroupMIA._too_few_holdout_groups(holdout_windows)
        if holdout_reason is not None or holdout_windows is None:
            details["evaluation_mode"] = "copy_diagnostics_only"
            details["per_group"] = WindowGroupMIA._per_group(training_windows, training_distances, None)
            return WindowGroupMIA(
                score=EvaluationScore(
                    notes=WindowGroupMIA._with_excluded_note(
                        f"{holdout_reason} Reporting copy diagnostics only.", excluded_columns
                    )
                ),
                details=details,
            )

        holdout_values = (holdout_windows.values - mean) / scale
        holdout_distances = WindowGroupMIA._nearest(holdout_values, support)
        details["distances"]["holdout"] = holdout_distances.tolist()
        flags = WindowGroupMIA._flag_counts(training_distances, holdout_distances)
        overlap = window / stride
        offset = WindowGroupMIA._holdout_offset(training, holdout_values, mean, scale, window, stride, cfg.max_windows)
        loss = flags.privacy_loss
        noise = flags.binomial_noise(overlap) + offset
        debiased_loss = float(np.sqrt(max(loss**2 - noise, 0.0))) if loss > 0 else 0.0
        holdout_exact_rate = float(np.mean(holdout_distances == 0.0))
        exact_loss = flags.exact_copy_loss(max(details["exact_copy_rate"] - holdout_exact_rate, 0.0))
        scored_loss = max(debiased_loss, exact_loss)
        protection = float(np.exp(-scored_loss))
        score = EvaluationScore.finalize_grade(raw_score=protection, score=10.0 * protection, is_privacy=True)
        score.notes = WindowGroupMIA._with_excluded_note(None, excluded_columns)

        details.update(
            {
                "evaluation_mode": "full",
                "attack": {
                    "threshold": flags.threshold,
                    "flagged_training_windows": flags.flagged_training,
                    "flagged_holdout_windows": flags.flagged_holdout,
                    "training_flag_rate": flags.flagged_training / flags.training,
                    "holdout_flag_rate": flags.flagged_holdout / flags.holdout,
                    "privacy_loss": loss,
                    "binomial_noise": flags.binomial_noise(overlap),
                    "holdout_offset_noise": offset,
                    "privacy_loss_beyond_chance": debiased_loss,
                    "holdout_exact_match_rate": holdout_exact_rate,
                    "exact_copy_privacy_loss": exact_loss,
                    "scored_privacy_loss": scored_loss,
                },
                "per_group": WindowGroupMIA._per_group(training_windows, training_distances, flags.threshold),
            }
        )
        return WindowGroupMIA(score=score, details=details)

    @staticmethod
    def _with_excluded_note(note: str | None, excluded_columns: list[str]) -> str | None:
        """Append the note naming numeric training columns left out of evaluation, if any."""
        notes = [part for part in (note, excluded_columns_note(excluded_columns)) if part]
        return " ".join(notes) if notes else None

    @staticmethod
    def _holdout_sequences(
        datasets: EvaluationDatasets,
        columns: list[str],
        group_column: str | None,
        timestamp_column: str | None,
    ) -> tuple[list[_Sequence] | None, str | None]:
        """Return holdout sequences, or a user-facing reason the holdout can't be used."""
        if group_column is None:
            return None, (
                "Window membership inference needs a group column (data.group_training_examples_by) so the "
                "holdout contains whole sequences."
            )
        holdout = datasets.test
        if holdout is None or holdout.empty:
            return None, "No holdout data is available."
        missing = [column for column in [group_column, *columns] if column not in holdout]
        if missing:
            return None, "The holdout data is missing columns: " + ", ".join(repr(column) for column in missing) + "."
        return WindowGroupMIA._sequences(holdout, columns, group_column, timestamp_column), None

    @staticmethod
    def _too_few_windows(training: _Windows, holdout: _Windows) -> str | None:
        if min(len(training), len(holdout)) >= WINDOW_GROUP_MIA_MIN_WINDOWS:
            return None
        return (
            f"The score needs at least {WINDOW_GROUP_MIA_MIN_WINDOWS} training and holdout windows; "
            f"found {len(training)} training and {len(holdout)} holdout windows."
        )

    @staticmethod
    def _too_few_holdout_groups(holdout: _Windows) -> str | None:
        groups = len(set(holdout.groups))
        if groups >= WINDOW_GROUP_MIA_MIN_HOLDOUT_GROUPS:
            return None
        return (
            f"The score needs at least {WINDOW_GROUP_MIA_MIN_HOLDOUT_GROUPS} holdout groups; found {groups}. "
            f"With {groups} holdout {'group' if groups == 1 else 'groups'}, chance can hide copying."
        )

    @staticmethod
    def _sequences(
        df: pd.DataFrame,
        columns: list[str],
        group_column: str | None,
        timestamp_column: str | None,
    ) -> list[_Sequence]:
        """Split a dataset into time-ordered groups and fill gaps by linear interpolation.

        Every group in ``df`` becomes a sequence, not only the groups shared
        with another dataset, because training, holdout, and synthetic groups
        are different sequences by design.
        """
        labels = [None] if group_column is None else sorted(df[group_column].dropna().unique(), key=group_sort_key)
        sequences = []
        for label in labels:
            frame = group_frame(df, group_column, label, timestamp_column)
            values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            group = None if label is None else str(label)
            sequences.append(_Sequence(group=group, values=WindowGroupMIA._interpolate(values)))
        return sequences

    @staticmethod
    def _interpolate(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fill missing and non-finite values within each column linearly, constant beyond the ends.

        Columns with fewer than two finite values are left as they are, so
        their windows stay incomplete and are not compared.
        """
        filled = np.where(np.isfinite(values), values, np.nan)
        positions = np.arange(len(values))
        for column in range(values.shape[1]):
            finite = np.isfinite(filled[:, column])
            if 2 <= finite.sum() < len(values):
                filled[:, column] = np.interp(positions, positions[finite], filled[finite, column])
        return filled

    @staticmethod
    def _window_starts(values: NDArray[np.float64], window: int, stride: int) -> NDArray[np.int64]:
        """Starting positions, every ``stride`` steps, of windows with every value finite."""
        if len(values) < window:
            return np.empty(0, dtype=np.int64)
        complete = np.isfinite(values).all(axis=1).astype(int)
        finite_in_window = np.convolve(complete, np.ones(window, dtype=int), mode="valid")
        starts = np.arange(0, len(finite_in_window), stride)
        return starts[finite_in_window[starts] == window]

    @staticmethod
    def _balanced(sequences: list[_Sequence], window: int, stride: int, limit: int) -> _Windows:
        """Build at most ``limit`` windows every ``stride`` steps, round robin across groups.

        One generator with a fixed seed shuffles the window positions within
        each group, then the order in which the round robin visits the groups,
        so repeated runs sample the same windows. When ``limit`` ends a round
        early, or there are more groups than ``limit``, every group has the
        same chance of a window, whatever its name. The selected windows are
        returned round by round in the order of ``sequences``.
        """
        rng = np.random.default_rng(_SAMPLING_SEED)
        queues: list[tuple[int, NDArray[np.int64]]] = []
        total = 0
        for index, sequence in enumerate(sequences):
            starts = WindowGroupMIA._window_starts(sequence.values, window, stride)
            total += len(starts)
            if len(starts):
                queues.append((index, starts[rng.permutation(len(starts))]))
        queues = [queues[i] for i in rng.permutation(len(queues))]
        selected: list[tuple[int, int, int]] = []
        depth = 0
        while queues and len(selected) < limit:
            for index, starts in queues:
                if len(selected) < limit:
                    selected.append((depth, index, int(starts[depth])))
            depth += 1
            queues = [(index, starts) for index, starts in queues if len(starts) > depth]
        selected.sort()
        return WindowGroupMIA._build(sequences, [(index, start) for _, index, start in selected], total, window)

    @staticmethod
    def _every_step(
        sequences: list[_Sequence], window: int, limit: int = WINDOW_GROUP_MIA_MAX_SYNTHETIC_WINDOWS
    ) -> _Windows:
        """Build the windows starting at every step, or a reproducible sample of ``limit`` of them."""
        per_sequence = [WindowGroupMIA._window_starts(sequence.values, window, 1) for sequence in sequences]
        offsets = np.cumsum([0, *(len(starts) for starts in per_sequence)])
        total = int(offsets[-1])
        chosen = np.arange(total)
        if total > limit:
            chosen = np.random.default_rng(_SAMPLING_SEED).choice(total, limit, replace=False)
        indices = np.searchsorted(offsets, chosen, side="right") - 1
        selected = [
            (int(index), int(per_sequence[index][position - offsets[index]]))
            for index, position in zip(indices, chosen, strict=True)
        ]
        return WindowGroupMIA._build(sequences, selected, total, window)

    @staticmethod
    def _build(sequences: list[_Sequence], selected: list[tuple[int, int]], total: int, window: int) -> _Windows:
        """Materialize the selected ``(sequence, start)`` windows, flattened time-major across columns."""
        if not selected:
            return _Windows(values=np.empty((0, 0)), groups=[], starts=[], total=total)
        offsets = np.arange(window)
        rows = [sequences[index].values[start + offsets].reshape(-1) for index, start in selected]
        return _Windows(
            values=np.vstack(rows),
            groups=[sequences[index].group for index, _ in selected],
            starts=[start for _, start in selected],
            total=total,
        )

    @staticmethod
    def _standardizer(training: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Center and scale every window feature by the sampled training windows."""
        scale = training.std(axis=0)
        return training.mean(axis=0), np.where(scale <= _MIN_SCALE, 1.0, scale)

    @staticmethod
    def _nearest(query: NDArray[np.float64], support: NDArray[np.float64]) -> NDArray[np.float64]:
        """Euclidean distance from each query window to its nearest support window."""
        return WindowGroupMIA._nearest_with_index(query, support)[0]

    @staticmethod
    def _nearest_with_index(
        query: NDArray[np.float64], support: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Distance from each query window to its nearest support window, and that window's index.

        The nearest window is found with the inner-product form in chunks, then
        its distance is recomputed directly so exact copies come out as exactly 0.
        """
        support_norms = np.einsum("ij,ij->i", support, support)
        distances = np.empty(len(query))
        indices = np.empty(len(query), dtype=np.int64)
        for start in range(0, len(query), _DISTANCE_CHUNK):
            block = query[start : start + _DISTANCE_CHUNK]
            # macOS Accelerate raises spurious divide-by-zero warnings in matmul on finite input.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                squared = support_norms[None, :] - 2.0 * block @ support.T
            best = np.argmin(squared, axis=1)
            indices[start : start + _DISTANCE_CHUNK] = best
            distances[start : start + _DISTANCE_CHUNK] = np.sqrt(np.sum((block - support[best]) ** 2, axis=1))
        return distances, indices

    @staticmethod
    def _flag_counts(training: NDArray[np.float64], holdout: NDArray[np.float64]) -> _FlagCounts:
        threshold = float(np.quantile(holdout, WINDOW_GROUP_MIA_FLAG_RATE))
        return _FlagCounts(
            flagged_training=int(np.sum(training <= threshold)),
            training=len(training),
            flagged_holdout=int(np.sum(holdout <= threshold)),
            holdout=len(holdout),
            threshold=threshold,
        )

    @staticmethod
    def _holdout_offset(
        training: list[_Sequence],
        holdout: NDArray[np.float64],
        mean: NDArray[np.float64],
        scale: NDArray[np.float64],
        window: int,
        stride: int,
        max_windows: int,
    ) -> float:
        """Chance variance of the privacy loss from how the holdout groups differ from training groups.

        The training groups are split in half several times. One half plays
        the training data and the other the synthetic data, and the attack runs
        against the real holdout. Nothing is copied, so the privacy loss in
        each split, minus its binomial noise, measures the holdout's own offset.
        The synthetic data never enters, so a copied group can't inflate it.
        """
        usable = [
            i
            for i, sequence in enumerate(training)
            if len(WindowGroupMIA._window_starts(sequence.values, window, stride))
        ]
        if len(usable) < 2:
            return 0.0
        rng = np.random.default_rng(_SAMPLING_SEED)
        excess = []
        for _ in range(_OFFSET_SPLITS):
            pseudo_synthetic = set(rng.choice(usable, len(usable) // 2, replace=False).tolist())
            members = [training[i] for i in usable if i not in pseudo_synthetic]
            reference = [training[i] for i in sorted(pseudo_synthetic)]
            member_windows = WindowGroupMIA._balanced(members, window, stride, max_windows)
            reference_windows = WindowGroupMIA._every_step(reference, window)
            if len(member_windows) == 0 or len(reference_windows) == 0:
                continue
            support = (reference_windows.values - mean) / scale
            flags = WindowGroupMIA._flag_counts(
                WindowGroupMIA._nearest((member_windows.values - mean) / scale, support),
                WindowGroupMIA._nearest(holdout, support),
            )
            excess.append(flags.privacy_loss**2 - flags.binomial_noise(window / stride))
        return max(float(np.mean(excess)), 0.0) if excess else 0.0

    @staticmethod
    def _counts(
        training_windows: _Windows,
        synthetic_windows: _Windows,
        holdout_windows: _Windows | None = None,
    ) -> dict[str, int]:
        """Window counts per split, before and after sampling, and the holdout groups with windows."""
        counts = {
            "training_windows_total": training_windows.total,
            "training_windows": len(training_windows),
            "synthetic_windows_total": synthetic_windows.total,
            "synthetic_windows": len(synthetic_windows),
            "holdout_windows_total": 0 if holdout_windows is None else holdout_windows.total,
            "holdout_windows": 0 if holdout_windows is None else len(holdout_windows),
        }
        counts["holdout_groups_with_windows"] = 0 if holdout_windows is None else len(set(holdout_windows.groups))
        return counts

    @staticmethod
    def _per_group(windows: _Windows, distances: NDArray[np.float64], threshold: float | None) -> list[dict[str, Any]]:
        """Summarize each training group's windows: count, exact copies, and flags."""
        by_group: dict[str | None, list[float]] = {}
        for group, distance in zip(windows.groups, distances, strict=True):
            by_group.setdefault(group, []).append(float(distance))
        summaries = []
        for group, values in sorted(by_group.items(), key=lambda item: str(item[0])):
            group_distances = np.array(values)
            summary: dict[str, Any] = {
                "group": group,
                "windows": len(values),
                "exact_copy_rate": float(np.mean(group_distances == 0.0)),
            }
            if threshold is not None:
                summary["flag_rate"] = float(np.mean(group_distances <= threshold))
            summaries.append(summary)
        return summaries

    @staticmethod
    def _closest_match(
        training: _Windows,
        synthetic: _Windows,
        distances: NDArray[np.float64],
        matches: NDArray[np.int64],
        column_count: int,
    ) -> dict[str, Any]:
        """The training window closest to a synthetic window, with the synthetic window's values by column.

        The training window's values are stored only for an exact copy, whose
        values the synthetic data already contains, so the details never
        carry training values into the report otherwise. Values are in the
        data's own units, after gaps are filled, as ``values[column][step]``.
        """
        index = int(np.argmin(distances))
        match = int(matches[index])
        closest: dict[str, Any] = {
            "group": training.groups[index],
            "start": training.starts[index],
            "distance": float(distances[index]),
            "exact_copy": bool(distances[index] == 0.0),
            "synthetic_values": synthetic.values[match].reshape(-1, column_count).T.tolist(),
        }
        if closest["exact_copy"]:
            closest["training_values"] = training.values[index].reshape(-1, column_count).T.tolist()
        return closest
