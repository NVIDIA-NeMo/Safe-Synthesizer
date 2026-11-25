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
memorization and 0 means every training window is flagged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from ...config.evaluate import (
    WINDOW_GROUP_MIA_MIN_WINDOWS,
    WindowGroupMIAParameters,
)
from ...config.evaluation_report_data import WindowMembershipInferenceMetric
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
    window_value_count,
)

logger = get_logger(__name__)

WINDOW_GROUP_MIA_FLAG_RATE = 0.05
WINDOW_GROUP_MIA_MIN_HOLDOUT_GROUPS = 2
WINDOW_GROUP_MIA_MAX_SYNTHETIC_WINDOWS = 50_000
_OFFSET_SPLITS = 6
_SAMPLING_SEED = 2112
_DISTANCE_CHUNK = 256
_MIN_SCALE = 1e-12


@dataclass(frozen=True)
class _Sequence:
    """One group's numeric values in time order, with gaps filled by interpolation.

    ``values`` has shape ``(steps, columns)``, one row per time step, before
    any windows are cut from it. ``group`` keeps the original label, so labels
    such as ``1`` and ``"1"`` stay separate groups until they are displayed.
    """

    group: Any
    values: NDArray[np.float64]


@dataclass(frozen=True)
class _Windows:
    """Windows cut from sequences and flattened, with the group each came from.

    ``values`` has shape ``(len(groups), window * columns)``. Each row is one
    window flattened time-major, so its first ``columns`` values are the
    window's first step. ``total`` counts the candidate windows before
    sampling, so ``values.shape[0] == len(groups) <= total``.
    """

    values: NDArray[np.float64]
    groups: list[Any]
    total: int

    def __len__(self) -> int:
        return len(self.groups)


@dataclass(frozen=True)
class _FlagCounts:
    """Training and holdout windows at or below the holdout's flag-rate quantile of squared distance."""

    flagged_training: int
    training: int
    flagged_holdout: int
    holdout: int
    squared_threshold: float

    @property
    def holdout_rate(self) -> float:
        """Share of holdout windows flagged, with half a window added so it stays above 0."""
        return (self.flagged_holdout + 0.5) / (self.holdout + 1)

    @property
    def privacy_loss(self) -> float:
        """``log(TPR / FPR)``, with half a window added to each flagged count so it stays finite."""
        training_rate = (self.flagged_training + 0.5) / (self.training + 1)
        return float(np.log(training_rate) - np.log(self.holdout_rate))

    def binomial_noise(self, overlap: float) -> float:
        """Delta-method variance of ``privacy_loss`` for two binomial rates, times ``overlap``.

        ``overlap`` is ``window / stride`` for the training and holdout
        windows, the number of those windows that contain each point.
        """
        # A heuristic, not an exact accounting. Windows that share points have
        # correlated flags. If flags were fully correlated within blocks of
        # ``overlap`` windows and independent across blocks, the variance of each
        # flag rate would be ``overlap`` times the independent-window variance.
        # The true factor depends on how strongly the flags of overlapping
        # windows are correlated, which is not estimated here.
        terms = (
            1 / (self.flagged_training + 0.5)
            - 1 / (self.training + 1)
            + 1 / (self.flagged_holdout + 0.5)
            - 1 / (self.holdout + 1)
        )
        return overlap * terms

    def exact_copy_loss(self, excess_exact_rate: float) -> float:
        """Privacy loss implied by exact copies alone, which chance can't explain.

        If a share ``x`` (``excess_exact_rate``) of the training windows is
        certainly flagged and the rest are flagged at the holdout rate
        ``alpha``, the training flag rate divided by the holdout flag rate is
        ``(x + (1 - x) * alpha) / alpha``. This returns its log. Exact copies of
        continuous values almost never arise by chance, so this loss is not
        reduced by the noise terms.
        """
        holdout_rate = self.holdout_rate
        return float(np.log1p(excess_exact_rate * (1 - holdout_rate) / holdout_rate))

    def protection(self, loss: float) -> float:
        """Map a privacy loss to 0--1, from 1 at no loss to 0 when every training window is flagged.

        ``exp(-privacy_loss)`` is the holdout flag rate divided by the training
        flag rate, which falls to the holdout flag rate when every training
        window is flagged. That rate is rescaled to 0, and so is the largest
        exact-copy loss, ``log(1 / holdout_rate)``.
        """
        holdout_rate = self.holdout_rate
        return float(np.clip((np.exp(-loss) - holdout_rate) / (1 - holdout_rate), 0.0, 1.0))


class WindowGroupMIA(TimeSeriesComponent[WindowMembershipInferenceMetric]):
    """Measure how well synthetic data protects training sequences from window membership inference.

    Training and holdout windows are compared with their nearest synthetic
    window. A window is flagged when it is closer than all but the closest 5%
    of holdout windows. The privacy loss is the log ratio of the training and
    holdout flag rates. Chance variation from the binomial counts and from the
    holdout groups' own offset is removed before scoring
    ``10 * (exp(-privacy_loss) - alpha) / (1 - alpha)``, where ``alpha`` is the
    holdout flag rate, so 10 means equal flag rates and 0 means every training
    window is flagged. Exact copies beyond the holdout's own exact-match rate
    set a floor on the privacy loss that chance can't remove, so a small
    holdout can't hide them. Without a group column or a usable holdout of at
    least two groups, the score is unavailable and only copy diagnostics are
    reported. With one holdout group, chance alone can cancel the privacy loss
    of blurred copies and report a perfect score.

    Attributes:
        name: Display name used in serialized evaluation results.
        details: Flag rates, copy diagnostics, window counts, distances, and per-group summaries.
    """

    metric_key: ClassVar[str] = "window_membership_inference_protection"
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
        description="Flag rates, copy diagnostics, window counts, distances, and per-group summaries.",
    )

    @classmethod
    def report_context(cls, metric: WindowMembershipInferenceMetric) -> dict[str, Any]:
        """Return score, details, card attributes, and chart figures for report rendering."""
        context = super().report_context(metric)
        details = metric.details
        figures: list[FigureSpec] = []
        if details.distances is not None and details.distances.training:
            figures.append(
                (
                    "Distance to synthetic data",
                    generate_window_group_mia_distance_figure(details),
                    False,
                    "window_group_mia_distance_info",
                )
            )
        if details.attack is not None:
            figures.append(
                (
                    "Groups flagged most",
                    generate_window_group_mia_group_figure(details.per_group, details.attack.holdout_flag_rate),
                    False,
                    "window_group_mia_groups_info",
                )
            )
        if details.closest_match is not None:
            figures.append(
                (
                    "Closest match",
                    generate_window_group_mia_closest_match_figure(details),
                    True,
                    "window_group_mia_closest_info",
                )
            )
        context["figures"] = cls.figure_rows(figures)
        match = details.closest_match
        context["summary_columns"] = details.columns
        context["summary_traces_per_column"] = 2 if match is not None and match.training_values is not None else 1
        if match is not None and match.distance is not None:
            context["closest_match_distance_per_value"] = match.distance / np.sqrt(window_value_count(details))
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

        Training and holdout windows use a stride of half the window and are
        capped at ``cfg.max_windows`` each. Synthetic windows use a stride of 1,
        so a copy shifted off the training stride is still found, and are
        capped at ``WINDOW_GROUP_MIA_MAX_SYNTHETIC_WINDOWS``.

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
            return WindowGroupMIA._build_unavailable(group_error, {})

        columns, column_error, excluded_columns = select_value_columns(
            datasets, cfg.value_columns, timestamp_column, group_column, "window membership inference"
        )
        if column_error is not None:
            return WindowGroupMIA._build_unavailable(column_error, {})
        if not columns:
            return WindowGroupMIA._build_unavailable(
                "No shared numeric value columns.", {"excluded_columns": excluded_columns}
            )
        if cfg.value_columns is None and len(columns) > cfg.max_columns:
            columns = sorted(random.Random(_SAMPLING_SEED).sample(columns, k=cfg.max_columns))

        window = cfg.window_size
        stride = max(window // 2, 1)
        training = WindowGroupMIA._get_sequences(datasets.training, columns, group_column, timestamp_column)
        holdout, holdout_reason = WindowGroupMIA._get_holdout_sequences(
            datasets, columns, group_column, timestamp_column
        )
        synthetic = WindowGroupMIA._get_sequences(datasets.synthetic, columns, group_column, timestamp_column)
        training_windows = WindowGroupMIA._build_windows(training, window, stride, cfg.max_windows)
        holdout_windows = (
            None if holdout is None else WindowGroupMIA._build_windows(holdout, window, stride, cfg.max_windows)
        )
        synthetic_windows = WindowGroupMIA._build_windows(synthetic, window, 1, WINDOW_GROUP_MIA_MAX_SYNTHETIC_WINDOWS)
        details: dict[str, Any] = {
            "columns": columns,
            "excluded_columns": excluded_columns,
            "window_size": window,
            "counts": WindowGroupMIA._count_windows(training_windows, synthetic_windows, holdout_windows),
        }
        if len(training_windows) == 0 or len(synthetic_windows) == 0:
            details["evaluation_mode"] = "unavailable"
            return WindowGroupMIA._build_unavailable(
                f"Training and synthetic data each need at least one complete {window}-step window.", details
            )

        mean, scale = WindowGroupMIA._fit_standardizer(training_windows.values)
        support = (synthetic_windows.values - mean) / scale
        training_squared, matches = WindowGroupMIA._find_nearest_with_index(
            (training_windows.values - mean) / scale, support
        )
        # The nearest squared distance is recomputed directly with the same
        # standardization for both splits, so identical windows give exactly 0.
        details["exact_copy_rate"] = float(np.mean(training_squared == 0.0))
        details["closest_match"] = WindowGroupMIA._find_closest_match(
            training_windows, synthetic_windows, training_squared, matches, len(columns)
        )
        details["distances"] = {"training": np.sqrt(training_squared).tolist()}

        if holdout_windows is not None:
            holdout_reason = WindowGroupMIA._check_too_few_windows(
                training_windows, holdout_windows
            ) or WindowGroupMIA._check_too_few_holdout_groups(holdout_windows)
        if holdout_windows is None or holdout_reason is not None:
            details["evaluation_mode"] = "copy_diagnostics_only"
            details["per_group"] = WindowGroupMIA._summarize_groups(training_windows, training_squared, None)
            return WindowGroupMIA._build_unavailable(f"{holdout_reason} Reporting copy diagnostics only.", details)

        holdout_values = (holdout_windows.values - mean) / scale
        holdout_squared = WindowGroupMIA._find_nearest(holdout_values, support)
        details["distances"]["holdout"] = np.sqrt(holdout_squared).tolist()
        flags = WindowGroupMIA._count_flags(training_squared, holdout_squared)
        offset = WindowGroupMIA._estimate_holdout_offset(
            training, holdout_values, mean, scale, window, stride, cfg.max_windows
        )
        loss = flags.privacy_loss
        noise = flags.binomial_noise(window / stride) + offset
        debiased_loss = float(np.sqrt(max(loss**2 - noise, 0.0))) if loss > 0 else 0.0
        holdout_exact_rate = float(np.mean(holdout_squared == 0.0))
        exact_loss = flags.exact_copy_loss(max(details["exact_copy_rate"] - holdout_exact_rate, 0.0))
        protection = flags.protection(max(debiased_loss, exact_loss))
        score = EvaluationScore.finalize_grade(raw_score=protection, score=10.0 * protection, is_privacy=True)
        score.notes = excluded_columns_note(excluded_columns)

        details.update(
            {
                "evaluation_mode": "full",
                "attack": {
                    "threshold": float(np.sqrt(flags.squared_threshold)),
                    "training_flag_rate": flags.flagged_training / flags.training,
                    "holdout_flag_rate": flags.flagged_holdout / flags.holdout,
                    "privacy_loss_beyond_chance": debiased_loss,
                    "holdout_exact_match_rate": holdout_exact_rate,
                    "exact_copy_privacy_loss": exact_loss,
                },
                "per_group": WindowGroupMIA._summarize_groups(
                    training_windows, training_squared, flags.squared_threshold
                ),
            }
        )
        return WindowGroupMIA(score=score, details=details)

    @staticmethod
    def _build_unavailable(reason: str, details: dict[str, Any]) -> WindowGroupMIA:
        """Return a component without a score, noting ``reason`` and any numeric columns left out."""
        notes = [reason, excluded_columns_note(details.get("excluded_columns", []))]
        return WindowGroupMIA(score=EvaluationScore(notes=" ".join(note for note in notes if note)), details=details)

    @staticmethod
    def _get_holdout_sequences(
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
        return WindowGroupMIA._get_sequences(holdout, columns, group_column, timestamp_column), None

    @staticmethod
    def _check_too_few_windows(training: _Windows, holdout: _Windows) -> str | None:
        """Return a user-facing reason if either split has too few windows to score, otherwise None."""
        if min(len(training), len(holdout)) >= WINDOW_GROUP_MIA_MIN_WINDOWS:
            return None
        return (
            f"The score needs at least {WINDOW_GROUP_MIA_MIN_WINDOWS} training and {WINDOW_GROUP_MIA_MIN_WINDOWS} "
            f"holdout windows and found {len(training)} training and {len(holdout)} holdout windows."
        )

    @staticmethod
    def _check_too_few_holdout_groups(holdout: _Windows) -> str | None:
        """Return a user-facing reason if the holdout windows come from too few groups, otherwise None."""
        groups = len(set(holdout.groups))
        if groups >= WINDOW_GROUP_MIA_MIN_HOLDOUT_GROUPS:
            return None
        return f"The score needs at least {WINDOW_GROUP_MIA_MIN_HOLDOUT_GROUPS} holdout groups and found {groups}."

    @staticmethod
    def _get_sequences(
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
            values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            sequences.append(_Sequence(group=label, values=WindowGroupMIA._interpolate(values)))
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
    def _find_window_starts(values: NDArray[np.float64], window: int, stride: int) -> NDArray[np.int64]:
        """Starting positions, every ``stride`` steps, of windows with every value finite.

        After ``_interpolate``, only columns with fewer than two finite values
        still hold non-finite values, so this skips every window of a group
        with such a column.
        """
        if len(values) < window:
            return np.empty(0, dtype=np.int64)
        complete = np.isfinite(values).all(axis=1).astype(int)
        finite_in_window = np.convolve(complete, np.ones(window, dtype=int), mode="valid")
        starts = np.arange(0, len(finite_in_window), stride)
        return starts[finite_in_window[starts] == window]

    @staticmethod
    def _build_windows(sequences: list[_Sequence], window: int, stride: int, limit: int) -> _Windows:
        """Build the complete windows that start every ``stride`` steps, or a reproducible sample of ``limit``.

        The sample is uniform over all candidate windows across groups, so
        longer groups contribute more windows and group names don't affect
        which windows are chosen. Windows are returned in the order of
        ``sequences``, then by start.
        """
        per_sequence = [WindowGroupMIA._find_window_starts(sequence.values, window, stride) for sequence in sequences]
        offsets = np.cumsum([0, *(len(starts) for starts in per_sequence)])
        total = int(offsets[-1])
        chosen = np.arange(total)
        if total > limit:
            chosen = np.sort(np.random.default_rng(_SAMPLING_SEED).choice(total, limit, replace=False))
        if not len(chosen):
            return _Windows(values=np.empty((0, 0)), groups=[], total=total)
        indices = np.searchsorted(offsets, chosen, side="right") - 1
        steps = np.arange(window)
        rows = [
            sequences[index].values[per_sequence[index][position - offsets[index]] + steps].reshape(-1)
            for index, position in zip(indices, chosen, strict=True)
        ]
        return _Windows(values=np.vstack(rows), groups=[sequences[index].group for index in indices], total=total)

    @staticmethod
    def _fit_standardizer(training: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the mean and scale of each flattened window value over the sampled training windows.

        Each flattened position is one column at one step of the window, so
        every column is standardized separately at each step.
        """
        scale = training.std(axis=0)
        return training.mean(axis=0), np.where(scale <= _MIN_SCALE, 1.0, scale)

    @staticmethod
    def _find_nearest(query: NDArray[np.float64], support: NDArray[np.float64]) -> NDArray[np.float64]:
        """Squared Euclidean distance from each query window to its nearest support window."""
        return WindowGroupMIA._find_nearest_with_index(query, support)[0]

    @staticmethod
    def _find_nearest_with_index(
        query: NDArray[np.float64], support: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Squared distance from each query window to its nearest support window, and that window's index.

        The nearest window is found with the inner-product form in chunks, then
        its squared distance is recomputed directly so exact copies come out as
        exactly 0.
        """
        support_norms = np.einsum("ij,ij->i", support, support)
        squared = np.empty(len(query))
        indices = np.empty(len(query), dtype=np.int64)
        for start in range(0, len(query), _DISTANCE_CHUNK):
            block = query[start : start + _DISTANCE_CHUNK]
            # macOS Accelerate raises spurious divide-by-zero warnings in matmul on finite input.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                partial = support_norms[None, :] - 2.0 * block @ support.T
            best = np.argmin(partial, axis=1)
            indices[start : start + _DISTANCE_CHUNK] = best
            squared[start : start + _DISTANCE_CHUNK] = np.sum((block - support[best]) ** 2, axis=1)
        return squared, indices

    @staticmethod
    def _count_flags(training: NDArray[np.float64], holdout: NDArray[np.float64]) -> _FlagCounts:
        """Count the windows at or below the holdout's flag-rate quantile of squared distance."""
        threshold = float(np.quantile(holdout, WINDOW_GROUP_MIA_FLAG_RATE))
        return _FlagCounts(
            flagged_training=int(np.sum(training <= threshold)),
            training=len(training),
            flagged_holdout=int(np.sum(holdout <= threshold)),
            holdout=len(holdout),
            squared_threshold=threshold,
        )

    @staticmethod
    def _estimate_holdout_offset(
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
            if len(WindowGroupMIA._find_window_starts(sequence.values, window, stride))
        ]
        if len(usable) < 2:
            return 0.0
        rng = np.random.default_rng(_SAMPLING_SEED)
        excess = []
        for _ in range(_OFFSET_SPLITS):
            pseudo_synthetic = set(rng.choice(usable, len(usable) // 2, replace=False).tolist())
            members = [training[i] for i in usable if i not in pseudo_synthetic]
            reference = [training[i] for i in sorted(pseudo_synthetic)]
            member_windows = WindowGroupMIA._build_windows(members, window, stride, max_windows)
            reference_windows = WindowGroupMIA._build_windows(
                reference, window, 1, WINDOW_GROUP_MIA_MAX_SYNTHETIC_WINDOWS
            )
            if len(member_windows) == 0 or len(reference_windows) == 0:
                continue
            support = (reference_windows.values - mean) / scale
            flags = WindowGroupMIA._count_flags(
                WindowGroupMIA._find_nearest((member_windows.values - mean) / scale, support),
                WindowGroupMIA._find_nearest(holdout, support),
            )
            excess.append(flags.privacy_loss**2 - flags.binomial_noise(window / stride))
        return max(float(np.mean(excess)), 0.0) if excess else 0.0

    @staticmethod
    def _count_windows(
        training_windows: _Windows,
        synthetic_windows: _Windows,
        holdout_windows: _Windows | None,
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
    def _summarize_groups(
        windows: _Windows, squared: NDArray[np.float64], squared_threshold: float | None
    ) -> list[dict[str, Any]]:
        """Summarize each training group's windows: count, exact copies, and flags."""
        by_group: dict[Any, list[float]] = {}
        for group, distance in zip(windows.groups, squared, strict=True):
            by_group.setdefault(group, []).append(float(distance))
        summaries = []
        for group, values in sorted(by_group.items(), key=lambda item: group_sort_key(item[0])):
            group_squared = np.array(values)
            summary: dict[str, Any] = {
                "group": None if group is None else str(group),
                "windows": len(values),
                "exact_copy_rate": float(np.mean(group_squared == 0.0)),
            }
            if squared_threshold is not None:
                summary["flag_rate"] = float(np.mean(group_squared <= squared_threshold))
            summaries.append(summary)
        return summaries

    @staticmethod
    def _find_closest_match(
        training: _Windows,
        synthetic: _Windows,
        squared: NDArray[np.float64],
        matches: NDArray[np.int64],
        column_count: int,
    ) -> dict[str, Any]:
        """The training window closest to a synthetic window, with the synthetic window's values by column.

        The training window's values are stored only for an exact copy, whose
        values the synthetic data already contains, so the details never
        carry training values into the report otherwise. Values are in the
        data's own units, after gaps are filled, as ``values[column][step]``.
        """
        index = int(np.argmin(squared))
        match = int(matches[index])
        closest: dict[str, Any] = {
            "distance": float(np.sqrt(squared[index])),
            "exact_copy": bool(squared[index] == 0.0),
            "synthetic_values": synthetic.values[match].reshape(-1, column_count).T.tolist(),
        }
        if closest["exact_copy"]:
            closest["training_values"] = training.values[index].reshape(-1, column_count).T.tolist()
        return closest
