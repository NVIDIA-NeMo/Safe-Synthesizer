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

Classes:
    WindowGroupMIA: Component that computes and summarizes the window
        membership inference protection score.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from ...artifacts.analyzers.field_features import FieldType
from ...config.evaluate import WINDOW_GROUP_MIA_MIN_WINDOWS, WindowGroupMIAParameters
from ...config.parameters import SafeSynthesizerParameters
from ...defaults import PSEUDO_GROUP_COLUMN
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from .component import Component

logger = get_logger(__name__)

WINDOW_GROUP_MIA_FLAG_RATE = 0.05
WINDOW_GROUP_MIA_MAX_SYNTHETIC_WINDOWS = 50_000
_OFFSET_SPLITS = 6
_SAMPLING_SEED = 2112
_DISTANCE_CHUNK = 256
_HIGHEST_RISK_WINDOWS = 10
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


class WindowGroupMIA(Component):
    """Measure how well synthetic data protects training sequences from window membership inference.

    Training and holdout windows are compared with their nearest synthetic
    window. A window is flagged when it is closer than all but the closest 5%
    of holdout windows. The privacy loss is the log ratio of the training and
    holdout flag rates; chance variation from the binomial counts and from the
    holdout groups' own offset is removed before scoring
    ``10 * exp(-privacy_loss)``. Without a group column or a usable holdout,
    the score is unavailable and only copy diagnostics are reported.

    Attributes:
        name: Display name used in serialized evaluation results.
        details: Attack counts, noise terms, copy diagnostics, sampling counts, and per-group summaries.
    """

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
        """Return score and diagnostic details for report rendering."""
        context = super().jinja_context
        context["details"] = self.details
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
        group_column = config.data.group_training_examples_by if config is not None else None
        if (
            group_column == PSEUDO_GROUP_COLUMN
            and group_column not in datasets.training
            and group_column not in datasets.synthetic
        ):
            # Training injects this reserved column to reuse grouped sequence
            # infrastructure, but evaluation receives frames with it removed.
            group_column = None
        elif group_column is not None and (
            group_column not in datasets.training or group_column not in datasets.synthetic
        ):
            return WindowGroupMIA(
                score=EvaluationScore(notes=f"Configured group column {group_column!r} is missing from a dataset.")
            )

        columns, column_error = WindowGroupMIA._numeric_columns(datasets, cfg, timestamp_column, group_column)
        if column_error is not None:
            return WindowGroupMIA(score=EvaluationScore(notes=column_error))
        if not columns:
            return WindowGroupMIA(score=EvaluationScore(notes="No shared numeric value columns."))

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
            "window_size": window,
            "stride": stride,
            "flag_rate": WINDOW_GROUP_MIA_FLAG_RATE,
        }
        if len(training_windows) == 0 or len(synthetic_windows) == 0:
            details["evaluation_mode"] = "unavailable"
            details["counts"] = WindowGroupMIA._counts(training, synthetic, None, training_windows, synthetic_windows)
            return WindowGroupMIA(
                score=EvaluationScore(
                    notes=f"Training and synthetic data each need at least one complete {window}-step window."
                ),
                details=details,
            )

        mean, scale = WindowGroupMIA._standardizer(training_windows.values)
        support = (synthetic_windows.values - mean) / scale
        training_distances = WindowGroupMIA._nearest((training_windows.values - mean) / scale, support)
        details["exact_copy_rate"] = float(np.mean(training_distances == 0.0))

        holdout, holdout_reason = WindowGroupMIA._holdout_sequences(datasets, columns, group_column, timestamp_column)
        holdout_windows = (
            None if holdout is None else WindowGroupMIA._balanced(holdout, window, stride, cfg.max_windows)
        )
        details["counts"] = WindowGroupMIA._counts(
            training, synthetic, holdout, training_windows, synthetic_windows, holdout_windows
        )
        if holdout_reason is None and holdout_windows is not None:
            holdout_reason = WindowGroupMIA._too_few_windows(training_windows, holdout_windows)
        if holdout_reason is not None or holdout_windows is None:
            details["evaluation_mode"] = "copy_diagnostics_only"
            details["per_group"] = WindowGroupMIA._per_group(training_windows, training_distances, None)
            return WindowGroupMIA(
                score=EvaluationScore(notes=f"{holdout_reason} Reporting copy diagnostics only."),
                details=details,
            )

        holdout_values = (holdout_windows.values - mean) / scale
        flags = WindowGroupMIA._flag_counts(training_distances, WindowGroupMIA._nearest(holdout_values, support))
        overlap = window / stride
        offset = WindowGroupMIA._holdout_offset(training, holdout_values, mean, scale, window, stride, cfg.max_windows)
        loss = flags.privacy_loss
        noise = flags.binomial_noise(overlap) + offset
        debiased_loss = float(np.sqrt(max(loss**2 - noise, 0.0))) if loss > 0 else 0.0
        protection = float(np.exp(-debiased_loss))
        score = EvaluationScore.finalize_grade(raw_score=protection, score=10.0 * protection, is_privacy=True)
        if details["counts"]["holdout_groups_with_windows"] == 1:
            score.notes = "Only one holdout group was available, so this score can vary noticeably between runs."

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
                },
                "per_group": WindowGroupMIA._per_group(training_windows, training_distances, flags.threshold),
                "highest_risk_windows": WindowGroupMIA._highest_risk(training_windows, training_distances),
            }
        )
        return WindowGroupMIA(score=score, details=details)

    @staticmethod
    def _numeric_columns(
        datasets: EvaluationDatasets,
        cfg: WindowGroupMIAParameters,
        timestamp_column: str | None,
        group_column: str | None,
    ) -> tuple[list[str], str | None]:
        """Select shared numeric value columns combined into each window.

        Explicit ``value_columns`` retain their configured order and are used
        in full. Automatic selection is sorted and, above ``max_columns``,
        reduced to a reproducible sample. Timestamp and grouping columns are
        excluded even when their storage dtype is numeric.

        Args:
            datasets: Training and synthetic datasets with inferred field types.
            cfg: Resolved metric parameters.
            timestamp_column: Column used only to order observations.
            group_column: Column used only to separate sequences.

        Returns:
            Shared numeric value column names and an optional validation error.
        """
        if cfg.value_columns is not None:
            invalid_columns = WindowGroupMIA._invalid_value_columns(
                datasets,
                cfg.value_columns,
                timestamp_column,
                group_column,
            )
            if invalid_columns:
                return [], "Invalid window membership inference value columns: " + "; ".join(invalid_columns)
            return list(dict.fromkeys(cfg.value_columns)), None
        numeric = set(datasets.get_columns_of_type({FieldType.NUMERIC}, based_on="both"))
        numeric.difference_update([timestamp_column, group_column, PSEUDO_GROUP_COLUMN])
        columns = sorted(numeric)
        if len(columns) > cfg.max_columns:
            columns = sorted(random.Random(_SAMPLING_SEED).sample(columns, k=cfg.max_columns))
        return columns, None

    @staticmethod
    def _invalid_value_columns(
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
    def _sequences(
        df: pd.DataFrame,
        columns: list[str],
        group_column: str | None,
        timestamp_column: str | None,
    ) -> list[_Sequence]:
        """Split a dataset into time-ordered groups and fill gaps by linear interpolation.

        ``mergesort`` preserves input order for equal timestamps, which makes
        repeated runs stable without inventing a secondary ordering key.
        """
        if group_column is None:
            parts: list[tuple[str | None, pd.DataFrame]] = [(None, df)]
        else:
            labels = sorted(df[group_column].dropna().unique(), key=WindowGroupMIA._group_sort_key)
            parts = [(str(label), df[df[group_column] == label]) for label in labels]
        sequences = []
        for label, frame in parts:
            if timestamp_column and timestamp_column in frame:
                frame = frame.sort_values(timestamp_column, kind="mergesort")
            values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
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

        Window positions are shuffled within each group with a fixed seed, so
        repeated runs sample the same windows.
        """
        rng = np.random.default_rng(_SAMPLING_SEED)
        queues = []
        total = 0
        for index, sequence in enumerate(sequences):
            starts = WindowGroupMIA._window_starts(sequence.values, window, stride)
            total += len(starts)
            if len(starts):
                queues.append(deque((index, int(starts[i])) for i in rng.permutation(len(starts))))
        selected: list[tuple[int, int]] = []
        while queues and len(selected) < limit:
            for queue in queues:
                if len(selected) < limit:
                    selected.append(queue.popleft())
            queues = [queue for queue in queues if queue]
        return WindowGroupMIA._build(sequences, selected, total, window)

    @staticmethod
    def _every_step(
        sequences: list[_Sequence], window: int, limit: int = WINDOW_GROUP_MIA_MAX_SYNTHETIC_WINDOWS
    ) -> _Windows:
        """Build the windows starting at every step, or a reproducible sample of ``limit`` of them."""
        selected: list[tuple[int, int]] = []
        for index, sequence in enumerate(sequences):
            selected.extend((index, int(start)) for start in WindowGroupMIA._window_starts(sequence.values, window, 1))
        total = len(selected)
        if total > limit:
            chosen = np.random.default_rng(_SAMPLING_SEED).choice(total, limit, replace=False)
            selected = [selected[i] for i in chosen]
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
        """Euclidean distance from each query window to its nearest support window.

        The nearest window is found with the inner-product form in chunks, then
        its distance is recomputed directly so exact copies come out as exactly 0.
        """
        support_norms = np.einsum("ij,ij->i", support, support)
        distances = np.empty(len(query))
        for start in range(0, len(query), _DISTANCE_CHUNK):
            block = query[start : start + _DISTANCE_CHUNK]
            # macOS Accelerate raises spurious divide-by-zero warnings in matmul on finite input.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                squared = support_norms[None, :] - 2.0 * block @ support.T
            nearest = support[np.argmin(squared, axis=1)]
            distances[start : start + _DISTANCE_CHUNK] = np.sqrt(np.sum((block - nearest) ** 2, axis=1))
        return distances

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
        training: list[_Sequence],
        synthetic: list[_Sequence],
        holdout: list[_Sequence] | None,
        training_windows: _Windows,
        synthetic_windows: _Windows,
        holdout_windows: _Windows | None = None,
    ) -> dict[str, int]:
        """Group and window counts per split, before and after sampling."""
        counts = {
            "training_groups": len(training),
            "synthetic_groups": len(synthetic),
            "holdout_groups": 0 if holdout is None else len(holdout),
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
        """Summarize each training group's windows: count, exact copies, flags, and median distance."""
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
                "median_distance": float(np.median(group_distances)),
            }
            if threshold is not None:
                summary["flag_rate"] = float(np.mean(group_distances <= threshold))
            summaries.append(summary)
        return summaries

    @staticmethod
    def _highest_risk(windows: _Windows, distances: NDArray[np.float64]) -> list[dict[str, Any]]:
        """The training windows closest to a synthetic window, closest first."""
        order = np.argsort(distances, kind="stable")[:_HIGHEST_RISK_WINDOWS]
        return [
            {"group": windows.groups[i], "start": windows.starts[i], "distance": float(distances[i])} for i in order
        ]

    @staticmethod
    def _group_sort_key(value: Any) -> tuple[str, str]:
        """Return a stable ordering key for heterogeneous group values."""
        type_name = f"{type(value).__module__}.{type(value).__qualname__}"
        return type_name, str(value)
