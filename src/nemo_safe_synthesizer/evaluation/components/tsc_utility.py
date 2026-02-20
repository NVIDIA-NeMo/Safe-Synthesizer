# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from functools import cached_property

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field

from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.autocorrelation_similarity import resolve_columns, sort_time_series
from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_dataset import EvaluationDataset
from ...evaluation.data_model.evaluation_score import EvaluationScore, Grade
from ...observability import get_logger

logger = get_logger(__name__)

LABEL_COLUMN = "label"


def _df_to_aeon(
    df: pd.DataFrame,
    group_col: str,
    feature_cols: list[str],
    timestamp_col: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a long-format DataFrame to aeon arrays (X, y).

    Args:
        df: Input DataFrame with one row per timestep.
        group_col: Column identifying each time series instance.
        feature_cols: Ordered list of feature (channel) columns.
        timestamp_col: Optional timestamp column for ordering within each group.

    Returns:
        X: shape (n_samples, n_channels, series_length)
        y: shape (n_samples,) of string labels
    """
    X_list = []
    y_list = []

    for group_id, group_df in df.groupby(group_col, sort=False):
        if timestamp_col and timestamp_col in group_df.columns:
            group_df = sort_time_series(group_df, timestamp_col, None)

        label = group_df[LABEL_COLUMN].iloc[0]
        # values: (series_length, n_channels)
        values = group_df[feature_cols].values.astype(float)
        # transpose to (n_channels, series_length)
        X_list.append(values.T)
        y_list.append(str(label))

    if not X_list:
        raise ValueError("No groups found in DataFrame during aeon conversion.")

    X = np.stack(X_list, axis=0)  # (n_samples, n_channels, series_length)
    y = np.array(y_list)
    return X, y


def _train_test(clf, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Fit a classifier and return error rate and timing."""
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_s = round(time.perf_counter() - t0, 4)

    t1 = time.perf_counter()
    y_pred = clf.predict(X_test)
    pred_s = round(time.perf_counter() - t1, 4)

    accuracy = float(np.mean(y_pred == y_test))
    error = round(1.0 - accuracy, 4)
    return {"error": error, "train_s": train_s, "pred_s": pred_s}


class TSCUtility(Component):
    name: str = Field(default="Time Series Classification Utility")
    details: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        ctx = super().jinja_context
        ctx["details"] = self.details
        return ctx

    def get_json(self) -> str:
        import json

        return json.dumps(self.details)

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> TSCUtility:
        # Only run for time series data
        is_timeseries = config.time_series.is_timeseries if config and getattr(config, "time_series", None) else False
        if not is_timeseries:
            return TSCUtility(score=EvaluationScore(grade=Grade.UNAVAILABLE))

        try:
            return TSCUtility._compute(evaluation_dataset, config)
        except Exception as exc:
            logger.warning(f"TSCUtility failed: {exc}")
            return TSCUtility(score=EvaluationScore(grade=Grade.UNAVAILABLE, notes=str(exc)))

    @staticmethod
    def _compute(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None
    ) -> TSCUtility:
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
        from aeon.classification.convolution_based import MiniRocketClassifier
        from aeon.classification.feature_based import Catch22Classifier

        # Check test data is available
        if evaluation_dataset.test is None or evaluation_dataset.test.empty:
            return TSCUtility(
                score=EvaluationScore(grade=Grade.UNAVAILABLE, notes="No test data available for TSC evaluation.")
            )

        reference = evaluation_dataset.reference
        synthetic = evaluation_dataset.output
        test = evaluation_dataset.test

        # Check label column exists in all DataFrames
        for split_name, split_df in [("reference", reference), ("synthetic", synthetic), ("test", test)]:
            if LABEL_COLUMN not in split_df.columns:
                return TSCUtility(
                    score=EvaluationScore(
                        grade=Grade.UNAVAILABLE,
                        notes=f"Label column '{LABEL_COLUMN}' not found in {split_name} data.",
                    )
                )

        # Infer columns from config
        group_col: str | None = None
        timestamp_col: str | None = None
        if config:
            group_col = getattr(config.data, "group_training_examples_by", None)
            timestamp_col = getattr(config.time_series, "timestamp_column", None)

        if not group_col:
            return TSCUtility(
                score=EvaluationScore(
                    grade=Grade.UNAVAILABLE,
                    notes="group_training_examples_by is not set; cannot convert to aeon format.",
                )
            )

        # Resolve feature columns: numeric, excluding group/timestamp/label
        exclude = set(filter(None, [group_col, timestamp_col, LABEL_COLUMN]))
        feature_cols = resolve_columns(reference, None, exclude=exclude)
        if not feature_cols:
            return TSCUtility(
                score=EvaluationScore(
                    grade=Grade.UNAVAILABLE,
                    notes="No numeric feature columns found after excluding group/timestamp/label.",
                )
            )

        # Convert to aeon arrays
        X_orig, y_orig = _df_to_aeon(reference, group_col, feature_cols, timestamp_col)
        X_syn, y_syn = _df_to_aeon(synthetic, group_col, feature_cols, timestamp_col)
        X_test, y_test = _df_to_aeon(test, group_col, feature_cols, timestamp_col)

        # Augmented = original + synthetic
        X_aug = np.concatenate([X_orig, X_syn], axis=0)
        y_aug = np.concatenate([y_orig, y_syn], axis=0)

        n_train_original = len(X_orig)
        n_train_synthetic = len(X_syn)
        n_test = len(X_test)
        n_channels = X_orig.shape[1]
        series_length = X_orig.shape[2]

        classifiers = {
            "DTW": lambda: KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw", distance_params={"window": 0.1}),
            "MiniROCKET": lambda: MiniRocketClassifier(num_kernels=10_000, random_state=42),
            "Catch22": lambda: Catch22Classifier(catch24=True, random_state=42),
        }

        conditions = {
            "original": (X_orig, y_orig),
            "synthetic": (X_syn, y_syn),
            "augmented": (X_aug, y_aug),
        }

        results: dict[str, dict[str, dict]] = {}
        summary_rows = []

        for condition, (X_train, y_train) in conditions.items():
            results[condition] = {}
            for clf_name, clf_factory in classifiers.items():
                try:
                    clf = clf_factory()
                    metrics = _train_test(clf, X_train, y_train, X_test, y_test)
                    results[condition][clf_name] = metrics
                    summary_rows.append(
                        f"  {condition:10s} | {clf_name:12s} | error={metrics['error']:.4f} | train={metrics['train_s']}s | pred={metrics['pred_s']}s"
                    )
                except Exception as exc:
                    logger.warning(f"TSC [{condition}][{clf_name}] failed: {exc}")
                    results[condition][clf_name] = {"error": None, "train_s": None, "pred_s": None}
                    summary_rows.append(f"  {condition:10s} | {clf_name:12s} | FAILED: {exc}")

        # Log summary table
        logger.info("TSC Utility Results:")
        logger.info(f"  n_train_original={n_train_original}, n_train_synthetic={n_train_synthetic}, n_test={n_test}")
        logger.info(f"  n_channels={n_channels}, series_length={series_length}")
        for row in summary_rows:
            logger.info(row)

        # wandb logging
        try:
            import wandb
            if wandb.run is not None:
                wandb_metrics = {}
                for condition in ["original", "synthetic", "augmented"]:
                    for clf_name in ["DTW", "MiniROCKET", "Catch22"]:
                        error = results.get(condition, {}).get(clf_name, {}).get("error")
                        if error is not None:
                            wandb_metrics[f"tsc/{condition}/{clf_name}_error"] = error
                wandb.log(wandb_metrics)
        except Exception as exc:
            logger.debug(f"wandb logging skipped: {exc}")

        details = {
            "n_train_original": n_train_original,
            "n_train_synthetic": n_train_synthetic,
            "n_test": n_test,
            "n_channels": n_channels,
            "series_length": series_length,
            "label_column": LABEL_COLUMN,
            "group_column": group_col,
            "results": results,
        }

        return TSCUtility(score=EvaluationScore(grade=Grade.UNAVAILABLE), details=details)
