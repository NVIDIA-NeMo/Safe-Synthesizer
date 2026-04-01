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
    trim_start: int = 0,
    target_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a long-format DataFrame to aeon arrays (X, y).

    Args:
        df: Input DataFrame with one row per timestep.
        group_col: Column identifying each time series instance.
        feature_cols: Ordered list of feature (channel) columns.
        timestamp_col: Optional timestamp column for ordering within each group.
        trim_start: Number of leading timesteps to drop from each group
            (e.g. 3 to align reference data with generated data that omits
            the prompt context rows).
        target_length: If set, truncate each group to exactly this many
            timesteps (after trim_start).  Groups shorter than target_length
            are dropped with a warning.

    Returns:
        X: shape (n_samples, n_channels, series_length)
        y: shape (n_samples,) of string labels
    """
    X_list = []
    y_list = []

    for group_id, group_df in df.groupby(group_col, sort=False):
        if timestamp_col and timestamp_col in group_df.columns:
            group_df = sort_time_series(group_df, timestamp_col, None)

        if trim_start > 0:
            group_df = group_df.iloc[trim_start:]

        if target_length is not None:
            if len(group_df) < target_length:
                continue
            group_df = group_df.iloc[:target_length]

        if group_df.empty:
            continue

        label = group_df[LABEL_COLUMN].iloc[0]
        values = group_df[feature_cols].values.astype(float)
        X_list.append(values.T)  # (n_channels, series_length)
        y_list.append(str(label))

    if not X_list:
        raise ValueError("No groups found in DataFrame during aeon conversion.")

    X = np.stack(X_list, axis=0)  # (n_samples, n_channels, series_length)
    y = np.array(y_list)
    return X, y


def _min_group_length(
    df: pd.DataFrame, group_col: str, trim_start: int = 0,
) -> int:
    """Return the minimum number of timesteps across all groups after trimming."""
    return int(df.groupby(group_col).size().min()) - trim_start


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
        evaluation_dataset: EvaluationDataset,
        config: SafeSynthesizerParameters | None = None,
        n_trials: int = 1,
    ) -> TSCUtility:
        # Only run for time series data
        is_timeseries = config.time_series.is_timeseries if config and getattr(config, "time_series", None) else False
        if not is_timeseries:
            return TSCUtility(score=EvaluationScore(grade=Grade.UNAVAILABLE))

        try:
            return TSCUtility._compute(evaluation_dataset, config, n_trials=n_trials)
        except Exception as exc:
            logger.warning(f"TSCUtility failed: {exc}")
            return TSCUtility(score=EvaluationScore(grade=Grade.UNAVAILABLE, notes=str(exc)))

    @staticmethod
    def _compute(
        evaluation_dataset: EvaluationDataset,
        config: SafeSynthesizerParameters | None,
        n_trials: int = 1,
    ) -> TSCUtility:
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

        # The generated data omits the first few context rows per group
        # (typically 3). Trim reference/test to match, then align all
        # datasets to the shortest group across all three splits.
        CONTEXT_ROWS = 3
        target_len = min(
            _min_group_length(reference, group_col, trim_start=CONTEXT_ROWS),
            _min_group_length(synthetic, group_col, trim_start=0),
            _min_group_length(test, group_col, trim_start=CONTEXT_ROWS),
        )
        if target_len <= 0:
            return TSCUtility(
                score=EvaluationScore(
                    grade=Grade.UNAVAILABLE,
                    notes=f"Series too short after alignment (target_length={target_len}).",
                )
            )

        logger.info(
            f"TSCUtility: aligning series — trimming {CONTEXT_ROWS} context rows from "
            f"reference/test, truncating all to length {target_len}"
        )

        X_orig, y_orig = _df_to_aeon(
            reference, group_col, feature_cols, timestamp_col,
            trim_start=CONTEXT_ROWS, target_length=target_len,
        )
        X_syn, y_syn = _df_to_aeon(
            synthetic, group_col, feature_cols, timestamp_col,
            trim_start=0, target_length=target_len,
        )
        X_test, y_test = _df_to_aeon(
            test, group_col, feature_cols, timestamp_col,
            trim_start=CONTEXT_ROWS, target_length=target_len,
        )

        # Augmented = original + synthetic
        X_aug = np.concatenate([X_orig, X_syn], axis=0)
        y_aug = np.concatenate([y_orig, y_syn], axis=0)

        n_train_original = len(X_orig)
        n_train_synthetic = len(X_syn)
        n_test = len(X_test)
        n_channels = X_orig.shape[1]
        series_length = X_orig.shape[2]

        classifier_factories = {
            "MiniROCKET": lambda seed: MiniRocketClassifier(n_kernels=10_000, random_state=seed),
            "Catch22": lambda seed: Catch22Classifier(catch24=True, random_state=seed),
        }

        conditions = {
            "original": (X_orig, y_orig),
            "synthetic": (X_syn, y_syn),
            "augmented": (X_aug, y_aug),
        }

        seeds = list(range(42, 42 + n_trials))
        logger.info(f"TSCUtility: running {n_trials} trial(s) with seeds {seeds}")

        # Accumulate per-trial metrics: trial_data[condition][clf_name] = [metrics, ...]
        trial_data: dict[str, dict[str, list[dict]]] = {
            cond: {clf: [] for clf in classifier_factories} for cond in conditions
        }

        for seed in seeds:
            for condition, (X_train, y_train) in conditions.items():
                for clf_name, clf_factory in classifier_factories.items():
                    try:
                        clf = clf_factory(seed)
                        metrics = _train_test(clf, X_train, y_train, X_test, y_test)
                        trial_data[condition][clf_name].append(metrics)
                        logger.debug(
                            f"  seed={seed} | {condition:10s} | {clf_name:12s} | error={metrics['error']:.4f}"
                        )
                    except Exception as exc:
                        logger.warning(f"TSC [seed={seed}][{condition}][{clf_name}] failed: {exc}")

        # Aggregate across trials
        results: dict[str, dict[str, dict]] = {}
        summary_rows = []

        for condition in conditions:
            results[condition] = {}
            for clf_name in classifier_factories:
                trials = trial_data[condition][clf_name]
                if not trials:
                    results[condition][clf_name] = {
                        "error": None, "error_std": None, "error_trials": [],
                        "train_s": None, "pred_s": None,
                    }
                    summary_rows.append(f"  {condition:10s} | {clf_name:12s} | ALL TRIALS FAILED")
                else:
                    errors = [t["error"] for t in trials]
                    mean_err = round(float(np.mean(errors)), 4)
                    std_err = round(float(np.std(errors)), 4) if len(errors) > 1 else 0.0
                    mean_train = round(float(np.mean([t["train_s"] for t in trials])), 4)
                    mean_pred = round(float(np.mean([t["pred_s"] for t in trials])), 4)
                    results[condition][clf_name] = {
                        "error": mean_err,
                        "error_std": std_err,
                        "error_trials": errors,
                        "train_s": mean_train,
                        "pred_s": mean_pred,
                    }
                    summary_rows.append(
                        f"  {condition:10s} | {clf_name:12s} | "
                        f"error={mean_err:.4f}±{std_err:.4f} (n={len(errors)}) | "
                        f"train={mean_train}s | pred={mean_pred}s"
                    )

        # Log summary table
        logger.info("TSC Utility Results:")
        logger.info(f"  n_train_original={n_train_original}, n_train_synthetic={n_train_synthetic}, n_test={n_test}")
        logger.info(f"  n_channels={n_channels}, series_length={series_length}, n_trials={n_trials}")
        for row in summary_rows:
            logger.info(row)

        # wandb logging
        try:
            import wandb
            if wandb.run is not None:
                wandb_metrics = {}
                for condition in ["original", "synthetic", "augmented"]:
                    for clf_name in classifier_factories:
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
            "n_trials": n_trials,
            "seeds": seeds,
            "label_column": LABEL_COLUMN,
            "group_column": group_col,
            "results": results,
        }

        # Derive score from the average TSTR gap (error_synthetic - error_original).
        # Classifiers that failed in either condition are excluded.
        # gap=0  → similarity=1.0 (perfect utility preservation)
        # gap=0.5 → similarity=0.0 (synthetic trains a near-random classifier)
        # Negative gaps (synthetic beats original) are treated as gap=0.
        common_clfs = [
            clf for clf in classifier_factories
            if results.get("original", {}).get(clf, {}).get("error") is not None
            and results.get("synthetic", {}).get(clf, {}).get("error") is not None
        ]
        if common_clfs:
            avg_gap = float(np.mean([
                results["synthetic"][clf]["error"] - results["original"][clf]["error"]
                for clf in common_clfs
            ]))
            similarity = max(0.0, min(1.0, 1.0 - max(0.0, avg_gap) * 2))
            tsc_score = EvaluationScore.finalize_grade(raw_score=similarity, score=similarity * 10)
        else:
            tsc_score = EvaluationScore(
                grade=Grade.UNAVAILABLE,
                notes="No classifier pairs succeeded for both original and synthetic conditions.",
            )

        return TSCUtility(score=tsc_score, details=details)
