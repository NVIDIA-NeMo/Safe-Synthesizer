#!/usr/bin/env -S uv run --frozen
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the example figure for the Autocorrelation Similarity user guide.

Scores and profiles come from the metric itself, so rerun this script whenever
the calculation changes:

    uv run --frozen tools/gen_autocorrelation_doc_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nemo_safe_synthesizer.config.evaluate import EvaluationParameters, TimeSeriesEvaluationParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "user-guide"
    / "time-series-metrics"
    / "assets"
    / "autocorrelation-examples.svg"
)
POINTS = 240
PLOTTED_POINTS = 96
TRAINING_COLOR = "#3B82F6"
SYNTHETIC_COLOR = "#F59E0B"


def _seasonal_series(seed: int, period: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(POINTS)
    for t in range(1, POINTS):
        noise[t] = 0.6 * noise[t - 1] + rng.normal()
    return 3.0 * np.sin(2 * np.pi * np.arange(POINTS) / period) + noise


def _score(training: np.ndarray, synthetic: np.ndarray, config: SafeSynthesizerParameters) -> AutocorrelationSimilarity:
    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"time": np.arange(POINTS), "value": values})

    datasets = EvaluationDatasets(training=frame(training), synthetic=frame(synthetic))
    return AutocorrelationSimilarity.from_evaluation_datasets(datasets, config)


def main() -> None:
    config = SafeSynthesizerParameters(
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="time"),
        evaluation=EvaluationParameters(time_series=TimeSeriesEvaluationParameters(enabled=True)),
    )
    training = _seasonal_series(seed=1, period=24)
    examples = [
        ("Close match", _seasonal_series(seed=2, period=24)),
        ("Temporal order lost", np.random.default_rng(3).permutation(training)),
        ("Cycle too short", _seasonal_series(seed=2, period=12)),
    ]

    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "svg.hashsalt": "autocorrelation-examples"})
    figure, axes = plt.subplots(2, len(examples), figsize=(10, 5), sharey="row", layout="constrained")
    for column, (title, synthetic) in enumerate(examples):
        component = _score(training, synthetic, config)
        profile = component.details["profiles"][0]
        series_axis, profile_axis = axes[0, column], axes[1, column]

        series_axis.plot(training[:PLOTTED_POINTS], color=TRAINING_COLOR, linewidth=1.2, label="Training")
        series_axis.plot(synthetic[:PLOTTED_POINTS], color=SYNTHETIC_COLOR, linewidth=1.2, label="Synthetic")
        series_axis.set_title(f"{title}: score {component.score.score:.1f}", fontsize=10)
        series_axis.set_xlabel("Time step")

        lags = profile["lags"]
        training_acf = np.array(profile["training_acf"], dtype=float)
        synthetic_acf = np.array(profile["synthetic_acf"], dtype=float)
        profile_axis.fill_between(lags, training_acf, synthetic_acf, color="#9CA3AF", alpha=0.35, linewidth=0)
        profile_axis.plot(lags, training_acf, color=TRAINING_COLOR, marker="o", markersize=3, linewidth=1.2)
        profile_axis.plot(lags, synthetic_acf, color=SYNTHETIC_COLOR, marker="o", markersize=3, linewidth=1.2)
        profile_axis.axhline(0, color="#6B7280", linewidth=0.6)
        profile_axis.set_ylim(-1.05, 1.05)
        profile_axis.set_xlabel("Lag")

    axes[0, 0].set_ylabel("Value")
    axes[1, 0].set_ylabel("Autocorrelation")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="svg", facecolor="white", metadata={"Date": None})
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
