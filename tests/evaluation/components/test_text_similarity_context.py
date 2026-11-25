# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import warnings

import numpy as np

from nemo_safe_synthesizer.config.evaluation_report_data import (
    DifferentialPrivacySettings,
    EvaluationReportData,
    Metrics,
    Scores,
)
from nemo_safe_synthesizer.evaluation.components.text_semantic_similarity import _suppress_ks_exact_fallback
from nemo_safe_synthesizer.evaluation.render import render_report
from nemo_safe_synthesizer.evaluation.report_context import report_context


def _report_data(**metrics: object) -> EvaluationReportData:
    payload = json.loads(_EMPTY_REPORT.model_dump_json())
    payload["metrics"].update(metrics)
    return EvaluationReportData.model_validate(payload)


_UNAVAILABLE = {"raw_score": None, "score": None, "grade": "Unavailable", "notes": None}
_EMPTY_REPORT = EvaluationReportData(
    schema_version=1,
    library_version="test",
    job_id=None,
    scores=Scores(synthetic_quality=None, data_privacy=None),
    metrics=Metrics(**dict.fromkeys(Metrics.model_fields)),
    dataset=None,
    columns=[],
    any_column_transformed=False,
    differential_privacy=DifferentialPrivacySettings(enabled=False, epsilon=None, delta=None),
    pii_replacement=None,
)


def test_report_without_dataset_counts_renders_without_dataset_statistics() -> None:
    html = render_report(_EMPTY_REPORT)

    assert html is not None
    assert 'id="dataset-statistics"' not in html


def test_text_semantic_similarity_figures_have_a_column_heading() -> None:
    points = {"x": [0.1, 0.3, 0.2], "y": [0.2, 0.1, 0.4]}
    projection = {"axes": ["pc1", "pc2"], "training": points, "synthetic": points}
    data = _report_data(text_semantic_similarity={"score": _UNAVAILABLE, "projections": {"review": projection}})

    figures = report_context(data)["text_semantic_similarity"]["figures"]

    assert [figure["title"] for figure in figures] == ["review"]
    assert "plotly-graph-div" in figures[0]["html"]


def test_text_semantic_similarity_figure_embeds_float32_embeddings_as_float32() -> None:
    values = np.array([0.1, 0.3, 0.2], dtype=np.float32).tolist()
    projection = {
        "axes": ["pc1", "pc2"],
        "training": {"x": values, "y": values},
        "synthetic": {"x": values, "y": values},
    }
    data = _report_data(text_semantic_similarity={"score": _UNAVAILABLE, "projections": {"review": projection}})

    html = report_context(data)["text_semantic_similarity"]["figures"][0]["html"]

    assert '"dtype":"f4"' in html
    assert '"dtype":"f8"' not in html


def test_text_structure_similarity_figures_have_a_column_heading() -> None:
    distribution = {"labels": ["1"], "bin_edges": [0.5, 1.5], "training_percent": [100.0], "synthetic_percent": [100.0]}
    statistics = dict.fromkeys(["sentence_count", "words_per_sentence", "characters_per_word"], distribution)
    data = _report_data(text_structure_similarity={"score": _UNAVAILABLE, "columns": {"review": statistics}})

    figures = report_context(data)["text_structure_similarity"]["figures"]

    assert [figure["title"] for figure in figures] == ["review"]
    assert "plotly-graph-div" in figures[0]["html"]


def test_suppress_ks_exact_fallback_is_scoped_to_the_scipy_notice():
    """Silences SciPy's asymptotic-fallback notice without swallowing other warnings."""
    # Verbatim from scipy/stats/_stats_py.py where ks_2samp abandons the exact method.
    notice = "ks_2samp: Exact calculation unsuccessful. Switching to method=asymp."

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _suppress_ks_exact_fallback():
            warnings.warn(notice, RuntimeWarning)
            warnings.warn("an unrelated problem", RuntimeWarning)
        assert [str(w.message) for w in caught] == ["an unrelated problem"]
