# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import pandas as pd
import plotly.graph_objects as go
import pytest

from nemo_safe_synthesizer.evaluation.components import multi_modal_figures
from nemo_safe_synthesizer.evaluation.components.text_semantic_similarity import (
    TextSemanticSimilarity,
    TextSemanticSimilarityDatum,
    _suppress_ks_exact_fallback,
)
from nemo_safe_synthesizer.evaluation.components.text_structure_similarity import (
    TextDataSetStatistics,
    TextStructureSimilarity,
)


def _figure() -> go.Figure:
    return go.Figure(data=[go.Bar(x=[1], y=[2])])


def test_text_semantic_similarity_jinja_context_includes_column_heading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(multi_modal_figures, "generate_text_semantic_similarity_figures", lambda *_: _figure())
    pca = pd.DataFrame({"pc1": [0.1], "pc2": [0.2]})
    component = TextSemanticSimilarity(
        text_semantic_similarity_dict={
            "review": TextSemanticSimilarityDatum(training_pca=pca, synthetic_pca=pca),
        }
    )

    context = component.jinja_context

    assert context["figures"][0]["title"] == "review"
    assert "plotly-graph-div" in context["figures"][0]["html"]


def test_text_structure_similarity_jinja_context_includes_column_heading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(multi_modal_figures, "generate_text_structure_similarity_figures", lambda *_: _figure())
    statistics = TextDataSetStatistics()
    component = TextStructureSimilarity(
        training_statistics={"review": statistics},
        synthetic_statistics={"review": statistics},
    )

    context = component.jinja_context

    assert context["figures"][0]["title"] == "review"
    assert "plotly-graph-div" in context["figures"][0]["html"]


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
