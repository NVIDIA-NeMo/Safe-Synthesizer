# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU evaluation smoke test -- MultimodalReport.from_dataframes().

Exercises the evaluation pipeline (column distribution, correlation,
deep structure, text similarity) on a small dataset with privacy
metrics disabled. Catches dep breakage in the evaluation stack
(scipy, plotly, sentence-transformers, etc.).
"""

import pytest

pytest.importorskip(
    "sentence_transformers", reason="sentence-transformers required (install with: uv sync --extra cpu)"
)

from nemo_safe_synthesizer.config.parameters import EvaluationParameters, SafeSynthesizerParameters
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport


def test_multimodal_report_from_dataframes(iris_df):
    """Build a MultimodalReport from iris_df on CPU with privacy metrics off."""
    config = SafeSynthesizerParameters(
        evaluation=EvaluationParameters(mia_enabled=False, aia_enabled=False),
    )
    training_df = iris_df.copy()
    synthetic_df = iris_df.sample(frac=0.8, random_state=42).reset_index(drop=True)

    report = MultimodalReport.from_dataframes(
        training=training_df,
        synthetic=synthetic_df,
        config=config,
    )
    assert report is not None
    assert len(report.components) > 0

    score_dict = report.get_dict()
    assert "Synthetic Quality Score" in score_dict
