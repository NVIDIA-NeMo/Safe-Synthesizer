# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU PII replacement smoke test -- NemoPII classify + transform.

Exercises the PII replacement pipeline with mocked external dependencies
(GLiNER model, column classifier). Catches dep breakage in the PII stack
without requiring network access or GPU.
"""

import pytest

pytest.importorskip("torch", reason="torch required (install with: uv sync --extra cpu)")

from unittest.mock import MagicMock, patch

import pandas as pd

from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig
from nemo_safe_synthesizer.pii_replacer.nemo_pii import NemoPII


@pytest.fixture
def pii_test_df():
    """Small DataFrame with PII-like columns for testing the replacement pipeline."""
    return pd.DataFrame(
        {
            "name": ["Alice Johnson", "Bob Smith", "Carol White", "Dave Brown"],
            "email": ["alice@example.com", "bob@test.org", "carol@mail.com", "dave@demo.net"],
            "score": [85, 92, 78, 95],
        }
    )


@patch("nemo_safe_synthesizer.pii_replacer.nemo_pii.build_entity_extractor", return_value=MagicMock())
def test_pii_classify_with_local_detection(_mock_extractor, pii_test_df):
    """Classify columns with enable_classify=False (local field detection only, no LLM)."""
    config = PiiReplacerConfig.get_default_config()
    config.globals.classify.enable_classify = False

    nemo_pii = NemoPII(config=config)
    classification = nemo_pii.classify_df(pii_test_df)

    assert classification is not None
    assert len(classification) == len(pii_test_df.columns)
    classified_names = {field.field_name for field in classification}
    assert classified_names == {"name", "email", "score"}


@patch("nemo_safe_synthesizer.pii_replacer.nemo_pii.build_entity_extractor", return_value=MagicMock())
def test_pii_transform_with_mocked_classifier(_mock_extractor, pii_test_df):
    """Run classify + transform with a mocked column classifier that detects name and email."""
    mock_classifier = MagicMock()
    mock_classifier.detect_types.return_value = {
        "name": "first_name",
        "email": "email",
        "score": None,
    }

    with patch(
        "nemo_safe_synthesizer.pii_replacer.nemo_pii.get_column_classifier",
        return_value=mock_classifier,
    ):
        nemo_pii = NemoPII()
        nemo_pii.transform_df(pii_test_df)

    result = nemo_pii.result
    assert result is not None
    assert result.transformed_df is not None
    assert len(result.transformed_df) == len(pii_test_df)
    assert result.transformed_df["score"].equals(pii_test_df["score"])
    assert result.column_statistics is not None
