# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Skip all tests in this module if torch is not available
pytest.importorskip("torch", reason="torch is required for these tests (install with: uv sync --extra cpu)")

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from nemo_safe_synthesizer.pii_replacer.nemo_pii import NemoPII


@patch("nemo_safe_synthesizer.pii_replacer.nemo_pii.build_entity_extractor", return_value=MagicMock())
def test_nemo_pii_classify_df(_build_entity_extractor):
    df = pd.read_csv(Path(__file__).parent / "fake_people_dataset.csv")

    mock_column_classifier = MagicMock()
    mock_column_classifier.detect_types.return_value = {
        "fname": "first_name",
        "lname": "last_name",
        "email": "email",
        "full address": "address",
        "height": "none",
        "date of birth": "none",
    }

    with patch(
        "nemo_safe_synthesizer.pii_replacer.nemo_pii.get_column_classifier", return_value=mock_column_classifier
    ):
        nemo_pii = NemoPII()
        result = nemo_pii.classify_df(df)
        assert result["columns"] == {
            "fname": "first_name",
            "lname": "last_name",
            "email": "email",
            "full address": "address",
            "height": "none",
            "date of birth": "none",
            "notes": "text",
        }

        assert result["entities"] == {
            "fname": "first_name",
            "lname": "last_name",
            "email": "email",
            "full address": "address",
            "height": None,
            "date of birth": None,
            # notes is not here since ColumnClassifierLLM drops columns with too
            # many characters.
        }


@patch("nemo_safe_synthesizer.pii_replacer.nemo_pii.build_entity_extractor", return_value=MagicMock())
def test_nemo_pii_classify_df_no_column_classifier(_build_entity_extractor):
    df = pd.read_csv(Path(__file__).parent / "fake_people_dataset.csv")

    mock_column_classifier = MagicMock()
    mock_column_classifier.detect_types.side_effect = Exception("Classification failed")

    with patch(
        "nemo_safe_synthesizer.pii_replacer.nemo_pii.get_column_classifier", return_value=mock_column_classifier
    ):
        nemo_pii = NemoPII()
        result = nemo_pii.classify_df(df)
        assert result["columns"] == {
            "fname": None,
            "lname": None,
            "email": None,
            "full address": "text",
            "height": None,
            "date of birth": None,
            "notes": "text",
        }

        assert result["entities"] == {}


@patch("nemo_safe_synthesizer.pii_replacer.nemo_pii.build_entity_extractor", return_value=MagicMock())
def test_nemo_pii_default_config_national_id(_build_entity_extractor):
    # Added to confirm a typo fix works in the condition for national_id
    # and tax_id entities in the default config.
    df = pd.DataFrame(
        {
            "national_id": ["123456789", "1234567890", "12345678901"],
            "tax_id": ["123456789", "1234567890", "12345678901"],
        }
    )

    classifications = {
        "entities": {
            "national_id": "national_id",
            "tax_id": "tax_id",
        },
        "columns": {
            "national_id": "national_id",
            "tax_id": "tax_id",
        },
    }
    nemo_pii = NemoPII()

    nemo_pii.transform_df(df, classifications)
    result = nemo_pii.result

    assert not result.transformed_df["national_id"].equals(df["national_id"])
    assert not result.transformed_df["tax_id"].equals(df["tax_id"])
