# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Skip all tests in this module if torch is not available
pytest.importorskip("torch", reason="torch is required for these tests (install with: uv sync --extra cpu)")

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from nemo_safe_synthesizer.pii_replacer.data_editor.edit import TransformFnAccounting
from nemo_safe_synthesizer.pii_replacer.nemo_pii import ColumnClassification, NemoPII, _build_column_statistics


@patch("nemo_safe_synthesizer.pii_replacer.nemo_pii.build_entity_extractor", return_value=MagicMock())
def test_nemo_pii_classify_df(_build_entity_extractor):
    df = pd.read_csv(Path(__file__).parent / "fake_people_dataset.csv")

    mock_column_classifier = MagicMock()
    mock_column_classifier.detect_types.return_value = {
        "fname": "first_name",
        "lname": "last_name",
        "email": "email",
        "full address": "address",
        "height": None,
        "date of birth": "date",
        "notes": None,
    }

    with patch(
        "nemo_safe_synthesizer.pii_replacer.nemo_pii.get_column_classifier", return_value=mock_column_classifier
    ):
        nemo_pii = NemoPII()
        classification = nemo_pii.classify_df(df)
        assert {field.field_name: field.entity for field in classification} == {
            "fname": "first_name",
            "lname": "last_name",
            "email": "email",
            "full address": "address",
            "height": None,
            "date of birth": "date",
            "notes": None,
        }
        assert {field.field_name: field.entity_count for field in classification} == {
            "fname": 20,
            "lname": 20,
            "email": 20,
            "full address": 20,
            "height": None,
            "date of birth": 20,
            "notes": None,
        }

        assert {field.field_name: field.column_type for field in classification} == {
            "fname": "first_name",
            "lname": "last_name",
            "email": "email",
            "full address": "address",
            "height": None,
            "date of birth": "date",
            "notes": "text",
        }


@patch("nemo_safe_synthesizer.pii_replacer.nemo_pii.build_entity_extractor", return_value=MagicMock())
def test_classify_df_no_column_classifier(_build_entity_extractor):
    df = pd.read_csv(Path(__file__).parent / "fake_people_dataset.csv")

    mock_column_classifier = MagicMock()
    mock_column_classifier.detect_types.side_effect = Exception("Classification failed")

    with patch(
        "nemo_safe_synthesizer.pii_replacer.nemo_pii.get_column_classifier", return_value=mock_column_classifier
    ):
        nemo_pii = NemoPII()
        classification = nemo_pii.classify_df(df)
        # Extract column types and entities from column classification results.
        assert {field.field_name: field.column_type for field in classification} == {
            "fname": None,
            "lname": None,
            "email": None,
            "full address": "text",
            "height": None,
            "date of birth": None,
            "notes": "text",
        }

        assert {field.entity for field in classification} == {None}


@patch("nemo_safe_synthesizer.pii_replacer.nemo_pii.build_entity_extractor", return_value=MagicMock())
def test_nemo_pii_classify_disabled(_build_entity_extractor):
    """Test that when enable_classify is False, no external API calls are made."""
    df = pd.read_csv(Path(__file__).parent / "fake_people_dataset.csv")

    # Create a config with classify disabled
    from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig

    config = PiiReplacerConfig.get_default_config()
    config.globals.classify.enable_classify = False

    mock_column_classifier = MagicMock()
    mock_column_classifier.detect_types.return_value = {
        "fname": "first_name",
        "lname": "last_name",
    }

    with patch(
        "nemo_safe_synthesizer.pii_replacer.nemo_pii.get_column_classifier", return_value=mock_column_classifier
    ) as mock_get_classifier:
        nemo_pii = NemoPII(config=config)
        classification = nemo_pii.classify_df(df)
        # Verify the classifier was never instantiated (no API call made)
        mock_get_classifier.assert_not_called()

        # Verify we still get a valid result with text columns detected via local field detection
        assert {field.field_name: field.column_type for field in classification} == {
            "fname": None,
            "lname": None,
            "email": None,
            "full address": "text",
            "height": None,
            "date of birth": None,
            "notes": "text",
        }
        # verify entity and it's related values are none for all columns
        assert {field.entity for field in classification} == {None}
        assert {field.entity_count for field in classification} == {None}
        assert [field.entity_values for field in classification] == [[] for _ in classification]


@patch("nemo_safe_synthesizer.pii_replacer.nemo_pii.build_entity_extractor", return_value=MagicMock())
def test_nemo_pii_default_config_national_id(_build_entity_extractor):
    # Added to confirm a typo fix works in the condition for national_id
    # and tax_id entities in the default config.
    df = pd.DataFrame(
        {
            "national_id": ["123456789", "1234567890", "12345678901"],
            "tax_id": ["123456789", "1234567890", "12345678901"],
            "height": [189.8, 154.4, 157.4],
        }
    )
    mock_column_classifier = MagicMock()
    mock_column_classifier.detect_types.return_value = {
        "national_id": "national_id",
        "tax_id": "tax_id",
        "height": None,
    }

    with patch(
        "nemo_safe_synthesizer.pii_replacer.nemo_pii.get_column_classifier", return_value=mock_column_classifier
    ):
        nemo_pii = NemoPII()
        classifications = nemo_pii.classify_df(df)
        nemo_pii.transform_df(df, classifications)
        result = nemo_pii.result
        # check that the transformed dataframe is not the same as the original dataframe for ID fields
        assert not result.transformed_df["national_id"].equals(df["national_id"])
        assert not result.transformed_df["tax_id"].equals(df["tax_id"])
        assert result.transformed_df["height"].equals(df["height"])

        assert result.column_statistics["national_id"].is_transformed
        assert result.column_statistics["tax_id"].is_transformed
        assert not result.column_statistics["height"].is_transformed


def test_build_column_statistics():
    """Test _build_column_statistics with mocked classification and transformer."""

    # Create mock classifications for a small dataset/easier to test.
    mock_classifications = [
        ColumnClassification(
            field_name="fname",
            column_type="first_name",
            entity="first_name",
            entity_count=5,
            entity_values=["John", "Jane", "Bob"],
        ),
        ColumnClassification(
            field_name="height",
            column_type=None,
            entity=None,
            entity_count=5,
            entity_values=[170, 180, 165, 155],
        ),
        ColumnClassification(
            field_name="notes",
            column_type="text",
            entity=None,  # Text fields have entity as None
            entity_count=5,
            entity_values=[
                "Angela frequently updates her contact with angelas@gmail.com and 1431 newhampshire avenue nw",
                "she updates her residential information to ensure accurate documentation.",
                " While no major conditions have been reported, Angela maintains a detailed health record.You can reach Angela via email at angelas@gmail.com",
                "Ryan Doe currently resides at 8654 Derrick Radial Suite 149, South Andrew, DC 43785.",
                "Ryan Doe recently updated his email address to ryan.doe@example.com.",
            ],
        ),
    ]

    # Create mock TransformFnAccounting
    mock_transform_fn_accounting = MagicMock(spec=TransformFnAccounting)
    mock_transform_fn_accounting.column_fns = {
        "fname": {"fake"},
        "notes": {"redact_entities"},
    }

    # mock NerReport (column_report)
    mock_column_report = {
        "notes": {
            "email": MagicMock(count=3, values={"angelas@gmail.com", "ryan.doe@example.com"}),
            "name": MagicMock(count=5, values={"Angela", "Ryan Doe"}),
            "address": MagicMock(
                count=2, values={"1431 newhampshire avenue nw", "8654 Derrick Radial Suite 149, South Andrew, DC 43785"}
            ),
        }
    }

    column_statistics = _build_column_statistics(
        mock_classifications,
        mock_transform_fn_accounting,
        mock_column_report,
    )
    # Test for correct column names in column statistics
    assert set(column_statistics.keys()) == {"notes", "fname", "height"}

    # Test for correct assigned type for each column
    assert column_statistics["fname"].assigned_type == "first_name"
    assert column_statistics["height"].assigned_type is None
    assert column_statistics["notes"].assigned_type == "text"

    # Test for correct assigned entity for each column
    assert column_statistics["fname"].assigned_entity == "first_name"
    assert column_statistics["height"].assigned_entity is None
    assert column_statistics["notes"].assigned_entity is None  # Text fields have entity as None

    # Test for detected entity counts for each column
    assert column_statistics["fname"].detected_entity_counts == {"first_name": 5}
    assert column_statistics["height"].detected_entity_counts == {}
    assert column_statistics["notes"].detected_entity_counts == {"email": 3, "name": 5, "address": 2}

    # Test for detected entity values
    assert column_statistics["fname"].detected_entity_values == {"first_name": {"John", "Jane", "Bob"}}
    assert column_statistics["height"].detected_entity_values == {}
    assert column_statistics["notes"].detected_entity_values == {
        "email": {"angelas@gmail.com", "ryan.doe@example.com"},
        "name": {"Angela", "Ryan Doe"},
        "address": {"1431 newhampshire avenue nw", "8654 Derrick Radial Suite 149, South Andrew, DC 43785"},
    }

    # Test if is_transformed is applied for entities.
    assert column_statistics["fname"].is_transformed is True
    assert column_statistics["height"].is_transformed is False
    assert column_statistics["notes"].is_transformed is True

    # Test for transform functions
    assert column_statistics["fname"].transform_functions == {"fake"}
    assert column_statistics["height"].transform_functions == set()
    assert column_statistics["notes"].transform_functions == {"redact_entities"}
