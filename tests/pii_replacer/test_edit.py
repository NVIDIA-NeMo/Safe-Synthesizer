# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Skip all tests in this module if torch is not available
pytest.importorskip("torch", reason="torch is required for these tests (install with: uv sync --extra cpu)")

import re
from unittest import TestCase

import pandas as pd

from nemo_safe_synthesizer.pii_replacer.data_editor.edit import Editor, ProgressLog

# Entity types that support the bothify transformation (structure-preserving replacement).
BOTHIFY_ENTITIES = [
    "unique_identifier",
    "medical_record_number",
    "health_plan_beneficiary_number",
    "account_number",
    "certificate_license_number",
    "device_identifier",
    "biometric_identifier",
    "bank_routing_number",
    "swift_bic",
    "employee_id",
    "api_key",
    "customer_id",
    "user_name",
    "password",
    "http_cookie",
]

# The value expression from the default config that uses bothify
# Note: In YAML, \\\\d becomes \\d which is the regex pattern for digits.
# In Python, we need to use \\d directly (or '\\\\d' in a regular string).
BOTHIFY_VALUE_EXPRESSION = 'fake.bothify(re.sub("\\\\d", "#", re.sub("[A-Z]", "?", (this | string))))'


def make_bothify_config(seed: int = 42, entity_type: str = "account_number") -> dict:
    """Create a minimal config for testing the bothify transformation."""
    return {
        "globals": {
            "seed": seed,
            "locales": ["en_US"],
        },
        "steps": [
            {
                "rows": {
                    "update": [
                        {
                            "condition": f'column.entity == "{entity_type}" and not (this | isna)',
                            "value": BOTHIFY_VALUE_EXPRESSION,
                        }
                    ]
                }
            }
        ],
    }


class TestBothifyTransformation(TestCase):
    """Test the bothify transformation used for identifier entities.

    The bothify transformation uses Faker's bothify function which:
    - Replaces uppercase letters [A-Z] with random letters (any case via ?)
    - Replaces digits with random digits (via #)
    - Preserves lowercase letters and special characters as-is
    """

    def test_bothify_preserves_structure(self):
        """Test that bothify preserves the structure: uppercase->letter, digit->digit."""
        df = pd.DataFrame(
            {
                "account": ["ABC123", "XYZ789", "DEF456"],
            }
        )
        entities = {"account": "account_number"}
        column_types = {"account": None}

        editor = Editor(make_bothify_config(seed=42, entity_type="account_number"), entity_extractor=None)
        result = editor.process_df(df, entities, column_types)

        # Values should be transformed
        self.assertFalse(result["account"].equals(df["account"]))

        # Each transformed value should have the same structure
        for original, transformed in zip(df["account"], result["account"]):
            # Same length
            self.assertEqual(len(original), len(transformed))
            # Check character types
            for i, (orig_char, trans_char) in enumerate(zip(original, transformed)):
                if orig_char.isupper():
                    # Uppercase letters become random letters (upper or lower)
                    self.assertTrue(
                        trans_char.isalpha(),
                        f"Position {i}: expected letter, got '{trans_char}'",
                    )
                elif orig_char.isdigit():
                    self.assertTrue(
                        trans_char.isdigit(),
                        f"Position {i}: expected digit, got '{trans_char}'",
                    )
                else:
                    # Lowercase and special chars preserved as-is
                    self.assertEqual(orig_char, trans_char)

    def test_bothify_preserves_lowercase_and_special_chars(self):
        """Test that bothify preserves lowercase letters and special characters.

        Note: Faker's bothify has special handling for certain characters:
        - ? -> random letter (any case)
        - # -> random digit
        - @ -> random letter OR digit (unpredictable)

        So we avoid using @ in test data to ensure predictable behavior.
        """
        df = pd.DataFrame(
            {
                # Avoid @ as it's special in bothify
                "identifier": ["abc-123_xyz", "foo.bar-456"],
            }
        )
        entities = {"identifier": "unique_identifier"}
        column_types = {"identifier": None}

        editor = Editor(make_bothify_config(seed=42, entity_type="unique_identifier"), entity_extractor=None)
        result = editor.process_df(df, entities, column_types)

        for original, transformed in zip(df["identifier"], result["identifier"]):
            # Same length
            self.assertEqual(len(original), len(transformed))
            for i, (orig_char, trans_char) in enumerate(zip(original, transformed)):
                # Lowercase letters and special chars (except @) should be preserved exactly
                if orig_char.islower() or not orig_char.isalnum():
                    self.assertEqual(
                        orig_char,
                        trans_char,
                        f"Position {i}: expected '{orig_char}' to be preserved, got '{trans_char}'",
                    )
                elif orig_char.isdigit():
                    # Digits become random digits
                    self.assertTrue(
                        trans_char.isdigit(),
                        f"Position {i}: expected digit, got '{trans_char}'",
                    )

    def test_bothify_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        df = pd.DataFrame(
            {
                "account": ["ABC123", "XYZ789"],
            }
        )
        entities = {"account": "account_number"}
        column_types = {"account": None}

        editor1 = Editor(make_bothify_config(seed=42, entity_type="account_number"), entity_extractor=None)
        result1 = editor1.process_df(df.copy(), entities, column_types)

        editor2 = Editor(make_bothify_config(seed=99, entity_type="account_number"), entity_extractor=None)
        result2 = editor2.process_df(df.copy(), entities, column_types)

        # Different seeds should produce different results
        self.assertFalse(result1["account"].equals(result2["account"]))


@pytest.mark.parametrize("entity_type", BOTHIFY_ENTITIES)
def test_bothify_for_entity_type(entity_type):
    """Test that each entity type in BOTHIFY_ENTITIES is correctly transformed."""
    df = pd.DataFrame(
        {
            "col": ["ABC123-xyz", "XY99-abc", "TEST42-data"],
        }
    )
    entities = {"col": entity_type}
    column_types = {"col": None}

    editor = Editor(make_bothify_config(seed=42, entity_type=entity_type), entity_extractor=None)
    result = editor.process_df(df, entities, column_types)

    # Values should be transformed
    assert not result["col"].equals(df["col"])

    # Structure should be preserved
    for original, transformed in zip(df["col"], result["col"]):
        assert len(original) == len(transformed)
        for orig_char, trans_char in zip(original, transformed):
            if orig_char.isupper():
                # Uppercase letters become random letters (any case)
                assert trans_char.isalpha(), f"Expected letter for '{orig_char}', got '{trans_char}'"
            elif orig_char.isdigit():
                assert trans_char.isdigit(), f"Expected digit for '{orig_char}', got '{trans_char}'"
            else:
                assert orig_char == trans_char, f"Expected '{orig_char}' to be preserved, got '{trans_char}'"


def test_bothify_only_uppercase_value():
    """Test bothify with a value containing only uppercase letters."""
    df = pd.DataFrame({"col": ["ABCDEF"]})
    entities = {"col": "unique_identifier"}
    column_types = {"col": None}

    editor = Editor(make_bothify_config(seed=42, entity_type="unique_identifier"), entity_extractor=None)
    result = editor.process_df(df, entities, column_types)

    transformed = result["col"].iloc[0]
    assert len(transformed) == 6
    # Uppercase letters become random letters (any case)
    assert transformed.isalpha()


def test_bothify_only_digits_value():
    """Test bothify with a value containing only digits."""
    df = pd.DataFrame({"col": ["123456"]})
    entities = {"col": "account_number"}
    column_types = {"col": None}

    editor = Editor(make_bothify_config(seed=42, entity_type="account_number"), entity_extractor=None)
    result = editor.process_df(df, entities, column_types)

    transformed = result["col"].iloc[0]
    assert len(transformed) == 6
    assert transformed.isdigit()


def test_bothify_mixed_case_preserves_lowercase():
    """Test that lowercase letters are preserved in bothify transformation."""
    df = pd.DataFrame({"col": ["ABCdef123"]})
    entities = {"col": "employee_id"}
    column_types = {"col": None}

    editor = Editor(make_bothify_config(seed=42, entity_type="employee_id"), entity_extractor=None)
    result = editor.process_df(df, entities, column_types)

    transformed = result["col"].iloc[0]
    # Lowercase portion should be preserved
    assert transformed[3:6] == "def"
    # Uppercase portion should be letters (any case from bothify)
    assert transformed[:3].isalpha()
    # Digit portion should be digits
    assert transformed[6:].isdigit()


def test_bothify_complex_identifier_format():
    """Test bothify with a complex identifier format like 'XX-1234-ABCD-5678'."""
    df = pd.DataFrame({"col": ["XX-1234-ABCD-5678"]})
    entities = {"col": "device_identifier"}
    column_types = {"col": None}

    editor = Editor(make_bothify_config(seed=42, entity_type="device_identifier"), entity_extractor=None)
    result = editor.process_df(df, entities, column_types)

    transformed = result["col"].iloc[0]
    # Check format is preserved: letters-digits-letters-digits with dashes
    # Note: bothify produces random letters (any case), not just uppercase
    assert re.match(r"^[A-Za-z]{2}-\d{4}-[A-Za-z]{4}-\d{4}$", transformed)


class TestProgressLog(TestCase):
    def test_logs(self):
        progress = ProgressLog(30)

        progress.last_log = progress.last_log - 300
        progress.start_time = progress.last_log
        progress.status.step_n = 2
        progress.status.step_n_total = 4
        progress.status.update_rule_n = 0
        progress.status.update_rule_n_total = 3
        progress.status.row_n = 5000
        progress.status.row_n_total = 10000
        progress.status.update_rule_description = "This does a thing"

        with self.assertLogs() as logs:
            progress.log_throttled()

        # Data is logged via extra={"ctx": {...}}, so check the record's ctx attribute
        ctx = getattr(logs.records[0], "ctx", None)
        assert ctx is not None, "Expected log record to have ctx attribute"
        tabular_data = ctx["tabular_data"]

        assert tabular_data["transform_time"] == "5.0 minutes"
        assert tabular_data["step"] == "3 of 4"
        assert tabular_data["rule"] == '1 of 3 "This does a thing"'
        assert tabular_data["speed"] == "🐇 16.7 rows per second."
        assert tabular_data["progress"] == "5000 rows out of 10000 transformed"

    def test_throttled(self):
        progress = ProgressLog(30)
        progress.last_log = progress.last_log - 300
        progress.start_time = progress.last_log
        with self.assertLogs() as logs:
            progress.log_throttled()
            progress.log_throttled()
            progress.log_throttled()
            progress.log_throttled()
            progress.log_throttled()
        self.assertEqual(len(logs.output), 1)

    def test_forced(self):
        progress = ProgressLog(30)
        progress.last_log = progress.last_log - 300
        progress.start_time = progress.last_log
        with self.assertLogs() as logs:
            progress.log_throttled(force=True)
            progress.log_throttled(force=True)
            progress.log_throttled(force=True)
            progress.log_throttled(force=True)
            progress.log_throttled(force=True)
        self.assertEqual(len(logs.output), 5)
