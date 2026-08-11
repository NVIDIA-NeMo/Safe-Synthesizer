# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity recognition from values: shapes, coverage, and the gates around them."""

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.config.pii_replacement import (
    PiiEntity,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.entities import Config, config_from_replace_pii
from tests.pii_replacer.helpers import column_spec


def test_analyze_column_patterns_date_dominant():
    from nemo_safe_synthesizer.pii_replacer.detection import analyze_column_patterns

    series = pd.Series(["04/17/2023"] * 95 + ["08/2010"] * 4 + ["unknown"] * 1)
    analysis = analyze_column_patterns(series, Config())
    assert analysis["entity"] == "date"
    assert analysis["pattern"] == "%m/%d/%Y"
    # Coverage sums all date formats (95% + 4%), not a single pattern bucket.
    assert analysis["coverage"] == 99.0
    assert analysis["structured"] is True


def test_analyze_column_patterns_below_threshold_not_structured():
    from nemo_safe_synthesizer.pii_replacer.detection import analyze_column_patterns

    # Mixed entities: neither reaches the 85% structured gate.
    series = pd.Series(["04/17/2023"] * 70 + ["alice@example.com"] * 30)
    analysis = analyze_column_patterns(series, Config())
    assert analysis["coverage"] == 70.0
    assert analysis["structured"] is False


def test_analyze_column_patterns_aggregates_mixed_phone_formats():
    from nemo_safe_synthesizer.pii_replacer.detection import analyze_column_patterns

    series = pd.Series(["+1-415-555-0100"] * 60 + ["(415) 555-0101"] * 40)
    analysis = analyze_column_patterns(series, Config(), phone_min_digits=7)
    assert analysis["entity"] == "phone_number"
    assert analysis["coverage"] == 100.0
    assert analysis["structured"] is True


def test_analyze_column_patterns_datetime_dominant():
    from nemo_safe_synthesizer.pii_replacer.detection import analyze_column_patterns

    series = pd.Series(["2023-04-17 14:30:00"] * 95 + ["2023-05-01 09:00:00"] * 5)
    analysis = analyze_column_patterns(series, Config())
    assert analysis["entity"] == "datetime"
    assert analysis["pattern"] == "%Y-%m-%d %H:%M:%S"
    assert analysis["coverage"] == 100.0
    assert analysis["structured"] is True


def test_analyze_column_patterns_time_dominant():
    from nemo_safe_synthesizer.pii_replacer.detection import analyze_column_patterns

    series = pd.Series(["14:30:00"] * 90 + ["09:15:00"] * 10)
    analysis = analyze_column_patterns(series, Config())
    assert analysis["entity"] == "time"
    assert analysis["pattern"] == "%H:%M:%S"
    assert analysis["coverage"] == 100.0
    assert analysis["structured"] is True


def test_analyze_column_patterns_duration_dominant():
    from nemo_safe_synthesizer.pii_replacer.detection import analyze_column_patterns

    series = pd.Series(["PT2H30M"] * 85 + ["45 min"] * 15)
    analysis = analyze_column_patterns(series, Config())
    assert analysis["entity"] == "duration"
    assert analysis["pattern"] == "iso8601"
    # iso8601 + human duration formats aggregate to full entity coverage.
    assert analysis["coverage"] == 100.0
    assert analysis["structured"] is True


def test_phone_with_extension_and_short_national_detected():
    from nemo_safe_synthesizer.pii_replacer.detection import analyze_column_patterns, match_value_entity
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    assert match_value_entity("020 7946 0958 x123", phone_min_digits=7) == "phone_number"
    assert match_value_entity("555-1234", phone_min_digits=7) == "phone_number"
    series = pd.Series(["020 7946 0958 x123"] * 20 + ["555-1234"] * 10)
    analysis = analyze_column_patterns(series, Config(), phone_min_digits=7)
    assert analysis["entity"] == "phone_number" and analysis["structured"] is True

    n = 30
    df = pd.DataFrame({"phone": [f"555-{1000 + i:04d}" for i in range(n)]})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    phone = column_spec(plan.standalone_columns_to_replace, "phone")
    assert phone is not None and phone.entity_type == PiiEntity.phone_number


def test_opaque_hex_is_unique_identifier_not_api_key():
    from nemo_safe_synthesizer.pii_replacer.detection import match_value_entity
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    hex64 = "a" * 64
    assert match_value_entity(hex64) == "unique_identifier"
    assert match_value_entity("sk-" + "x" * 24) == "api_key"
    n = 30
    df = pd.DataFrame({"subject_key": [f"{i:064x}" for i in range(n)]})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    spec = column_spec(plan.standalone_columns_to_replace, "subject_key")
    assert spec is not None and spec.entity_type == PiiEntity.unique_identifier


def test_org_name_column_skipped_mary_health_kept(caplog):
    import logging

    from nemo_safe_synthesizer.pii_replacer.detection import looks_like_person_name
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    assert looks_like_person_name("Mary Health")
    assert not looks_like_person_name("Regional Health Partners")
    caplog.set_level(logging.WARNING)
    n = 30
    df = pd.DataFrame({"provider_name": ["Regional Health Partners"] * n})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    assert not any(
        spec.column_name == "provider_name"
        for persona in plan.persona_backed_columns
        for spec in persona.columns_to_replace
    )


def test_temporal_values_identified_without_temporal_header(caplog):
    """Identify-not-replaced temporals still come from value shape alone."""
    import logging

    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    caplog.set_level(logging.INFO)
    n = 40
    df = pd.DataFrame(
        {
            "first_name": [f"First{i}" for i in range(n)],
            "misc_col": [f"2023-04-{(i % 28) + 1:02d}" for i in range(n)],
        }
    )
    discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    assert any("Identified temporal column 'misc_col'" in r.getMessage() for r in caplog.records)


def test_multi_person_cell_not_auto_assigned(caplog):
    import logging

    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    caplog.set_level(logging.WARNING)
    n = 30
    df = pd.DataFrame({"guardians": [f"Jane Doe and John Doe {i}" for i in range(n)]})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    assert not plan.persona_backed_columns or not any(
        spec.column_name == "guardians" for p in plan.persona_backed_columns for spec in p.columns_to_replace
    )


def test_failed_luhn_card_does_not_become_phone():
    from nemo_safe_synthesizer.pii_replacer.detection import match_value_entity

    # 16 digits, fails Luhn, no phone punctuation → not phone
    assert match_value_entity("4111111111111112") is None
    assert match_value_entity("4111-1111-1111-1111") == "credit_debit_card"  # valid Luhn Visa test


def test_sparse_email_column_unique_ratio_not_null_diluted():
    from nemo_safe_synthesizer.pii_replacer.detection import column_stats

    df = pd.DataFrame({"emailish": [None] * 90 + [f"user{i}@example.com" for i in range(10)]})
    stats = column_stats(df)["emailish"]
    # Denominator is non-null rows (10), not full length (100), so nulls do not dilute.
    assert stats["unique_ratio"] == 1.0
    assert stats["n_unique"] == 10


def test_sequential_integer_id_skipped_any_origin():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "row_id": list(range(1, n + 1)),
            "member_id": [100000 + i for i in range(n)],
            "gapped_id": [100000 + i * 17 for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    assert column_spec(plan.standalone_columns_to_replace, "row_id") is None
    assert column_spec(plan.standalone_columns_to_replace, "member_id") is None
    gapped = column_spec(plan.standalone_columns_to_replace, "gapped_id")
    assert gapped is not None and gapped.entity_type == PiiEntity.unique_identifier


def test_street_name_only_not_planned_as_street_address():
    from nemo_safe_synthesizer.pii_replacer.detection import looks_like_street_address
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    assert looks_like_street_address("123 Main St")
    assert not looks_like_street_address("Maple Avenue")
    n = 30
    df = pd.DataFrame(
        {
            "street": [f"Maple Avenue {chr(65 + (i % 26))}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    for persona in plan.persona_backed_columns:
        assert column_spec(persona.columns_to_replace, "street") is None


def test_numeric_token_column_not_api_key():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 40
    df = pd.DataFrame({"token": list(range(1000, 1000 + n))})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    assert column_spec(plan.standalone_columns_to_replace, "token") is None
