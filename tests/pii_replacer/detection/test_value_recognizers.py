# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity recognition from values: shapes, coverage, and the gates around them."""

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
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
    analysis = analyze_column_patterns(series, Config(), phone_min_digits=7, name_label="phone_number")
    assert analysis["entity"] == "phone_number"
    assert analysis["coverage"] == 100.0
    assert analysis["structured"] is True

    # Replaceable entities are never inferred from values alone: without a
    # supporting header, phone is not even a candidate.
    assert analyze_column_patterns(series, Config(), phone_min_digits=7)["entity"] is None


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
    from nemo_safe_synthesizer.pii_replacer.detection import analyze_column_patterns, collect_value_entities
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    assert collect_value_entities("020 7946 0958 x123", phone_min_digits=7) == ["phone_number"]
    assert collect_value_entities("555-1234", phone_min_digits=7) == ["phone_number"]
    series = pd.Series(["020 7946 0958 x123"] * 20 + ["555-1234"] * 10)
    analysis = analyze_column_patterns(series, Config(), phone_min_digits=7, name_label="phone_number")
    assert analysis["entity"] == "phone_number" and analysis["structured"] is True

    n = 30
    df = pd.DataFrame({"phone": [f"555-{1000 + i:04d}" for i in range(n)]})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    phone = column_spec(plan, "phone")
    assert phone is not None and phone.entity_type == EntityType.phone_number


def test_opaque_hex_is_unique_identifier_not_api_key():
    from nemo_safe_synthesizer.pii_replacer.detection import collect_value_entities
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    hex64 = "a" * 64
    assert collect_value_entities(hex64) == ["unique_identifier"]
    assert collect_value_entities("sk-" + "x" * 24) == ["api_key"]
    n = 30
    df = pd.DataFrame({"subject_key": [f"{i:064x}" for i in range(n)]})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    spec = column_spec(plan, "subject_key")
    assert spec is not None and spec.entity_type == EntityType.unique_identifier


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
    assert column_spec(plan, "provider_name") is None


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
    assert column_spec(plan, "guardians") is None


def test_dotted_phone_not_jwt_unique_identifier():
    from nemo_safe_synthesizer.pii_replacer.detection import analyze_column_patterns, collect_value_entities
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    assert collect_value_entities("818.470.1711") == ["phone_number"]
    # Real JWT-shaped tokens still classify as unique_identifier.
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    assert collect_value_entities(jwt) == ["unique_identifier"]

    series = pd.Series(["818.470.1711"] * 70 + ["415-555-0100"] * 30)
    analysis = analyze_column_patterns(series, Config(), phone_min_digits=7, name_label="phone_number")
    assert analysis["entity"] == "phone_number"
    assert analysis["structured"] is True

    n = 30
    df = pd.DataFrame({"phone_number": [f"818.470.{1000 + i}" for i in range(n)]})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    phone = column_spec(plan, "phone_number")
    assert phone is not None and phone.entity_type == EntityType.phone_number


def test_entity_coverage_counts_each_entity_independently():
    """A value matching two entities counts toward both; no ordering picks a winner."""
    from nemo_safe_synthesizer.pii_replacer.detection import collect_value_entities, entity_coverage

    # Amex test PAN with separators satisfies both the card and phone shapes.
    amex = "3782-822463-10005"
    assert collect_value_entities(amex) == ["credit_debit_card", "phone_number"]

    table = entity_coverage(pd.Series([amex] * 20), phone_min_digits=7)
    assert table.total == 20
    assert table.coverage("credit_debit_card") == 100.0
    assert table.coverage("phone_number") == 100.0
    assert table.coverage("email") == 0.0


def test_header_entity_decides_between_overlapping_shapes():
    """Same values, different headers -> the header's entity is what gets verified."""
    from nemo_safe_synthesizer.pii_replacer.detection import analyze_column_patterns

    series = pd.Series(["3782-822463-10005"] * 20)

    as_card = analyze_column_patterns(series, Config(), name_label="credit_debit_card")
    assert as_card["entity"] == "credit_debit_card"
    assert as_card["structured"] is True

    as_phone = analyze_column_patterns(series, Config(), phone_min_digits=7, name_label="phone_number")
    assert as_phone["entity"] == "phone_number"
    assert as_phone["structured"] is True


def test_shadowed_entity_no_longer_disqualifies_named_column():
    """Regression: a competing recognizer must not dilute the named entity's coverage.

    Dotted phone values also satisfy an opaque three-segment token shape. Under
    per-entity coverage the phone column reads 100% regardless.
    """
    from nemo_safe_synthesizer.pii_replacer.detection import analyze_column_patterns, entity_coverage

    series = pd.Series([f"818.470.{1000 + i}" for i in range(40)])
    table = entity_coverage(series, phone_min_digits=7)
    assert table.coverage("phone_number") == 100.0

    analysis = analyze_column_patterns(series, Config(), phone_min_digits=7, name_label="phone_number")
    assert analysis["coverage"] == 100.0
    assert analysis["structured"] is True


def test_candidate_entities_never_allows_replaceable_from_values_alone():
    """The candidate set is what enforces "no replaceable PII without a header"."""
    from nemo_safe_synthesizer.pii_replacer.detection import candidate_entities

    temporals = ["datetime", "date", "time", "duration"]

    # No header match: only identify-not-replaced temporals are eligible.
    assert candidate_entities(None) == temporals

    # Header match: its own entity first (so it wins coverage ties), then temporals.
    assert candidate_entities("phone_number") == ["phone_number", *temporals]
    assert candidate_entities("email") == ["email", *temporals]

    # Replaceable entities the header does not name are never candidates.
    for label in ("ssn", "credit_debit_card", "unique_identifier", "api_key", "email"):
        assert label not in candidate_entities(None)
        assert label not in candidate_entities("phone_number")

    # Entities with no value recognizer contribute no candidate of their own.
    assert candidate_entities("first_name") == temporals


def test_value_entity_labels_cover_every_recognizer_output():
    """``_VALUE_ENTITY_LABELS`` must list every label the recognizer can emit."""
    from nemo_safe_synthesizer.pii_replacer.detection import collect_value_entities
    from nemo_safe_synthesizer.pii_replacer.detection.value_recognizers import _VALUE_ENTITY_LABELS

    corpus = [
        "jane@acme.com",
        "192.168.1.1",
        "2001:db8::1",
        "a" * 64,
        "123-45-6789",
        "2023-04-17 14:30:00",
        "2023-04-17",
        "14:30:00",
        "PT2H30M",
        "sk-" + "x" * 24,
        "4111-1111-1111-1111",
        "415-555-0100",
    ]
    emitted = {label for value in corpus for label in collect_value_entities(value, phone_min_digits=7)}
    assert emitted, "corpus produced no labels"
    assert emitted <= set(_VALUE_ENTITY_LABELS), emitted - set(_VALUE_ENTITY_LABELS)


def test_card_requires_issuer_prefix_and_brand_length():
    """Luhn alone passes ~1 in 10 numbers; the issuer prefix keeps phones out."""
    from nemo_safe_synthesizer.pii_replacer.detection import card_brand, collect_value_entities

    # Standard test PANs across brands (including 14- and 16-digit Diners).
    for pan in (
        "4111111111111111",
        "4222222222222",
        "4917610000000000003",
        "378282246310005",
        "5555555555554444",
        "2223003122003222",
        "6011111111111117",
        "3530111333300000",
        "6200000000000005",
        "30569309025904",
        "3056930009020004",
    ):
        assert card_brand(pan) is not None, pan
        assert "credit_debit_card" in collect_value_entities(pan), pan

    # Luhn-valid but no issuer prefix owns that length -> not a card.
    assert card_brand("7111111111111111") is None
    assert "credit_debit_card" not in collect_value_entities("7111-1111-1111-1111")


def test_phone_suppressed_for_shapes_a_phone_cannot_take():
    from nemo_safe_synthesizer.pii_replacer.detection import collect_value_entities

    # Dotted quad, strict 3-2-4 SSN, and a parseable date are never phones,
    # even though their digits and separators satisfy the phone shape.
    assert collect_value_entities("255.255.255.0", phone_min_digits=7) == ["ipv4"]
    assert collect_value_entities("192.168.1.1", phone_min_digits=7) == ["ipv4"]
    assert collect_value_entities("123-45-6789", phone_min_digits=7) == ["ssn"]
    assert collect_value_entities("2023-04-17", phone_min_digits=7) == ["date"]

    # Genuine phone formats are untouched.
    for phone in ("415-555-0100", "(415) 555-0100", "818.470.1711", "+49 30 901820"):
        assert collect_value_entities(phone, phone_min_digits=7) == ["phone_number"], phone


def test_failed_luhn_card_does_not_become_phone():
    from nemo_safe_synthesizer.pii_replacer.detection import collect_value_entities

    # 16 digits, fails Luhn, no phone punctuation → not phone
    assert collect_value_entities("4111111111111112") == []
    assert collect_value_entities("4111-1111-1111-1111") == ["credit_debit_card"]  # valid Luhn Visa test


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
    assert column_spec(plan, "row_id") is None
    assert column_spec(plan, "member_id") is None
    gapped = column_spec(plan, "gapped_id")
    assert gapped is not None and gapped.entity_type == EntityType.unique_identifier


def test_numeric_ssn_and_national_id_keep_header_entity():
    """Numeric probe must preserve ssn/national_id; sequential skip is unique_identifier-only."""
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            # Contiguous ints under an ssn header must still be planned as ssn.
            "ssn": [100000000 + i for i in range(n)],
            # Contiguous ints under national_id must still be planned as national_id.
            "national_id": [200000000 + i for i in range(n)],
            # Gapped numeric unique_identifier still planned (unchanged).
            "member_id": [100000 + i * 17 for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    ssn = column_spec(plan, "ssn")
    nat = column_spec(plan, "national_id")
    mid = column_spec(plan, "member_id")
    assert ssn is not None and ssn.entity_type == EntityType.ssn
    assert nat is not None and nat.entity_type == EntityType.national_id
    assert mid is not None and mid.entity_type == EntityType.unique_identifier


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
    street = column_spec(plan, "street")
    # Street-name-only values are not street_address; they may still be free_text
    # once structured PII exists to propagate into.
    assert street is None or street.entity_type is not EntityType.street_address


def test_numeric_token_column_not_api_key():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 40
    df = pd.DataFrame({"token": list(range(1000, 1000 + n))})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    assert column_spec(plan, "token") is None
