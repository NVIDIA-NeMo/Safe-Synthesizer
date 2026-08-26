# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Header reading: normalization, aliases, and what a name alone may claim."""

from __future__ import annotations

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.entities import config_from_replace_pii
from tests.pii_replacer.helpers import column_spec


def test_dob_and_national_id_alias_headers_discovered():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "born_on": [f"1980-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)],
            "aadhaar": [f"AAAA{i:08d}Z" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    dob = column_spec(plan, "born_on")
    assert dob is not None and dob.entity_type == EntityType.date_of_birth
    # national_id is entity-driven; lands in columns_to_replace
    nat = column_spec(plan, "aadhaar")
    assert nat is not None and nat.entity_type in {EntityType.national_id, EntityType.unique_identifier}


def test_numeric_compact_dob_and_id_probed():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "birth_ymd": [19800101 + i for i in range(n)],
            # Non-contiguous numeric ids (gaps) still probe as unique_identifier.
            "member_id": [100000 + i * 17 for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    dob = column_spec(plan, "birth_ymd")
    mid = column_spec(plan, "member_id")
    assert dob is not None and dob.entity_type == EntityType.date_of_birth
    assert mid is not None and mid.entity_type == EntityType.unique_identifier


def test_ssn_shaped_order_code_not_detected_as_ssn():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame({"order_code": [f"{100 + i:03d}-45-{6000 + i:04d}" for i in range(n)]})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    assert column_spec(plan, "order_code") is None


def test_value_only_entities_not_assigned_without_name_match():
    """Emails, phones, UUIDs, and IPs in oddly named columns stay unplanned."""
    import uuid

    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "contact": [f"user{i}@example.com" for i in range(n)],
            "misc_code": [f"+1-415-555-{1000 + i:04d}" for i in range(n)],
            "blob": [str(uuid.uuid4()) for _ in range(n)],
            "endpoint": [f"10.0.{i // 256}.{i % 256}" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    planned = {spec.column_name for spec in plan.columns_to_replace}
    assert planned.isdisjoint({"contact", "misc_code", "blob", "endpoint"})


def test_email_header_requires_email_shaped_values(caplog):
    """A simple content check is required when one exists."""
    import logging

    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    caplog.set_level(logging.WARNING)
    n = 30
    df = pd.DataFrame({"email": [f"not-an-email-{i}" for i in range(n)]})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    assert column_spec(plan, "email") is None
    assert any("looks like email by name" in r.getMessage() for r in caplog.records)


def test_compound_street_headers_planned_as_street_address():
    """CamelCase / compound headers with street-like values classify as street_address."""
    from nemo_safe_synthesizer.pii_replacer.detection.column_names import normalize_column_name_for_match
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    assert normalize_column_name_for_match("MailingStreet") == "mailing street"
    assert normalize_column_name_for_match("AddressLine1") == "address line 1"
    # Underscores stay so applicant_id does not fuzzy-match as a person name column.
    assert normalize_column_name_for_match("applicant_id") == "applicant_id"
    assert normalize_column_name_for_match("dependent_id") == "dependent_id"

    n = 40
    df = pd.DataFrame(
        {
            "MailingStreet": [f"{100 + i} Lantern Avenue" for i in range(n)],
            "AddressLine1": [f"{200 + i} Orchard Road" for i in range(n)],
            "HomeStreet": [f"{300 + i} Willow Drive" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
            "City": [f"City{i % 10}" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    street_cols = {
        spec.column_name
        for spec in plan.columns_to_replace
        if spec.entity_type == EntityType.street_address
    }
    assert {"MailingStreet", "AddressLine1", "HomeStreet"} <= street_cols


def test_normalize_column_name_preserves_underscores():
    from nemo_safe_synthesizer.pii_replacer.detection.column_names import match_label, normalize_column_name_for_match
    from nemo_safe_synthesizer.pii_replacer.entities import ENTITY_NAME_PATTERNS

    assert normalize_column_name_for_match("applicant_id") == "applicant_id"
    assert normalize_column_name_for_match("ApplicantID") == "applicant id"
    # Underscore-separated IDs must not be treated as person-name columns.
    assert match_label("applicant_id", ENTITY_NAME_PATTERNS) != "full_name"
    assert match_label("dependent_id", ENTITY_NAME_PATTERNS) != "full_name"


def test_ipaddress_header_is_not_street_address():
    """Lowercase ``ipaddress`` must not match street_address ahead of ipv4."""
    from nemo_safe_synthesizer.pii_replacer.detection.column_names import match_label
    from nemo_safe_synthesizer.pii_replacer.entities import ENTITY_NAME_PATTERNS
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    assert match_label("ipaddress", ENTITY_NAME_PATTERNS) == "ipv4"
    assert match_label("ip_address", ENTITY_NAME_PATTERNS) == "ipv4"
    assert match_label("IPAddress", ENTITY_NAME_PATTERNS) == "ipv4"
    assert match_label("address", ENTITY_NAME_PATTERNS) == "street_address"

    n = 30
    df = pd.DataFrame(
        {
            "ipaddress": [f"10.0.0.{i % 250}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    ip = column_spec(plan, "ipaddress")
    assert ip is not None and ip.entity_type == EntityType.ipv4


def test_entity_name_patterns_have_fuzzy_keywords():
    from nemo_safe_synthesizer.pii_replacer.entities import ENTITY_NAME_PATTERNS, FUZZY_KEYWORDS

    missing = sorted(set(ENTITY_NAME_PATTERNS) - set(FUZZY_KEYWORDS))
    assert missing == [], f"entity types missing fuzzy keywords: {missing}"


def test_match_labels_returns_all_regex_hits():
    from nemo_safe_synthesizer.pii_replacer.detection.column_names import match_labels

    patterns = {
        "first_name": [r"name"],
        "full_name": [r"name"],
        "email": [r"mail"],
    }
    assert match_labels("patient_name", patterns) == ["first_name", "full_name"]
    assert match_labels("email", patterns) == ["email"]
    assert match_labels("weight", patterns) == []


def test_fuzzy_match_label_resolves_multi_regex_with_warning(caplog, monkeypatch):
    """Overlapping name patterns pick the best fuzzy candidate and warn."""
    import logging

    from nemo_safe_synthesizer.pii_replacer.detection import column_names

    patterns = {
        "first_name": [r"name"],
        "full_name": [r"name"],
    }
    monkeypatch.setattr(
        column_names,
        "FUZZY_KEYWORDS",
        {
            "first_name": ("firstname", "fname"),
            "full_name": ("patientname", "fullname"),
        },
    )
    caplog.set_level(logging.WARNING)
    chosen = column_names.fuzzy_match_label("patient_name", patterns, threshold=0.86)
    assert chosen == "full_name"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "patient_name" in m
        and "multiple header labels" in m
        and "full_name" in m
        and "first_name" in m
        and "Review the replacement plan" in m
        and "bug report" in m
        for m in messages
    )


def test_match_column_header_resolves_entity_vs_demo_collision(caplog, monkeypatch):
    """Entity and demographic regex hits share the same multi-match warning path."""
    import logging

    from nemo_safe_synthesizer.pii_replacer.detection import column_names

    entity_patterns = {"first_name": [r"sex|name"]}
    demo_patterns = {"sex": [r"^sex$", r"gender"]}
    monkeypatch.setattr(column_names, "FUZZY_KEYWORDS", {"first_name": ("firstname", "fname")})
    monkeypatch.setattr(column_names, "DEMO_FUZZY_KEYWORDS", {"sex": ("sex", "gender")})
    caplog.set_level(logging.WARNING)

    name_label, demo_label = column_names.match_column_header("sex", entity_patterns, demo_patterns, threshold=0.86)
    assert name_label is None
    assert demo_label == "sex"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "sex" in m and "multiple header labels" in m and "first_name" in m and "chose 'sex'" in m for m in messages
    )


def test_fuzzy_match_label_single_regex_skips_warning(caplog):
    import logging

    from nemo_safe_synthesizer.pii_replacer.detection.column_names import fuzzy_match_label
    from nemo_safe_synthesizer.pii_replacer.entities import ENTITY_NAME_PATTERNS

    caplog.set_level(logging.WARNING)
    assert fuzzy_match_label("email", ENTITY_NAME_PATTERNS, threshold=0.86) == "email"
    assert not any("multiple header labels" in record.getMessage() for record in caplog.records)


def test_match_column_header_assigns_demo_without_entity():
    from nemo_safe_synthesizer.pii_replacer.detection.column_names import match_column_header
    from nemo_safe_synthesizer.pii_replacer.entities import DEMO_LABEL_PATTERNS, ENTITY_NAME_PATTERNS

    name_label, demo_label = match_column_header("gender", ENTITY_NAME_PATTERNS, DEMO_LABEL_PATTERNS, threshold=0.86)
    assert name_label is None
    assert demo_label == "gender"


def test_weak_id_headers_need_dominant_identifier_template():
    """English ``*id`` leftovers stay unplanned unless values look like opaque IDs."""
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 40
    cfg = config_from_replace_pii(ReplacePiiConfig())
    # Category-like values under weak headers must not be replaced.
    labels = pd.DataFrame(
        {
            "valid": (["yes", "no"] * (n // 2))[:n],
            "hybrid": [f"type_{i % 3}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
        }
    )
    label_plan = discover_plan(labels, None, cfg, ReplacePiiConfig())
    assert column_spec(label_plan, "valid") is None
    assert column_spec(label_plan, "hybrid") is None

    # Weak ``userid`` with a dominant opaque template is still planned.
    userids = pd.DataFrame(
        {
            "userid": [f"user{i:04d}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
        }
    )
    user_plan = discover_plan(userids, None, cfg, ReplacePiiConfig())
    user = column_spec(user_plan, "userid")
    assert user is not None and user.entity_type == EntityType.unique_identifier

    # Mostly-blank ``userid``: blanks must not dilute uniqueness of nonempty IDs.
    sparse = pd.DataFrame(
        {
            "userid": ([""] * (n - 10)) + [f"user{i:04d}" for i in range(10)],
            "first_name": [f"First{i}" for i in range(n)],
        }
    )
    sparse_plan = discover_plan(sparse, None, cfg, ReplacePiiConfig())
    sparse_user = column_spec(sparse_plan, "userid")
    assert sparse_user is not None and sparse_user.entity_type == EntityType.unique_identifier

    # Strong ``patient_id`` keeps today's path (no weak template gate).
    patients = pd.DataFrame(
        {
            "patient_id": [f"pmc-6{i:05d}-1" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
        }
    )
    patient_plan = discover_plan(patients, None, cfg, ReplacePiiConfig())
    patient = column_spec(patient_plan, "patient_id")
    assert patient is not None and patient.entity_type == EntityType.unique_identifier
