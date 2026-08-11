# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Header reading: normalization, aliases, and what a name alone may claim."""

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.config.replace_pii import (
    PiiEntity,
    PiiReplacerConfig,
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
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    dob = column_spec(plan.standalone_columns_to_replace, "born_on")
    assert dob is not None and dob.entity_type == PiiEntity.date_of_birth
    # national_id is entity-driven; lands in standalone_columns_to_replace
    nat = column_spec(plan.standalone_columns_to_replace, "aadhaar")
    assert nat is not None and nat.entity_type in {PiiEntity.national_id, PiiEntity.unique_identifier}


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
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    dob = column_spec(plan.standalone_columns_to_replace, "birth_ymd")
    mid = column_spec(plan.standalone_columns_to_replace, "member_id")
    assert dob is not None and dob.entity_type == PiiEntity.date_of_birth
    assert mid is not None and mid.entity_type == PiiEntity.unique_identifier


def test_ssn_shaped_order_code_not_detected_as_ssn():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame({"order_code": [f"{100 + i:03d}-45-{6000 + i:04d}" for i in range(n)]})
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    for persona in plan.persona_backed_columns:
        assert column_spec(persona.columns_to_replace, "order_code") is None
    assert column_spec(plan.standalone_columns_to_replace, "order_code") is None


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
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    planned = {spec.column_name for spec in plan.standalone_columns_to_replace}
    for persona in plan.persona_backed_columns:
        planned.update(spec.column_name for spec in persona.columns_to_replace)
    assert planned.isdisjoint({"contact", "misc_code", "blob", "endpoint"})


def test_email_header_requires_email_shaped_values(caplog):
    """A simple content check is required when one exists."""
    import logging

    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    caplog.set_level(logging.WARNING)
    n = 30
    df = pd.DataFrame({"email": [f"not-an-email-{i}" for i in range(n)]})
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    for persona in plan.persona_backed_columns:
        assert column_spec(persona.columns_to_replace, "email") is None
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
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    street_cols = {
        spec.column_name
        for persona in plan.persona_backed_columns
        for spec in persona.columns_to_replace
        if spec.entity_type == PiiEntity.street_address
    }
    assert {"MailingStreet", "AddressLine1", "HomeStreet"} <= street_cols
    for col in ("MailingStreet", "AddressLine1", "HomeStreet"):
        assert column_spec(plan.standalone_columns_to_replace, col) is None


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
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    ip = column_spec(plan.standalone_columns_to_replace, "ipaddress")
    assert ip is not None and ip.entity_type == PiiEntity.ipv4
    for persona in plan.persona_backed_columns:
        assert column_spec(persona.columns_to_replace, "ipaddress") is None


def test_entity_name_patterns_have_fuzzy_keywords():
    from nemo_safe_synthesizer.pii_replacer.entities import ENTITY_NAME_PATTERNS, FUZZY_KEYWORDS

    missing = sorted(set(ENTITY_NAME_PATTERNS) - set(FUZZY_KEYWORDS))
    assert missing == [], f"entity types missing fuzzy keywords: {missing}"
