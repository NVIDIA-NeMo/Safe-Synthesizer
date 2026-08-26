# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-person clustering: role keys, name agreement, and splits."""

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.config.replace_pii import PiiSamplerBackend, PiiSamplerConfig, ReplacePiiConfig
from nemo_safe_synthesizer.pii_replacer.entities import config_from_replace_pii
from tests.pii_replacer.helpers import column_spec, depends_on_columns


def test_multi_role_full_name_columns_are_distinct_replace_targets():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "attending_name": [f"Dr Attending{i}" for i in range(n)],
            "surgeon_name": [f"Dr Surgeon{i}" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    cols = {spec.column_name for spec in plan.columns_to_replace}
    assert {"attending_name", "surgeon_name"} <= cols
    # Distinct roles: no depends_on linking them.
    attending = column_spec(plan, "attending_name")
    surgeon = column_spec(plan, "surgeon_name")
    assert attending is not None and surgeon is not None
    assert "surgeon_name" not in depends_on_columns(attending)
    assert "attending_name" not in depends_on_columns(surgeon)


def test_non_medical_name_roles_discovered_as_distinct_columns():
    from nemo_safe_synthesizer.pii_replacer.detection.column_names import match_label
    from nemo_safe_synthesizer.pii_replacer.entities import ENTITY_NAME_PATTERNS
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    for col in (
        "policyholder_name",
        "guarantor_name",
        "emergency_contact",
        "attorney_name",
        "student_name",
        "account_holder",
    ):
        assert match_label(col, ENTITY_NAME_PATTERNS) == "full_name"
    assert match_label("applicant_id", ENTITY_NAME_PATTERNS) != "full_name"
    assert match_label("dependent_id", ENTITY_NAME_PATTERNS) != "full_name"

    n = 30
    df = pd.DataFrame(
        {
            "policyholder_name": [f"Holder {i}" for i in range(n)],
            "attorney_name": [f"Counsel {i}" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    cols = {spec.column_name for spec in plan.columns_to_replace}
    assert {"policyholder_name", "attorney_name"} <= cols


def test_role_key_from_column_name():
    from nemo_safe_synthesizer.pii_replacer.detection.column_grouping import _role_key

    assert _role_key("patient_first_name") == "patient"
    assert _role_key("provider_email") == "provider"
    assert _role_key("emergency_contact_name") == "emergency_contact"
    assert _role_key("first_name") == ""
    assert _role_key("Name") == ""
    assert _role_key("primary_name") == ""


def test_agreeing_name_parts_share_depends_on_edges():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "patient_first_name": [f"Alice{i}" for i in range(n)],
            "patient_last_name": [f"Smith{i}" for i in range(n)],
            "patient_full_name": [f"Alice{i} Smith{i}" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    cols = {spec.column_name for spec in plan.columns_to_replace}
    assert cols == {"patient_first_name", "patient_last_name", "patient_full_name"}
    first = column_spec(plan, "patient_first_name")
    last = column_spec(plan, "patient_last_name")
    assert first is not None and last is not None
    assert "patient_full_name" in depends_on_columns(first)
    assert "patient_full_name" in depends_on_columns(last)


def test_disagreeing_full_name_does_not_link_to_name_parts(caplog):
    import logging

    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    caplog.set_level(logging.WARNING)
    n = 30
    df = pd.DataFrame(
        {
            "patient_first_name": ["Alice"] * n,
            "patient_last_name": ["Smith"] * n,
            "patient_full_name": ["Bob Jones"] * n,
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    cols = {spec.column_name for spec in plan.columns_to_replace}
    assert {"patient_first_name", "patient_last_name", "patient_full_name"} <= cols
    first = column_spec(plan, "patient_first_name")
    assert first is not None
    assert "patient_full_name" not in depends_on_columns(first)
    assert any("does not agree" in r.getMessage() for r in caplog.records)


def test_demo_only_group_omitted_from_plan():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "sex": (["Female", "Male"] * (n // 2))[:n],
            "race": (["White", "Black"] * (n // 2))[:n],
            "amount": list(range(n)),
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    assert plan.columns_to_replace == []


def test_group_constant_name_and_varying_email_do_not_share_depends_on(fixture_group_grain_df: pd.DataFrame):
    """One name per group cannot condition a per-row email (different grain)."""
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    config = ReplacePiiConfig(sampler=PiiSamplerConfig(backend=PiiSamplerBackend.faker))
    plan = discover_plan(fixture_group_grain_df, "patient_id", config_from_replace_pii(config), config)

    name = column_spec(plan, "full_name")
    email = column_spec(plan, "email")
    assert name is not None and email is not None
    assert "full_name" not in depends_on_columns(email)
