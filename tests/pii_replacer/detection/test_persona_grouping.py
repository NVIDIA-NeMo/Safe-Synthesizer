# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which columns belong to one person: role keys, agreement, and splits."""

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig
from nemo_safe_synthesizer.pii_replacer.entities import config_from_replace_pii


def test_multi_role_full_name_columns_get_distinct_person_ids():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "attending_name": [f"Dr Attending{i}" for i in range(n)],
            "surgeon_name": [f"Dr Surgeon{i}" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    personas = {p.persona for p in plan.persona_backed_columns}
    assert personas == {"attending", "surgeon"}
    cols = {spec.column_name for persona in plan.persona_backed_columns for spec in persona.columns_to_replace}
    assert {"attending_name", "surgeon_name"} <= cols


def test_non_medical_name_roles_discovered_as_distinct_personas():
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
    # Do not steal ID columns that only share a role prefix.
    assert match_label("applicant_id", ENTITY_NAME_PATTERNS) != "full_name"
    assert match_label("dependent_id", ENTITY_NAME_PATTERNS) != "full_name"

    n = 30
    df = pd.DataFrame(
        {
            "policyholder_name": [f"Holder {i}" for i in range(n)],
            "attorney_name": [f"Counsel {i}" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    personas = {p.persona for p in plan.persona_backed_columns}
    assert personas == {"policyholder", "attorney"}
    cols = {spec.column_name for persona in plan.persona_backed_columns for spec in persona.columns_to_replace}
    assert {"policyholder_name", "attorney_name"} <= cols


def test_persona_role_key_from_column_name():
    from nemo_safe_synthesizer.pii_replacer.detection.persona_grouping import _persona_role_key

    assert _persona_role_key("patient_first_name") == "patient"
    assert _persona_role_key("provider_email") == "provider"
    assert _persona_role_key("emergency_contact_name") == "emergency_contact"
    assert _persona_role_key("first_name") == ""
    assert _persona_role_key("Name") == ""
    assert _persona_role_key("primary_name") == ""


def test_agreeing_name_parts_share_role_persona():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "patient_first_name": [f"Alice{i}" for i in range(n)],
            "patient_last_name": [f"Smith{i}" for i in range(n)],
            "patient_full_name": [f"Alice{i} Smith{i}" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    assert len(plan.persona_backed_columns) == 1
    assert plan.persona_backed_columns[0].persona == "patient"
    cols = {spec.column_name for spec in plan.persona_backed_columns[0].columns_to_replace}
    assert cols == {"patient_first_name", "patient_last_name", "patient_full_name"}


def test_disagreeing_full_name_gets_split_persona(caplog):
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
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    personas = {p.persona: p for p in plan.persona_backed_columns}
    assert "patient" in personas
    assert "patient_2" in personas
    patient_cols = {s.column_name for s in personas["patient"].columns_to_replace}
    split_cols = {s.column_name for s in personas["patient_2"].columns_to_replace}
    assert patient_cols == {"patient_first_name", "patient_last_name"}
    assert split_cols == {"patient_full_name"}
    assert any("does not agree" in r.getMessage() for r in caplog.records)


def test_demo_only_persona_omits_match_persona_by_from_plan():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "sex": (["Female", "Male"] * (n // 2))[:n],
            "race": (["White", "Black"] * (n // 2))[:n],
            "amount": list(range(n)),
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    assert plan.persona_backed_columns == []
