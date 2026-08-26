# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-subject person columns and depends_on edges."""

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.config.replace_pii import PiiSamplerBackend, PiiSamplerConfig, ReplacePiiConfig
from nemo_safe_synthesizer.pii_replacer.entities import config_from_replace_pii
from tests.pii_replacer.helpers import column_spec, depends_on_columns


def test_duplicate_full_name_columns_emit_unlinked_plan():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan, discover_plan_with_hints

    n = 30
    df = pd.DataFrame(
        {
            "attending_name": [f"Dr Attending{i}" for i in range(n)],
            "surgeon_name": [f"Dr Surgeon{i}" for i in range(n)],
            "gender": (["Female", "Male"] * (n // 2))[:n],
        }
    )
    cfg = config_from_replace_pii(ReplacePiiConfig())
    plan, hints = discover_plan_with_hints(df, None, cfg, ReplacePiiConfig())
    assert {s.column_name for s in plan.columns_to_replace} == {"attending_name", "surgeon_name"}
    assert all(not s.depends_on for s in plan.columns_to_replace)
    assert any("attending_name" in h and "gender" in h for h in hints)
    assert any("surgeon_name" in h and "gender" in h for h in hints)
    # discover_plan stays a thin wrapper
    assert discover_plan(df, None, cfg, ReplacePiiConfig()).columns_to_replace


def test_non_medical_name_roles_still_match_full_name():
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
    df = pd.DataFrame({"policyholder_name": [f"Holder {i}" for i in range(n)]})
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    assert column_spec(plan, "policyholder_name") is not None


def test_name_parts_and_full_name_share_depends_on_edges():
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


def test_disagreeing_values_still_link_full_name_depends_on():
    """Heuristics mode does not check name agreement; one subject links by entity type."""
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "patient_first_name": ["Alice"] * n,
            "patient_last_name": ["Smith"] * n,
            "patient_full_name": ["Bob Jones"] * n,
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    first = column_spec(plan, "patient_first_name")
    assert first is not None
    assert "patient_full_name" in depends_on_columns(first)


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


def test_group_scope_still_links_name_and_email(fixture_group_scope_df: pd.DataFrame):
    """Group key sets plan scope only; email still depends_on full_name."""
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    config = ReplacePiiConfig(sampler=PiiSamplerConfig(backend=PiiSamplerBackend.faker))
    plan = discover_plan(fixture_group_scope_df, "patient_id", config_from_replace_pii(config), config)

    name = column_spec(plan, "full_name")
    email = column_spec(plan, "email")
    assert name is not None and email is not None
    assert "full_name" in depends_on_columns(email)
    assert plan.scope.value == "group"
