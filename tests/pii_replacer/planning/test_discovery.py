# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column discovery, detection heuristics, and plan emission."""

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    PiiReplacementScope,
    PiiSamplerBackend,
    PiiSamplerConfig,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.entities import Config, config_from_replace_pii
from nemo_safe_synthesizer.pii_replacer.planning import discover_plan, discover_plan_with_hints
from tests.pii_replacer.helpers import PHONE_MINORITY, column_spec, depends_on_columns


def _cfg(backend: PiiSamplerBackend = PiiSamplerBackend.managed) -> tuple[ReplacePiiConfig, Config]:
    config = ReplacePiiConfig(sampler=PiiSamplerConfig(backend=backend))
    return config, config_from_replace_pii(config)


def test_discover_event_date_identified_not_replaced():
    dominant_dates = [f"04/{(i % 28) + 1:02d}/2023" for i in range(95)]
    df = pd.DataFrame(
        {
            "event_date": dominant_dates + ["08/2010"] * 4 + ["unknown"] * 1,
            "notes": [f"Patient record {i} visited clinic for follow up care today" for i in range(100)],
        }
    )
    config, cfg = _cfg()
    plan = discover_plan(df, group_key=None, cfg=cfg, config=config)
    assert column_spec(plan, "event_date") is None


def test_discovery_logs_temporal_and_free_text_gates(caplog):
    import logging

    caplog.set_level(logging.INFO)
    dominant_dates = [f"04/{(i % 28) + 1:02d}/2023" for i in range(95)]
    df = pd.DataFrame(
        {
            "first_name": [f"First{i}" for i in range(100)],
            "event_date": dominant_dates + ["08/2010"] * 4 + ["unknown"] * 1,
            "weight": [135.0] * 100,
            "notes": [f"Patient record {i} visited clinic for follow up care today" for i in range(100)],
        }
    )
    config, cfg = _cfg()
    discover_plan(df, group_key=None, cfg=cfg, config=config)
    messages = [record.getMessage() for record in caplog.records]
    assert any("Identified temporal column 'event_date'" in message for message in messages)


def test_detected_to_plan_warns_on_unmapped_entity_label(caplog):
    import logging

    from nemo_safe_synthesizer.pii_replacer.models import DiscoveryResult
    from nemo_safe_synthesizer.pii_replacer.planning.discovery import _detected_to_plan

    caplog.set_level(logging.WARNING)
    plan, hints = _detected_to_plan(
        DiscoveryResult.from_dict(
            {
                "same_person_bundles": [
                    {
                        "bundle_id": "person",
                        "fields": {"not_a_real_entity": {"column": "weird_col", "pattern": None}},
                        "demographics": {},
                    }
                ],
                "standalone_columns": [],
                "identified_not_replaced": [],
                "free_text_columns": [],
            }
        ),
        scope=PiiReplacementScope.dataframe,
    )
    assert plan.columns_to_replace == []
    assert hints == []
    assert any("not_a_real_entity" in r.getMessage() for r in caplog.records)


def test_discover_date_of_birth_gets_its_format():
    df = pd.DataFrame(
        {
            "date_of_birth": [f"01/{(i % 28) + 1:02d}/1990" for i in range(40)],
            "first_name": [f"First{i}" for i in range(40)],
        }
    )
    config, cfg = _cfg()
    plan = discover_plan(df, None, cfg, config)
    dob = column_spec(plan, "date_of_birth")
    assert dob is not None
    assert dob.entity_type is EntityType.date_of_birth
    assert dob.pattern == "%m/%d/%Y"


def test_discovery_routes_phone_standalone_with_its_own_shape(fixture_phone_df: pd.DataFrame):
    config, cfg = _cfg(PiiSamplerBackend.faker)
    plan = discover_plan(fixture_phone_df, group_key=None, cfg=cfg, config=config)
    phone = column_spec(plan, "phone")
    assert phone is not None
    assert phone.entity_type is EntityType.phone_number
    # Dominant +1-415-555-#### covers ≥85%; minority (206) formats are omitted from the plan.
    assert phone.pattern is not None
    assert "415" in phone.pattern or "#" in phone.pattern
    assert PHONE_MINORITY not in (phone.pattern or "")


def test_discovery_reports_no_pattern_for_mixed_formats():
    """Below 85% coverage, discovery omits pattern rather than listing secondaries."""
    values = [f"PMC-{(i * 24851) % 1000000:06d}" for i in range(50)] + [
        f"ACC-{(i * 937) % 10000:04d}" for i in range(50)
    ]
    df = pd.DataFrame({"unique_id": values, "first_name": [f"N{i}" for i in range(100)]})
    config, cfg = _cfg()
    plan = discover_plan(df, None, cfg, config)
    uid = column_spec(plan, "unique_id")
    assert uid is not None
    assert uid.pattern is None


def test_discovery_reads_the_name_and_email_conventions(fixture_contact_df: pd.DataFrame):
    config, cfg = _cfg()
    plan = discover_plan(fixture_contact_df, group_key=None, cfg=cfg, config=config)
    name = column_spec(plan, "patient_name")
    email = column_spec(plan, "patient_email")
    assert name is not None and name.entity_type is EntityType.full_name
    assert email is not None and email.entity_type is EntityType.email
    assert name.pattern is not None
    assert email.pattern is not None
    # Email depends on full_name (no name parts in this fixture).
    assert "patient_name" in depends_on_columns(email)


def test_discovery_leaves_ip_columns_without_pattern():
    df = pd.DataFrame(
        {
            "ipv4": [f"10.0.0.{i}" for i in range(40)],
            "first_name": [f"N{i}" for i in range(40)],
        }
    )
    config, cfg = _cfg()
    plan = discover_plan(df, None, cfg, config)
    ip = column_spec(plan, "ipv4")
    assert ip is not None
    assert ip.pattern is None


def test_first_name_depends_on_gender_when_no_full_name():
    n = 40
    df = pd.DataFrame(
        {
            "first_name": [f"Alice{i}" for i in range(n)],
            "last_name": [f"Smith{i}" for i in range(n)],
            "gender": (["Female", "Male"] * (n // 2))[:n],
            "ethnicity": (["White", "Black"] * (n // 2))[:n],
        }
    )
    config, cfg = _cfg(PiiSamplerBackend.managed)
    plan = discover_plan(df, None, cfg, config)
    first = column_spec(plan, "first_name")
    last = column_spec(plan, "last_name")
    assert first is not None
    assert "gender" in depends_on_columns(first)
    assert "ethnicity" in depends_on_columns(first)
    assert last is not None
    assert "ethnicity" in depends_on_columns(last)
    assert "gender" not in depends_on_columns(last)


def test_faker_omits_ethnic_background_conditioner():
    n = 40
    df = pd.DataFrame(
        {
            "first_name": [f"Alice{i}" for i in range(n)],
            "gender": (["Female", "Male"] * (n // 2))[:n],
            "ethnicity": (["White", "Black"] * (n // 2))[:n],
        }
    )
    config, cfg = _cfg(PiiSamplerBackend.faker)
    plan = discover_plan(df, None, cfg, config)
    first = column_spec(plan, "first_name")
    assert first is not None
    assert "gender" in depends_on_columns(first)
    assert "ethnicity" not in depends_on_columns(first)


def test_name_parts_prefer_full_name_over_demographics():
    n = 40
    df = pd.DataFrame(
        {
            "first_name": [f"Alice{i}" for i in range(n)],
            "last_name": [f"Smith{i}" for i in range(n)],
            "full_name": [f"Alice{i} Smith{i}" for i in range(n)],
            "gender": (["Female", "Male"] * (n // 2))[:n],
        }
    )
    config, cfg = _cfg()
    plan = discover_plan(df, None, cfg, config)
    first = column_spec(plan, "first_name")
    assert first is not None
    assert "full_name" in depends_on_columns(first)
    assert "gender" not in depends_on_columns(first)


def test_prefix_free_columns_still_link_full_name_depends_on():
    """Distinct headers still emit first/last → full_name edges under one subject."""
    n = 40
    df = pd.DataFrame(
        {
            "first_name": [f"Alice{i}" for i in range(n)],
            "last_name": [f"Smith{i}" for i in range(n)],
            "legal_name": [f"Alice{i} Smith{i}" for i in range(n)],
            "gender": (["Female", "Male"] * (n // 2))[:n],
        }
    )
    config, cfg = _cfg()
    plan = discover_plan(df, None, cfg, config)
    first = column_spec(plan, "first_name")
    last = column_spec(plan, "last_name")
    legal = column_spec(plan, "legal_name")
    assert first is not None and last is not None and legal is not None
    assert legal.entity_type is EntityType.full_name
    assert "legal_name" in depends_on_columns(first)
    assert "legal_name" in depends_on_columns(last)
    assert "gender" not in depends_on_columns(first)


def test_email_prefers_name_parts_over_full_name():
    n = 40
    df = pd.DataFrame(
        {
            "first_name": [f"Alice{i}" for i in range(n)],
            "last_name": [f"Smith{i}" for i in range(n)],
            "full_name": [f"Alice{i} Smith{i}" for i in range(n)],
            "email": [f"alice{i}.smith{i}@acme.com" for i in range(n)],
        }
    )
    config, cfg = _cfg()
    plan = discover_plan(df, None, cfg, config)
    email = column_spec(plan, "email")
    assert email is not None
    deps = depends_on_columns(email)
    assert "first_name" in deps and "last_name" in deps
    assert "full_name" not in deps


def test_duplicate_persona_columns_emit_unlinked_plan_with_hints():
    """Duplicate persona entity types → flat plan, empty depends_on, YAML hints."""
    n = 40
    df = pd.DataFrame(
        {
            "first_name": [f"Alice{i}" for i in range(n)],
            "spouse_first_name": [f"Bob{i}" for i in range(n)],
            "full_name": [f"Alice{i} Smith{i}" for i in range(n)],
            "gender": (["Female", "Male"] * (n // 2))[:n],
        }
    )
    config, cfg = _cfg()
    plan, hints = discover_plan_with_hints(df, None, cfg, config)
    cols = {spec.column_name for spec in plan.columns_to_replace}
    assert cols == {"first_name", "spouse_first_name", "full_name"}
    assert all(not spec.depends_on for spec in plan.columns_to_replace)
    # name parts prefer full_name when present; full_name can depend on gender
    assert any("first_name (first_name) can depends_on full_name" in h for h in hints)
    assert any("spouse_first_name (first_name) can depends_on full_name" in h for h in hints)
    assert any("full_name (full_name) can depends_on gender" in h for h in hints)

    from nemo_safe_synthesizer.pii_replacer.planning import plan_to_commented_yaml

    yaml_text = plan_to_commented_yaml(plan, depends_on_hints=hints)
    assert "# depends_on omitted:" in yaml_text
    assert "#   - first_name (first_name) can depends_on full_name" in yaml_text


def test_discover_plan_falls_back_to_dataframe_scope_when_group_key_missing(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    df = pd.DataFrame({"first_name": [f"N{i}" for i in range(30)]})
    config, cfg = _cfg()
    plan = discover_plan(df, group_key="missing_id", cfg=cfg, config=config)
    assert plan.scope is PiiReplacementScope.dataframe
    assert any("dataframe scope" in r.getMessage() for r in caplog.records)


def test_discover_plan_uses_group_scope_when_key_present():
    df = pd.DataFrame(
        {
            "patient_id": ["A", "A", "B", "B"] * 10,
            "first_name": [f"N{i}" for i in range(40)],
        }
    )
    config, cfg = _cfg()
    plan = discover_plan(df, group_key="patient_id", cfg=cfg, config=config)
    assert plan.scope is PiiReplacementScope.group


def test_street_address_has_no_geo_depends_on():
    n = 40
    df = pd.DataFrame(
        {
            "street_address": [f"{100 + i} Main St" for i in range(n)],
            "city": ["Springfield"] * n,
            "first_name": [f"N{i}" for i in range(n)],
        }
    )
    config, cfg = _cfg()
    plan = discover_plan(df, None, cfg, config)
    street = column_spec(plan, "street_address")
    if street is not None:
        assert street.depends_on == []
