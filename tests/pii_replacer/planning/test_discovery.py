# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column discovery, detection heuristics, and plan emission."""

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    AUTO_DISCOVERY,
    PersonaColumnSet,
    PersonaMatchColumn,
    PiiColumnPlan,
    PiiEntity,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    PiiReplacementScope,
    PiiReplacerConfig,
)
from nemo_safe_synthesizer.pii_replacer.entities import Config, config_from_replace_pii
from nemo_safe_synthesizer.pii_replacer.planning import (
    iter_plan_advisories,
    resolve_plan,
    validate_plan,
)
from nemo_safe_synthesizer.pii_replacer.replacer import (
    TabularPiiReplacer,
)
from tests.pii_replacer.helpers import PHONE_MINORITY, column_spec, persona_set


def test_discover_event_date_identified_not_replaced():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    dominant_dates = [f"04/{(i % 28) + 1:02d}/2023" for i in range(95)]
    df = pd.DataFrame(
        {
            "event_date": dominant_dates + ["08/2010"] * 4 + ["unknown"] * 1,
            "notes": [f"Patient record {i} visited clinic for follow up care today" for i in range(100)],
        }
    )
    plan = discover_plan(
        df,
        group_key=None,
        cfg=config_from_replace_pii(PiiReplacerConfig()),
        config=PiiReplacerConfig(),
    )
    # A generic date column is identified as structured only to keep it out of the
    # free-text path; it is excluded from the replacement plan entirely.
    assert column_spec(plan.standalone_columns_to_replace, "event_date") is None
    for col_set in plan.persona_backed_columns:
        assert column_spec(col_set.columns_to_replace, "event_date") is None


def test_discovery_logs_temporal_and_free_text_gates(caplog):
    import logging

    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    caplog.set_level(logging.INFO)
    dominant_dates = [f"04/{(i % 28) + 1:02d}/2023" for i in range(95)]
    df = pd.DataFrame(
        {
            # A structured (person) column so free-text scanning is not skipped:
            # this test exercises the dtype/field-type free-text gate logging.
            "first_name": [f"First{i}" for i in range(100)],
            "event_date": dominant_dates + ["08/2010"] * 4 + ["unknown"] * 1,
            "weight": [135.0] * 100,
            "notes": [f"Patient record {i} visited clinic for follow up care today" for i in range(100)],
        }
    )
    discover_plan(
        df,
        group_key=None,
        cfg=config_from_replace_pii(PiiReplacerConfig()),
        config=PiiReplacerConfig(),
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any("Identified temporal column 'event_date'" in message for message in messages)
    free_text_msgs = [m for m in messages if "Free-text scan for PII detection" in m]
    assert free_text_msgs
    assert "scanned as text: notes" in free_text_msgs[0]
    assert "weight" in free_text_msgs[0].split("not scanned:", 1)[1]


def test_detected_to_plan_warns_on_unmapped_entity_label(caplog):
    """Detector/PiiEntity vocabulary drift must not silently drop a column."""
    import logging

    from nemo_safe_synthesizer.pii_replacer.planning.discovery import _detected_to_plan

    caplog.set_level(logging.WARNING)
    plan = _detected_to_plan(
        {
            "personas": [
                {
                    "persona": "person_1",
                    "fields": {
                        "first_name": {"column": "first_name", "patterns": []},
                        "alien_id": {"column": "alien_col", "patterns": []},
                    },
                    "match_persona_by": [],
                }
            ],
            "standalone_columns": [{"column": "token_col", "entity": "mystery_token", "patterns": []}],
            "free_text_columns": [],
        },
        scope=PiiReplacementScope.dataframe,
    )
    assert column_spec(plan.persona_backed_columns[0].columns_to_replace, "first_name") is not None
    assert column_spec(plan.persona_backed_columns[0].columns_to_replace, "alien_col") is None
    assert column_spec(plan.standalone_columns_to_replace, "token_col") is None
    messages = [record.getMessage() for record in caplog.records]
    assert any("alien_col" in m and "alien_id" in m and "not a known PiiEntity" in m for m in messages)
    assert any("token_col" in m and "mystery_token" in m and "not a known PiiEntity" in m for m in messages)
    # Log metadata only — never sample values from the column.
    assert not any("secret" in m.lower() for m in messages)


def test_discover_temporal_columns_identified_not_replaced():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 100
    df = pd.DataFrame(
        {
            # A structured (person) column so free-text scanning is not skipped;
            # this test verifies temporal columns are identified-but-not-replaced
            # while genuine free text is still planned.
            "first_name": [f"First{i}" for i in range(n)],
            "created_at": [f"2023-04-{(i % 28) + 1:02d} 14:30:00" for i in range(95)] + ["2023-05-01 09:00:00"] * 5,
            "shift_start": [f"{(i % 24):02d}:00:00" for i in range(95)] + ["09:15:00"] * 5,
            "wait_time": [f"PT{(i % 20) + 1}H30M" for i in range(95)] + ["45 min"] * 5,
            "notes": [f"Patient record {i} visited clinic for follow up care today" for i in range(n)],
        }
    )
    plan = discover_plan(
        df,
        group_key=None,
        cfg=config_from_replace_pii(PiiReplacerConfig()),
        config=PiiReplacerConfig(),
    )
    for col in ("created_at", "shift_start", "wait_time"):
        assert column_spec(plan.standalone_columns_to_replace, col) is None
        for col_set in plan.persona_backed_columns:
            assert column_spec(col_set.columns_to_replace, col) is None
    assert column_spec(plan.standalone_columns_to_replace, "notes") is not None


def test_discover_date_of_birth_gets_its_format():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 30
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
            "date_of_birth": [f"{(i % 12) + 1:02d}/{(i % 28) + 1:02d}/19{60 + (i % 30):02d}" for i in range(n)],
        }
    )
    plan = discover_plan(
        df,
        group_key="patient_id",
        cfg=config_from_replace_pii(PiiReplacerConfig()),
        config=PiiReplacerConfig(),
    )
    assert plan.scope == PiiReplacementScope.group
    # Birth dates are replaced independently of any persona, so they are placed
    # in the standalone section (not under a persona).
    dob_spec = column_spec(plan.standalone_columns_to_replace, "date_of_birth")
    assert dob_spec is not None
    assert dob_spec.entity_type == PiiEntity.date_of_birth
    assert dob_spec.patterns == ["%m/%d/%Y"]
    for col_set in plan.persona_backed_columns:
        assert column_spec(col_set.columns_to_replace, "date_of_birth") is None


def test_a_repeating_identifier_is_detected_with_or_without_a_group_key():
    """A column named like an ID is offered even when its values repeat across rows.

    How widely one value keeps its replacement is what scope decides. A group key
    still measures group-constant columns against groups (for free-text and scope
    tagging), but it is no longer required to surface the identifier itself.
    """
    from nemo_safe_synthesizer.pii_replacer.planning.discovery import _detect_full_dataframe

    rows = []
    for p in range(40):
        for i in range(3):
            rows.append(
                {
                    "patient_id": f"pmc-{6 if p % 2 else 8}{p:05d}-{(p % 4) + 1}",
                    "record_id": f"REC-{p:06d}",  # constant within a patient
                    "event_id": f"{p * 3 + i:08d}",  # per-row unique
                    "first_name": f"First{p}",
                }
            )
    df = pd.DataFrame(rows)
    cfg = config_from_replace_pii(PiiReplacerConfig())

    for group_key in (None, "patient_id"):
        discovery = _detect_full_dataframe(df, cfg, group_key=group_key)
        entities = {e.column: e for e in discovery.standalone_columns}
        for col in ("patient_id", "record_id", "event_id"):
            assert entities[col].entity == "unique_identifier"
            assert entities[col].patterns


def test_match_persona_by_only_sex_and_race_in_plan():
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="primary_person",
                columns_to_replace=[PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name)],
                match_persona_by=[
                    PersonaMatchColumn(persona_attribute="sex", column_name="sex"),
                    PersonaMatchColumn(persona_attribute="ethnic_background", column_name="race"),
                ],
            )
        ]
    )
    matchers = [(cond.persona_attribute, cond.column_name) for cond in plan.persona_backed_columns[0].match_persona_by]
    assert matchers == [
        ("sex", "sex"),
        ("ethnic_background", "race"),
    ]


def test_faker_discovery_omits_ethnic_background_from_match_persona_by():
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 40
    df = pd.DataFrame(
        {
            "first_name": [f"First{i}" for i in range(n)],
            "sex": (["Female", "Male"] * (n // 2))[:n],
            "race": (["White", "Black"] * (n // 2))[:n],
        }
    )
    cfg = PiiReplacerConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    plan = discover_plan(df, None, config_from_replace_pii(cfg), cfg)
    assert plan.persona_backed_columns
    matchers = [
        (cond.persona_attribute, cond.column_name)
        for col_set in plan.persona_backed_columns
        for cond in col_set.match_persona_by
    ]
    assert ("sex", "sex") in matchers
    assert not any(attr == "ethnic_background" for attr, _ in matchers)


def test_faker_ignores_ethnic_background_in_hand_written_plan():
    from nemo_safe_synthesizer.pii_replacer.replacement import PersonaEngine, extract_instances

    n = 20
    df = pd.DataFrame(
        {
            "first_name": [f"First{i}" for i in range(n)],
            "sex": (["Female", "Male"] * (n // 2))[:n],
            "race": (["White", "Black"] * (n // 2))[:n],
        }
    )
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="primary_person",
                columns_to_replace=[PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name)],
                match_persona_by=[
                    PersonaMatchColumn(persona_attribute="sex", column_name="sex"),
                    PersonaMatchColumn(persona_attribute="ethnic_background", column_name="race"),
                ],
            )
        ]
    )
    runtime = config_from_replace_pii(PiiReplacerConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker)))
    assert runtime.persona_backend == "faker"
    codes = {issue.code for issue in iter_plan_advisories(plan, persona_backend="faker")}
    assert "pii_plan_ethnic_background_ignored_under_faker" in codes

    instances = extract_instances(df, plan, runtime)
    assert all(inst.get("select_field_values") is None for inst in instances)
    assert all(inst.get("race_raw") is None for inst in instances)

    engine = PersonaEngine(runtime, len(instances))
    engine.assign(instances)
    # Sex conditioning still applies; ethnicity was not used for bucketing/sampling.
    assert all(inst.synthetic_person is not None and inst.synthetic_person["sex"] == inst.sex for inst in instances)


def test_discovery_routes_phone_standalone_with_its_own_shape(fixture_phone_df: pd.DataFrame):
    """No backend but the PGM has a phone, so the column is replaced from its own shape."""
    from nemo_safe_synthesizer.pii_replacer.patterns import value_matches_template
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    cfg = PiiReplacerConfig()
    plan = discover_plan(fixture_phone_df, group_key=None, cfg=config_from_replace_pii(cfg), config=cfg)

    assert not any(
        spec.column_name == "phone" for col_set in plan.persona_backed_columns for spec in col_set.columns_to_replace
    )
    spec = column_spec(plan.standalone_columns_to_replace, "phone")
    assert spec is not None
    assert spec.entity_type == PiiEntity.phone_number
    # The shape is inferred from the column itself, since no entity regex reports one.
    # Two numbers are too few to read a second shape from, so those keep their own.
    assert len(spec.patterns) == 1
    assert value_matches_template(fixture_phone_df["phone"][0], spec.patterns[0])
    assert not value_matches_template(PHONE_MINORITY, spec.patterns[0])


def test_discovery_reports_no_pattern_for_a_column_that_wears_none():
    """Detection measures coverage against the entity, which no template may describe.

    An IPv4 column whose octets vary in width has a 100%-confident entity and no
    inferable template; carrying that entity's coverage into the plan made
    discovery emit a plan its own validation rejects.
    """
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    ips = [f"{(i * 37) % 240 + 1}.{(i * 11) % 256}.{(i * 7) % 256}.{(i * 3) % 256}" for i in range(40)]
    df = pd.DataFrame({"ip_address": ips, "full_name": [f"Person {i}" for i in range(40)]})
    cfg = PiiReplacerConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    plan = discover_plan(df, group_key=None, cfg=config_from_replace_pii(cfg), config=cfg)

    spec = column_spec(plan.standalone_columns_to_replace, "ip_address")
    assert spec is not None
    assert spec.patterns == []
    # The plan resolved here is the one validation runs on, so the run completes.
    replacer = TabularPiiReplacer(cfg, data_config=DataParameters())
    replacer.transform_df(df)
    assert replacer.result is not None
    assert (replacer.result.transformed_df["ip_address"] != df["ip_address"]).all()


def test_discovery_reads_the_name_and_email_conventions(fixture_contact_df: pd.DataFrame):
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    cfg = PiiReplacerConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    plan = discover_plan(fixture_contact_df, group_key=None, cfg=config_from_replace_pii(cfg), config=cfg)
    specs = persona_set(plan, "patient").columns_to_replace

    name = column_spec(specs, "patient_name")
    email = column_spec(specs, "patient_email")
    assert name is not None and name.patterns == ["{LAST}, {First}"]
    assert email is not None and email.patterns == ["{f}.{last}@{domain}"]


def test_discovery_leaves_ip_columns_to_their_generator():
    """A template counts characters, so '###' would happily emit an octet of 660."""
    import ipaddress

    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    # Fixed-width octets: a template is inferable here, and still must not be used.
    ips = [f"{101 + i}.{110 + i}.{120 + i}.{130 + i}" for i in range(40)]
    df = pd.DataFrame({"ip_address": ips, "full_name": [f"Person {i}" for i in range(40)]})
    cfg = PiiReplacerConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    plan = discover_plan(df, group_key=None, cfg=config_from_replace_pii(cfg), config=cfg)

    spec = column_spec(plan.standalone_columns_to_replace, "ip_address")
    assert spec is not None
    assert spec.patterns == []

    replacer = TabularPiiReplacer(cfg, data_config=DataParameters())
    replacer.transform_df(df)
    assert replacer.result is not None
    for value in replacer.result.transformed_df["ip_address"]:
        ipaddress.IPv4Address(value)


def test_discovery_replaces_every_phone_column():
    """Each phone column stands on its own, so a second one is not left in the clear."""
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    df = pd.DataFrame(
        {
            "full_name": [f"Person {i}" for i in range(20)],
            "phone": [f"+1-415-555-{1000 + i:04d}" for i in range(20)],
            "mobile_phone": [f"(206) 555-{2000 + i:04d}" for i in range(20)],
        }
    )
    cfg = PiiReplacerConfig()
    plan = discover_plan(df, group_key=None, cfg=config_from_replace_pii(cfg), config=cfg)

    replaced = {spec.column_name for spec in plan.standalone_columns_to_replace}
    assert {"phone", "mobile_phone"} <= replaced


def test_auto_discovery_emits_plan_shape():
    n = 30
    # A person column ensures structured detection fires, so the free-text column
    # is scanned and planned (heuristic mode plans free text when any structured
    # entity column — persona-backed or standalone — exists).
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
            "notes": [f"Patient record {i} visited the clinic for follow up care today" for i in range(n)],
        }
    )
    replacer = TabularPiiReplacer(
        PiiReplacerConfig(replacement_plan=AUTO_DISCOVERY, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(df)
    assert replacer.resolved_plan is not None
    assert replacer.resolved_plan.scope == PiiReplacementScope.group
    notes = column_spec(replacer.resolved_plan.standalone_columns_to_replace, "notes")
    assert notes is not None
    assert notes.entity_type == PiiEntity.free_text


def test_an_identifier_that_repeats_is_still_an_identifier():
    """A ticket reference written on several rows is still a reference.

    Discovery reads a column of identifiers as identifiers whether or not their
    values repeat; how widely one value keeps its replacement is what scope
    decides. Refusing a repeated ID used to fail a whole auto-discovery run.
    """
    references = [f"550e8400-e29b-41d4-a716-4466554400{i:02d}" for i in range(20)]
    df = pd.DataFrame({"ticket_ref": references * 2, "note": [f"row {i}" for i in range(40)]})
    config = PiiReplacerConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    runtime = config_from_replace_pii(config)

    plan = resolve_plan(config, df, data_config=DataParameters(), cfg=runtime)

    assert ("ticket_ref", PiiEntity.unique_identifier) in [
        (s.column_name, s.entity_type) for s in plan.standalone_columns_to_replace
    ]


def test_low_cardinality_name_matched_pii_is_still_planned():
    """Entity typing does not depend on how many distinct values a column has."""
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    df = pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(8)],
            "first_name": [f"First{i}" for i in range(8)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    patient = column_spec(plan.standalone_columns_to_replace, "patient_id")
    assert patient is not None and patient.entity_type == PiiEntity.unique_identifier
    first = None
    for persona in plan.persona_backed_columns:
        first = column_spec(persona.columns_to_replace, "first_name") or first
    assert first is not None and first.entity_type == PiiEntity.first_name


def test_discover_plan_falls_back_to_dataframe_scope_when_group_key_missing(caplog):
    import logging

    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    caplog.set_level(logging.WARNING)
    df = pd.DataFrame({"first_name": ["Alice", "Bob", "Cleo"] * 10})
    plan = discover_plan(
        df, group_key="missing_group", cfg=config_from_replace_pii(PiiReplacerConfig()), config=PiiReplacerConfig()
    )
    assert plan.scope == PiiReplacementScope.dataframe
    assert any("dataframe scope instead of group" in r.getMessage() for r in caplog.records)


def test_transcription_job_id_pattern_inferred():
    import uuid

    from nemo_safe_synthesizer.pii_replacer.detection import match_value_entity
    from nemo_safe_synthesizer.pii_replacer.patterns import value_patterns
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    samples = [
        f"transcription-job-2024-03-{(i % 28) + 1:02d}T{(i % 24):02d}-{(i % 60):02d}-{(i % 60):02d}Z-"
        f"call-{uuid.uuid4()}-{1734638760 + i}"
        for i in range(40)
    ]
    assert match_value_entity(samples[0]) == "unique_identifier"
    patterns = value_patterns(pd.Series(samples), Config())
    assert patterns, "expected a character template for transcription-job ids"
    assert patterns[0].startswith("transcription-job-")

    df = pd.DataFrame({"job_id": samples})
    plan = discover_plan(df, None, config_from_replace_pii(PiiReplacerConfig()), PiiReplacerConfig())
    spec = column_spec(plan.standalone_columns_to_replace, "job_id")
    assert spec is not None and spec.entity_type == PiiEntity.unique_identifier
    assert spec.patterns


def test_entity_registry_membership_matches_product_rules():
    """EntitySpec fields and effective_apply_path match the product routing table."""
    from nemo_safe_synthesizer.pii_replacer import entities
    from nemo_safe_synthesizer.pii_replacer.planning import (
        DATE_PATTERN_ENTITIES,
        PERSONA_PATTERN_ENTITIES,
        TEMPLATE_PATTERN_ENTITIES,
    )

    def labels_where(**attrs: object) -> list[str]:
        return sorted(
            s.label for s in entities.ENTITY_REGISTRY.values() if all(getattr(s, k) == v for k, v in attrs.items())
        )

    assert labels_where(apply_path="standalone_map") == [
        "api_key",
        "credit_debit_card",
        "date_of_birth",
        "ipv4",
        "ipv6",
        "national_id",
        "ssn",
        "unique_identifier",
    ]
    assert labels_where(persona_only_backends=frozenset({"pgm"})) == ["phone_number"]
    assert labels_where(apply_path="identify_only") == [
        "city",
        "date",
        "datetime",
        "duration",
        "state",
        "time",
        "zipcode",
    ]
    assert labels_where(requires_value_match=True) == [
        "credit_debit_card",
        "email",
        "ipv4",
        "ipv6",
        "phone_number",
        "ssn",
    ]
    assert labels_where(name_shape_gates=True) == [
        "first_name",
        "full_name",
        "last_name",
        "middle_name",
    ]

    # Role-strip lexicon: label segments + aliases; never role words from name_patterns.
    assert {"first", "name", "email", "phone", "dob", "primary"} <= entities.ROLE_STRIP_TOKENS
    assert "of" not in entities.ROLE_STRIP_TOKENS  # date_of_birth stopword
    assert "patient" not in entities.ROLE_STRIP_TOKENS
    assert "provider" not in entities.ROLE_STRIP_TOKENS
    assert "free" not in entities.ROLE_STRIP_TOKENS and "text" not in entities.ROLE_STRIP_TOKENS

    free_text = entities.spec("free_text")
    dob = entities.spec("date_of_birth")
    assert free_text is not None and free_text.transform_method == "propagation"
    assert dob is not None and dob.transform_method == "perturbation"
    assert entities.effective_apply_path("phone_number", "pgm") == "persona"
    assert entities.effective_apply_path("phone_number", "faker") == "standalone_map"
    assert entities.effective_apply_path("phone_number", "managed") == "standalone_map"
    assert entities.effective_apply_path("ssn", "managed") == "standalone_map"
    assert entities.is_identify_only("date")
    assert entities.is_identify_only("city")
    assert entities.is_identify_only("state")
    assert entities.is_identify_only("zipcode")

    assert {e.value for e in DATE_PATTERN_ENTITIES} == {"date_of_birth"}
    assert {e.value for e in PERSONA_PATTERN_ENTITIES} == {
        "email",
        "first_name",
        "full_name",
        "last_name",
        "middle_name",
    }
    assert {e.value for e in TEMPLATE_PATTERN_ENTITIES} == {
        "api_key",
        "credit_debit_card",
        "phone_number",
        "unique_identifier",
    }


def test_discovered_secondary_pattern_validates_against_same_evidence_sample():
    """Discovery and validate_plan must share the pattern evidence slice.

    A secondary ID format that appears only after the first 1,000 distinct values
    can still dominate a seeded row sample. Validation used to inspect
    ``unique()[:1000]`` only and reject that discovered pattern with ParameterError.
    """
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan, resolve_plan

    a_vals = [f"CUST-{i:08d}" for i in range(1000)]
    b_vals = [f"NEWID{i:08d}" for i in range(400)]
    df = pd.DataFrame({"record_id": a_vals + b_vals * 5})
    cfg = PiiReplacerConfig(
        replacement_plan=AUTO_DISCOVERY,
        person=PiiPersonConfig(backend=PiiPersonBackend.faker),
    )
    runtime = config_from_replace_pii(cfg)
    plan = discover_plan(df, None, runtime, cfg)
    spec = column_spec(plan.standalone_columns_to_replace, "record_id")
    assert spec is not None
    assert any(p.startswith("NEWID") for p in (spec.patterns or [])), spec.patterns
    # Must not raise: shared evidence includes the NEWID rows discovery used.
    validate_plan(df, plan, data_config=DataParameters())
    resolve_plan(cfg, df, data_config=DataParameters(), cfg=runtime)
