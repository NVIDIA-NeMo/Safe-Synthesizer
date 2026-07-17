# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.pii_replacement import (
    AUTO_DISCOVERY,
    PiiColumnPlan,
    PiiEntity,
    PiiPersona,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.defaults import NSS_MANAGED_ASSETS_PATH_ENV, default_managed_assets_path
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.discovery import _nss_free_text_columns
from nemo_safe_synthesizer.pii_replacer.plan import plan_to_runtime, unique_id_advisories, validate_plan
from nemo_safe_synthesizer.pii_replacer.replacer import (
    TabularPiiReplacer,
    _person_transform_method,
    _plan_column_counts,
    _replacement_plan_source,
)
from nemo_safe_synthesizer.pii_replacer.runtime_config import runtime_config_from_replace_pii
from nemo_safe_synthesizer.preflight import PreflightContext
from nemo_safe_synthesizer.preflight.checks.pii import PiiReplacementConfigCheck, PiiUniqueIdAdvisoryCheck


@pytest.fixture
def patient_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["A", "A", "B", "B", "B"],
            "first_name": ["Alice", "Alice", "Bob", "Bob", "Bob"],
            "provider_name": ["Dr X", "Dr Y", "Dr Z", "Dr Z", "Dr Z"],
            "notes": [
                "Alice visited Dr X",
                "Alice visited Dr Y",
                "Bob visited Dr Z",
                "Bob again Dr Z",
                "Bob follow-up Dr Z",
            ],
        }
    )


def test_replace_pii_defaults():
    cfg = SafeSynthesizerParameters()
    assert cfg.replace_pii is not None
    assert cfg.replace_pii.replacement_plan == AUTO_DISCOVERY
    assert cfg.replace_pii.person.backend == PiiPersonBackend.managed


def test_default_managed_assets_path():
    assert default_managed_assets_path() == Path.home() / ".data-designer" / "managed-assets"


def test_managed_assets_path_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv(NSS_MANAGED_ASSETS_PATH_ENV, str(tmp_path))
    assert default_managed_assets_path() == tmp_path
    assert PiiPersonConfig().resolved_managed_assets_path() == tmp_path


def test_managed_assets_path_explicit_overrides_env(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    monkeypatch.setenv(NSS_MANAGED_ASSETS_PATH_ENV, str(tmp_path / "from_env"))
    cfg = PiiPersonConfig(managed_assets_path=str(explicit))
    assert cfg.resolved_managed_assets_path() == explicit


def test_managed_backend_smoke(patient_df: pd.DataFrame):
    assets_root = default_managed_assets_path()
    parquet = assets_root / "datasets" / "en_US.parquet"
    if not parquet.exists():
        pytest.skip(f"managed persona assets not installed at {parquet}")

    plan = PiiReplacementPlan(
        group_key="patient_id",
        identified_personas={"patient": None},
        columns={
            "first_name": PiiColumnPlan(entity_type=PiiEntity.first_name, persona="patient"),
        },
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(
            replacement_plan=plan,
            replacement={"locale": "en_US"},
            person={"backend": "managed"},
        ),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(patient_df)
    assert replacer.result is not None
    out = replacer.result.transformed_df
    assert out["first_name"].tolist() != patient_df["first_name"].tolist()


def test_person_transform_method_labels():
    assert _person_transform_method("en_US", "managed") == "en_US personas"
    assert _person_transform_method("en_US", "pgm") == "en_US PGM"
    assert _person_transform_method("en_US", "faker") == "en_US Faker"


def test_column_statistics_transform_methods_and_entity_counts(patient_df: pd.DataFrame):
    plan = PiiReplacementPlan(
        group_key="patient_id",
        identified_personas={"patient": None},
        columns={
            "first_name": PiiColumnPlan(entity_type=PiiEntity.first_name, persona="patient"),
            "notes": PiiColumnPlan(entity_type=PiiEntity.free_text),
        },
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(
            replacement_plan=plan,
            replacement={"locale": "en_US"},
            person={"backend": "faker"},
        ),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(patient_df)
    assert replacer.result is not None
    first_name_stats = replacer.result.column_statistics["first_name"]
    notes_stats = replacer.result.column_statistics["notes"]
    assert first_name_stats.transform_methods == {"en_US Faker"}
    assert notes_stats.transform_methods == {"propagation"}
    assert "free_text" not in notes_stats.detected_entity_counts


def test_column_statistics_dob_perturbation_and_unique_id_methods():
    df = pd.DataFrame(
        {
            "patient_id": ["A", "B", "C", "D"],
            "date_of_birth": ["01/02/1980", "03/04/1975", "05/06/1990", "07/08/1965"],
            "record_id": ["550e8400-e29b-41d4-a716-446655440000", "abc123", "def456", "ghi789"],
        }
    )
    plan = PiiReplacementPlan(
        group_key="patient_id",
        columns={
            "date_of_birth": PiiColumnPlan(
                entity_type=PiiEntity.date_of_birth,
                pattern="%m/%d/%Y",
                dominant_pattern_coverage=100.0,
            ),
            "record_id": PiiColumnPlan(entity_type=PiiEntity.unique_identifier),
        },
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(
            replacement_plan=plan,
            replacement={"locale": "en_US"},
            person={"backend": "faker"},
        ),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(df)
    assert replacer.result is not None
    dob_stats = replacer.result.column_statistics["date_of_birth"]
    record_id_stats = replacer.result.column_statistics["record_id"]
    assert dob_stats.transform_methods == {"perturbation"}
    # No inferred pattern on the unique_identifier spec -> Faker fallback.
    assert record_id_stats.transform_methods == {"Faker"}


def test_faker_persona_names_conditioned_on_sex():
    """Regression: the persona engine must condition synthetic names on the source
    ``sex`` demographic. A bug dropped the (sex, ethnicity) bucket signature so names
    were drawn with a random gender, producing e.g. a female name for a Male row."""
    from nemo_safe_synthesizer.pii_replacer.persona import PersonaEngine
    from nemo_safe_synthesizer.pii_replacer.replacement import extract_instances

    n = 40
    df = pd.DataFrame(
        {
            "Name": [f"Person {i}" for i in range(n)],
            "Gender": ["Male" if i % 2 else "Female" for i in range(n)],
        }
    )
    runtime_plan = {
        "group_key": None,
        "roles": [
            {
                "role": "primary_person",
                "fields": {"full_name": "Name"},
                "field_meta": {},
                "demographics": {"sex": "Gender", "race": None},
            }
        ],
        "non_person": [],
        "free_text_columns": [],
    }
    runtime = runtime_config_from_replace_pii(
        ReplacePiiConfig(person={"backend": "faker"}, replacement={"locale": "en_US"})
    )
    from nemo_safe_synthesizer.pii_replacer.replacement import _core_config

    instances = extract_instances(df, runtime_plan, _core_config(runtime))
    assert len(instances) == n
    engine = PersonaEngine(runtime, len(instances))
    assert engine.backend == "faker"
    engine.assign(instances)

    # Every instance carries the source sex, and its sampled persona matches it.
    assert all(inst["sex"] in ("Male", "Female") for inst in instances)
    assert all(inst["persona"]["sex"] == inst["sex"] for inst in instances)


def test_plan_column_counts():
    plan = PiiReplacementPlan(
        identified_personas={"patient": None},
        columns={
            "first_name": PiiColumnPlan(entity_type=PiiEntity.first_name, persona="patient"),
            "last_name": PiiColumnPlan(entity_type=PiiEntity.last_name, persona="patient"),
            "notes": PiiColumnPlan(entity_type=PiiEntity.free_text),
        },
    )
    assert _plan_column_counts(plan) == (1, 2, 1)


def test_replacement_plan_source():
    assert _replacement_plan_source(ReplacePiiConfig()) == "auto_discovery"
    assert _replacement_plan_source(ReplacePiiConfig(replacement_plan="/tmp/plan.json")) == "/tmp/plan.json"


def test_pii_replacement_logs_group_key_warning(patient_df: pd.DataFrame, caplog):
    import logging

    caplog.set_level(logging.INFO)
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=AUTO_DISCOVERY, person={"backend": "faker"}),
        data_config=DataParameters(),
    )
    replacer.transform_df(patient_df)

    messages = [record.getMessage() for record in caplog.records]
    assert any("Starting tabular PII replacement" in message for message in messages)
    assert any("Resolved PII replacement plan" in message for message in messages)
    assert any("no group_key" in message for message in messages)


def test_replace_pii_null_disables():
    cfg = SafeSynthesizerParameters.model_validate({"replace_pii": None})
    assert cfg.replace_pii is None


def test_analyze_column_patterns_date_dominant():
    from nemo_safe_synthesizer.pii_replacer.core import Config, analyze_column_patterns

    series = pd.Series(["04/17/2023"] * 95 + ["08/2010"] * 4 + ["unknown"] * 1)
    analysis = analyze_column_patterns(series, Config())
    assert analysis["entity"] == "date"
    assert analysis["pattern"] == "%m/%d/%Y"
    assert analysis["coverage"] == 95.0
    assert analysis["structured"] is True


def test_analyze_column_patterns_below_threshold_not_structured():
    from nemo_safe_synthesizer.pii_replacer.core import Config, analyze_column_patterns

    series = pd.Series(["04/17/2023"] * 70 + ["08/2010"] * 30)
    analysis = analyze_column_patterns(series, Config())
    assert analysis["coverage"] == 70.0
    assert analysis["structured"] is False


def test_discover_event_date_identified_not_replaced():
    from nemo_safe_synthesizer.pii_replacer.discovery import discover_plan

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
        runtime=runtime_config_from_replace_pii(ReplacePiiConfig()),
        config=ReplacePiiConfig(),
    )
    # A generic date column is identified as structured only to keep it out of the
    # free-text path; it is excluded from the replacement plan entirely.
    assert "event_date" not in plan.columns


def test_discovery_logs_temporal_and_free_text_gates(caplog):
    import logging

    from nemo_safe_synthesizer.pii_replacer.discovery import discover_plan

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
        runtime=runtime_config_from_replace_pii(ReplacePiiConfig()),
        config=ReplacePiiConfig(),
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any("Identified temporal column 'event_date'" in message for message in messages)
    assert any("not scanned as free text for PII detection" in message and "weight" in message for message in messages)
    assert any("Column 'notes' scanned as free text for PII detection" in message for message in messages)


def test_mvp_skips_free_text_scan_without_structured_columns(caplog):
    import logging

    from nemo_safe_synthesizer.pii_replacer.discovery import discover_plan

    caplog.set_level(logging.INFO)
    # No person columns and no replaceable non-person columns: only a generic
    # (identify-only) date and a free-text column. In MVP mode there is nothing
    # to propagate, so the free-text column must not be scanned/planned.
    dominant_dates = [f"04/{(i % 28) + 1:02d}/2023" for i in range(100)]
    df = pd.DataFrame(
        {
            "event_date": dominant_dates,
            "notes": [f"Patient record {i} visited clinic for follow up care today" for i in range(100)],
        }
    )
    plan = discover_plan(
        df,
        group_key=None,
        runtime=runtime_config_from_replace_pii(ReplacePiiConfig()),
        config=ReplacePiiConfig(),
    )
    assert "notes" not in plan.columns
    messages = [record.getMessage() for record in caplog.records]
    assert any("skipping free-text scan" in message for message in messages)


def test_llm_mode_still_scans_free_text_without_structured_columns():
    from nemo_safe_synthesizer.pii_replacer.discovery import discover_plan

    dominant_dates = [f"04/{(i % 28) + 1:02d}/2023" for i in range(100)]
    df = pd.DataFrame(
        {
            "event_date": dominant_dates,
            "notes": [f"Patient record {i} visited clinic for follow up care today" for i in range(100)],
        }
    )
    plan = discover_plan(
        df,
        group_key=None,
        runtime=runtime_config_from_replace_pii(ReplacePiiConfig(llm_enhancement=True)),
        config=ReplacePiiConfig(llm_enhancement=True),
    )
    notes = plan.columns.get("notes")
    assert notes is not None and notes.entity_type == PiiEntity.free_text


def test_date_entity_not_replaced_passthrough():
    from nemo_safe_synthesizer.pii_replacer.replacement import run_replacement
    from nemo_safe_synthesizer.pii_replacer.runtime_config import RuntimeConfig

    df = pd.DataFrame({"event_date": ["04/17/2023", "08/2010", "unknown date"]})
    runtime_plan = {
        "group_key": None,
        "roles": [],
        "non_person": [
            {
                "column": "event_date",
                "entity": "date",
                "pattern": "%m/%d/%Y",
                "dominant_pattern_coverage": 66.7,
            }
        ],
        "free_text_columns": [],
    }
    runtime = RuntimeConfig(
        locale="en_US",
        random_seed=42,
        replace_group_key=True,
        persona_backend="faker",
        sdg_pgms_src="/tmp",
        managed_assets_path=None,
    )
    out, details = run_replacement(df, runtime_plan, runtime)
    # Generic dates are never replaced: every value passes through unchanged.
    assert out["event_date"].tolist() == ["04/17/2023", "08/2010", "unknown date"]
    assert not details["changed_summary"]


def test_free_text_excludes_non_object_and_structured_field_types():
    from nemo_safe_synthesizer.pii_replacer.discovery import _nss_free_text_columns

    df = pd.DataFrame(
        {
            "weight": [135.0, 165.0, 142.0],
            "is_active": [True, False, True],
            "event_type": ["Admission", "Admission", "Discharge"],
            "notes": [
                "Patient visited clinic for follow up care today",
                "Another long clinical note about symptoms and treatment",
                "Third detailed note with multiple words in the sentence",
            ],
        }
    )
    free_text = _nss_free_text_columns(df, exclude=set())
    assert free_text == ["notes"]


def test_analyze_column_patterns_datetime_dominant():
    from nemo_safe_synthesizer.pii_replacer.core import Config, analyze_column_patterns

    series = pd.Series(["2023-04-17 14:30:00"] * 95 + ["2023-05-01 09:00:00"] * 5)
    analysis = analyze_column_patterns(series, Config())
    assert analysis["entity"] == "datetime"
    assert analysis["pattern"] == "%Y-%m-%d %H:%M:%S"
    assert analysis["coverage"] == 100.0
    assert analysis["structured"] is True


def test_analyze_column_patterns_time_dominant():
    from nemo_safe_synthesizer.pii_replacer.core import Config, analyze_column_patterns

    series = pd.Series(["14:30:00"] * 90 + ["09:15:00"] * 10)
    analysis = analyze_column_patterns(series, Config())
    assert analysis["entity"] == "time"
    assert analysis["pattern"] == "%H:%M:%S"
    assert analysis["coverage"] == 100.0
    assert analysis["structured"] is True


def test_analyze_column_patterns_duration_dominant():
    from nemo_safe_synthesizer.pii_replacer.core import Config, analyze_column_patterns

    series = pd.Series(["PT2H30M"] * 85 + ["45 min"] * 15)
    analysis = analyze_column_patterns(series, Config())
    assert analysis["entity"] == "duration"
    assert analysis["pattern"] == "iso8601"
    assert analysis["coverage"] == 85.0
    assert analysis["structured"] is True


def test_discover_temporal_columns_identified_not_replaced():
    from nemo_safe_synthesizer.pii_replacer.discovery import discover_plan

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
        runtime=runtime_config_from_replace_pii(ReplacePiiConfig()),
        config=ReplacePiiConfig(),
    )
    for col in ("created_at", "shift_start", "wait_time"):
        assert col not in plan.columns
    assert "notes" in plan.columns


def test_discover_date_of_birth_gets_pattern_and_coverage():
    from nemo_safe_synthesizer.pii_replacer.discovery import discover_plan

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
        runtime=runtime_config_from_replace_pii(ReplacePiiConfig()),
        config=ReplacePiiConfig(),
    )
    # Birth dates are replaced independently of any persona, so they are a plain
    # column with no persona reference (not associated with a person).
    dob_spec = plan.columns.get("date_of_birth")
    assert dob_spec is not None
    assert dob_spec.entity_type == PiiEntity.date_of_birth
    assert dob_spec.pattern == "%m/%d/%Y"
    assert dob_spec.dominant_pattern_coverage == 100.0
    assert dob_spec.persona is None


def test_grouped_unique_id_and_dob_replaced_globally_unique_and_group_consistent():
    """A group-key unique_identifier and a per-group birth date are replaced via
    the unassociated (non-person) path: globally unique for ids, consistent
    within each group, and age-preserving for DOB."""
    from nemo_safe_synthesizer.pii_replacer.discovery import discover_plan

    rows = []
    for p in range(30):
        for _ in range(3):
            rows.append(
                {
                    "patient_id": f"pmc-{6 if p % 2 else 8}{p:05d}-{(p % 4) + 1}",
                    "first_name": f"First{p}",
                    "date_of_birth": f"{(p % 12) + 1:02d}/{(p % 28) + 1:02d}/19{60 + (p % 30):02d}",
                    "sex": "Female" if p % 2 else "Male",
                }
            )
    df = pd.DataFrame(rows)
    config = ReplacePiiConfig(replacement_plan=AUTO_DISCOVERY, person={"backend": "faker"})
    data_config = DataParameters(group_training_examples_by="patient_id")
    plan = discover_plan(df, "patient_id", runtime_config_from_replace_pii(config), config)
    assert plan.columns["patient_id"].entity_type == PiiEntity.unique_identifier
    assert plan.columns["patient_id"].persona is None
    assert plan.columns["date_of_birth"].entity_type == PiiEntity.date_of_birth
    assert plan.columns["date_of_birth"].persona is None

    replacer = TabularPiiReplacer(config, data_config=data_config)
    replacer.transform_df(df)
    out = replacer.result.transformed_df

    # patient_id fully transformed, consistent within each group, and globally
    # unique (identifiers get a hard cross-group uniqueness guarantee).
    assert (out["patient_id"].values != df["patient_id"].values).all()
    orig_to_new: dict[str, str] = {}
    for o, n in zip(df["patient_id"], out["patient_id"]):
        orig_to_new.setdefault(o, n)
        assert orig_to_new[o] == n  # same original -> same synthetic within its group
    synth_ids = list(orig_to_new.values())
    assert len(set(synth_ids)) == len(synth_ids)  # no cross-group collisions
    assert not (set(synth_ids) & set(df["patient_id"]))  # never reuse a real id

    # DOB perturbed (changed) and consistent within each patient group.
    assert (out["date_of_birth"].values != df["date_of_birth"].values).all()
    for _pid, g in out.groupby(df["patient_id"].values):
        assert g["date_of_birth"].nunique() == 1


def test_per_group_unique_id_detected_when_group_key_set():
    """A per-group identifier repeats across every row of its group, so its
    per-row unique_ratio is far below the id_unique_ratio gate. When group_key
    is set, cardinality must be measured against groups (deduped) so the column
    is recognized as a unique_identifier rather than misread as free text."""
    from nemo_safe_synthesizer.pii_replacer.discovery import _core_config, _detect_full_dataframe

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
    cfg = _core_config(runtime_config_from_replace_pii(ReplacePiiConfig()))

    # Without a group key the per-group ids look low-cardinality and are missed.
    row_scoped = _detect_full_dataframe(df, cfg, group_key=None)
    row_entities = {e["column"]: e["entity"] for e in row_scoped["non_person"]}
    assert row_entities.get("record_id") != "unique_identifier"

    # With the group key they are correctly detected with patterns.
    grouped = _detect_full_dataframe(df, cfg, group_key="patient_id")
    grouped_entities = {e["column"]: e for e in grouped["non_person"]}
    for col in ("patient_id", "record_id", "event_id"):
        assert grouped_entities[col]["entity"] == "unique_identifier"
        assert grouped_entities[col]["pattern"]


def test_entity_driven_column_under_persona_routes_to_non_person():
    """Placement is irrelevant for entity-driven columns: a date_of_birth (or
    unique_identifier) associated with a persona is still routed through the
    non-person path, carrying its pattern/coverage, and round-trips as an
    unassociated column."""
    plan = PiiReplacementPlan(
        identified_personas={"primary_person": None},
        columns={
            "first_name": PiiColumnPlan(entity_type=PiiEntity.first_name, persona="primary_person"),
            "date_of_birth": PiiColumnPlan(
                entity_type=PiiEntity.date_of_birth,
                persona="primary_person",
                pattern="%m/%d/%Y",
                dominant_pattern_coverage=100.0,
            ),
        },
    )
    runtime = plan_to_runtime(plan)
    # DOB is not a persona field; it is emitted as a non-person entity.
    assert runtime["roles"][0]["fields"] == {"first_name": "first_name"}
    dob_ent = next(e for e in runtime["non_person"] if e["column"] == "date_of_birth")
    assert dob_ent["entity"] == "date_of_birth"
    assert dob_ent["pattern"] == "%m/%d/%Y"
    assert dob_ent["dominant_pattern_coverage"] == 100.0

    from nemo_safe_synthesizer.pii_replacer.plan import runtime_plan_to_pii_plan

    round_tripped = runtime_plan_to_pii_plan(runtime, group_key=None)
    dob = round_tripped.columns["date_of_birth"]
    assert dob.pattern == "%m/%d/%Y"
    assert dob.dominant_pattern_coverage == 100.0
    # DOB round-trips as a plain column with no persona (never persona-associated).
    assert round_tripped.columns["date_of_birth"].persona is None


def _dob_replacement(coverage: float, dates: list[str]) -> pd.Series:
    from nemo_safe_synthesizer.pii_replacer.replacement import run_replacement
    from nemo_safe_synthesizer.pii_replacer.runtime_config import RuntimeConfig

    df = pd.DataFrame({"date_of_birth": dates, "first_name": [f"First{i}" for i in range(len(dates))]})
    runtime_plan = {
        "group_key": None,
        "roles": [
            {
                "role": "primary_person",
                "fields": {"first_name": "first_name"},
                "field_meta": {},
                "demographics": {"sex": None, "race": None},
            }
        ],
        "non_person": [
            {"column": "date_of_birth", "entity": "date_of_birth", "pattern": "%m/%d/%Y", "dominant_pattern_coverage": coverage}
        ],
        "free_text_columns": [],
    }
    runtime = RuntimeConfig(
        locale="en_US",
        random_seed=7,
        replace_group_key=True,
        persona_backend="faker",
        sdg_pgms_src="/tmp",
        managed_assets_path=None,
    )
    out, _ = run_replacement(df, runtime_plan, runtime)
    return out["date_of_birth"]


def test_dob_replacement_whole_column_when_full_coverage():
    dates = [f"{(i % 12) + 1:02d}/{(i % 28) + 1:02d}/1980" for i in range(10)]
    result = _dob_replacement(100.0, dates)
    for original, new in zip(dates, result):
        assert new != original
        # Whole-column replacement keeps the dominant %m/%d/%Y format.
        assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(new))


def test_dob_replacement_per_value_preserves_minority_format():
    # Dominant %m/%d/%Y but one minority %Y-%m-%d row; coverage < 100 -> per-value matching.
    dates = ["01/15/1980", "02/20/1990", "1975-03-25"]
    result = _dob_replacement(90.0, dates)
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(result.iloc[0]))
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(result.iloc[1]))
    # The minority ISO row is re-formatted in its own format, not the dominant one.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(result.iloc[2]))


def test_conditioning_only_sex_and_race_in_plan():
    plan = PiiReplacementPlan(
        identified_personas={
            "primary_person": PiiPersona(gender="sex", ethnic_background="race"),
        },
        columns={
            "first_name": PiiColumnPlan(entity_type=PiiEntity.first_name, persona="primary_person"),
        },
    )
    runtime = plan_to_runtime(plan)
    demo = runtime["roles"][0]["demographics"]
    assert set(demo.keys()) == {"sex", "race"}
    assert demo["sex"] == "sex"
    assert demo["race"] == "race"


def test_validate_plan_group_key_mismatch(patient_df: pd.DataFrame):
    plan = PiiReplacementPlan(group_key="patient_id")
    with pytest.raises(ParameterError, match="must match"):
        validate_plan(patient_df, plan, data_config=DataParameters())


def test_grouped_replacement_distinct_doctors_within_group(patient_df: pd.DataFrame):
    plan = PiiReplacementPlan(
        group_key="patient_id",
        identified_personas={"patient": None, "doctor": None},
        columns={
            "first_name": PiiColumnPlan(entity_type=PiiEntity.first_name, persona="patient"),
            "provider_name": PiiColumnPlan(entity_type=PiiEntity.full_name, persona="doctor"),
            "notes": PiiColumnPlan(entity_type=PiiEntity.free_text),
        },
    )
    validate_plan(patient_df, plan, data_config=DataParameters(group_training_examples_by="patient_id"))
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=plan, person={"backend": "faker"}),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(patient_df)
    assert replacer.result is not None
    out = replacer.result.transformed_df

    a_names = set(out.loc[out["patient_id"] == "A", "first_name"])
    assert len(a_names) == 1
    assert "Alice" not in a_names

    a_providers = set(out.loc[out["patient_id"] == "A", "provider_name"])
    assert len(a_providers) == 2
    assert "Dr X" not in a_providers and "Dr Y" not in a_providers

    b_providers = set(out.loc[out["patient_id"] == "B", "provider_name"])
    assert len(b_providers) == 1
    assert "Dr Z" not in b_providers


def test_record_scoped_replacement_changes_per_row():
    df = pd.DataFrame(
        {
            "first_name": ["Alice", "Alice"],
            "provider_name": ["Dr Z", "Dr Z"],
        }
    )
    plan = PiiReplacementPlan(
        identified_personas={"person": None},
        columns={
            "first_name": PiiColumnPlan(entity_type=PiiEntity.first_name, persona="person"),
            "provider_name": PiiColumnPlan(entity_type=PiiEntity.full_name, persona="person"),
        },
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=plan, person={"backend": "faker"}),
        data_config=DataParameters(),
    )
    replacer.transform_df(df)
    assert replacer.result is not None
    names = replacer.result.transformed_df["first_name"].tolist()
    providers = replacer.result.transformed_df["provider_name"].tolist()
    assert names[0] != "Alice" and names[1] != "Alice"
    assert names[0] != names[1]
    assert providers[0] != "Dr Z" and providers[1] != "Dr Z"
    assert providers[0] != providers[1]


def test_auto_discovery_emits_plan_shape():
    n = 30
    # A high-cardinality person column ensures structured detection fires, so the
    # free-text column is scanned and planned (in MVP mode free text is only
    # planned when there is structured PII to propagate).
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
            "notes": [f"Patient record {i} visited the clinic for follow up care today" for i in range(n)],
        }
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=AUTO_DISCOVERY, person={"backend": "faker"}),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(df)
    assert replacer.resolved_plan is not None
    assert replacer.resolved_plan.group_key == "patient_id"
    notes = replacer.resolved_plan.columns.get("notes")
    assert notes is not None
    assert notes.entity_type == PiiEntity.free_text


def test_nss_free_text_detection_matches_describe_field():
    df = pd.DataFrame(
        {
            "code": list(range(20)),
            "tag": [f"tag{i:02d}" for i in range(20)],
            "notes": [f"Patient record {i} visited clinic for follow up care and discussion today" for i in range(20)],
        }
    )
    assert set(_nss_free_text_columns(df, set())) == {"notes", "tag"}


def test_plan_to_runtime_maps_unique_id_entity():
    plan = PiiReplacementPlan(
        columns={
            "event_id": PiiColumnPlan(entity_type=PiiEntity.unique_identifier),
        }
    )
    runtime = plan_to_runtime(plan)
    assert runtime["non_person"][0]["entity"] == "unique_identifier"


def test_llm_enhancement_not_implemented(patient_df: pd.DataFrame):
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(llm_enhancement=True, person={"backend": "faker"}),
        data_config=DataParameters(),
    )
    with pytest.raises(NotImplementedError):
        replacer.transform_df(patient_df)


def test_replacement_plan_artifact(tmp_path, patient_df: pd.DataFrame):
    import yaml

    from nemo_safe_synthesizer.pii_replacer.plan import PII_REPLACEMENT_PLAN_FILENAME, load_plan_from_path

    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=AUTO_DISCOVERY, person={"backend": "faker"}),
        data_config=DataParameters(group_training_examples_by="patient_id"),
        workdir=tmp_path,
    )
    replacer.transform_df(patient_df)
    plan_file = tmp_path / PII_REPLACEMENT_PLAN_FILENAME
    assert plan_file.exists()
    plan_data = yaml.safe_load(plan_file.read_text())

    def _contains_null(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, dict):
            return any(_contains_null(v) for v in value.values())
        if isinstance(value, list):
            return any(_contains_null(v) for v in value)
        return False

    assert not _contains_null(plan_data)
    load_plan_from_path(str(plan_file))


def test_unique_id_advisory_for_user_plan(patient_df: pd.DataFrame):
    plan = PiiReplacementPlan(
        columns={
            "patient_id": PiiColumnPlan(entity_type=PiiEntity.unique_identifier),
        }
    )
    warnings = unique_id_advisories(patient_df, plan, runtime_config_from_replace_pii(ReplacePiiConfig()))
    assert warnings


def test_preflight_faker_locale_error():
    config = SafeSynthesizerParameters(
        replace_pii=ReplacePiiConfig(
            replacement={"locale": "not_a_real_locale"},
            person={"backend": "faker"},
        )
    )
    issues = PiiReplacementConfigCheck().run(PreflightContext(config=config, data=pd.DataFrame(), metadata=MagicMock()))
    assert any(i.code == "pii_faker_locale_invalid" for i in issues)


def test_preflight_unique_id_advisory():
    df = pd.DataFrame({"category": ["A", "B", "C", "A", "B"]})
    plan = PiiReplacementPlan(
        columns={
            "category": PiiColumnPlan(entity_type=PiiEntity.unique_identifier),
        }
    )
    config = SafeSynthesizerParameters(
        replace_pii=ReplacePiiConfig(replacement_plan=plan),
    )
    issues = PiiUniqueIdAdvisoryCheck().run(PreflightContext(config=config, data=df, metadata=MagicMock()))
    assert any(i.code == "pii_unique_id_low_cardinality" for i in issues)


def test_plan_yaml_round_trip_new_shape(tmp_path):
    from nemo_safe_synthesizer.pii_replacer.plan import load_plan_from_path, save_plan_to_path

    plan = PiiReplacementPlan(
        group_key="patient_id",
        identified_personas={
            "patient": PiiPersona(gender="gender", ethnic_background="race"),
            "doctor": None,
            "emergency_contact": None,
        },
        columns={
            "first_name": PiiColumnPlan(entity_type=PiiEntity.first_name, persona="patient"),
            "provider_name": PiiColumnPlan(entity_type=PiiEntity.full_name, persona="doctor"),
            "date_of_birth": PiiColumnPlan(entity_type=PiiEntity.date_of_birth, pattern="%Y-%m-%d"),
            "patient_id": PiiColumnPlan(entity_type=PiiEntity.unique_identifier),
            "notes": PiiColumnPlan(entity_type=PiiEntity.free_text),
        },
    )
    path = tmp_path / "plan.yaml"
    save_plan_to_path(plan, path)
    loaded = load_plan_from_path(str(path))
    assert loaded.group_key == "patient_id"
    assert loaded.identified_personas["doctor"] is None
    assert loaded.columns["first_name"].persona == "patient"
    assert loaded.columns["date_of_birth"].persona is None


def test_personaless_dob_and_ssn_use_entity_correct_replacement():
    df = pd.DataFrame(
        {
            "date_of_birth": ["01/02/1980", "03/04/1975"],
            "ssn": ["123-45-6789", "987-65-4321"],
        }
    )
    plan = PiiReplacementPlan(
        columns={
            "date_of_birth": PiiColumnPlan(
                entity_type=PiiEntity.date_of_birth,
                pattern="%m/%d/%Y",
                dominant_pattern_coverage=100.0,
            ),
            "ssn": PiiColumnPlan(entity_type=PiiEntity.ssn),
        },
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=plan, person={"backend": "faker"}),
        data_config=DataParameters(),
    )
    replacer.transform_df(df)
    assert replacer.result is not None
    out = replacer.result.transformed_df
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(out["date_of_birth"].iloc[0]))
    assert out["date_of_birth"].iloc[0] != df["date_of_birth"].iloc[0]
    assert out["ssn"].iloc[0] != df["ssn"].iloc[0]
