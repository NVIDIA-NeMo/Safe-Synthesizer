# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replacement scope: record / group / dataframe behavior and protected columns."""

from __future__ import annotations

import re
from typing import cast

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    AUTO_DISCOVERY,
    PersonaColumnSet,
    PiiColumnPlan,
    PiiEntity,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    PiiReplacementScope,
    PiiReplacerConfig,
)
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.entities import Config, config_from_replace_pii
from nemo_safe_synthesizer.pii_replacer.models import ScopedValueMap
from nemo_safe_synthesizer.pii_replacer.planning import (
    resolve_plan,
    validate_plan,
)
from nemo_safe_synthesizer.pii_replacer.replacer import (
    TabularPiiReplacer,
)
from tests.pii_replacer.helpers import column_spec


def test_pii_replacement_logs_scope_warning_when_group_key_set(fixture_patient_df: pd.DataFrame, caplog):
    import logging

    caplog.set_level(logging.INFO)
    plan = PiiReplacementPlan(
        scope=PiiReplacementScope.dataframe,
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                ],
            ),
        ],
    )
    replacer = TabularPiiReplacer(
        PiiReplacerConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(fixture_patient_df)

    messages = [record.getMessage() for record in caplog.records]
    assert any("Starting tabular PII replacement" in message for message in messages)
    assert any("Resolved PII replacement plan" in message for message in messages)
    assert any("persona consistency follows scope" in message for message in messages)


def test_grouped_unique_id_and_dob_replaced_globally_unique_and_group_consistent():
    """A group-key unique_identifier and a per-group birth date are replaced via
    the standalone path: globally unique for ids, consistent
    within each group, and age-preserving for DOB.
    """
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

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
    config = PiiReplacerConfig(replacement_plan=AUTO_DISCOVERY, person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    data_config = DataParameters(group_training_examples_by="patient_id")
    plan = discover_plan(df, "patient_id", config_from_replace_pii(config), config)
    patient_id_spec = column_spec(plan.standalone_columns_to_replace, "patient_id")
    dob_spec = column_spec(plan.standalone_columns_to_replace, "date_of_birth")
    assert patient_id_spec is not None
    assert dob_spec is not None
    assert patient_id_spec.entity_type == PiiEntity.unique_identifier
    assert dob_spec.entity_type == PiiEntity.date_of_birth

    replacer = TabularPiiReplacer(config, data_config=data_config)
    replacer.transform_df(df)
    assert replacer.result is not None
    out = replacer.result.transformed_df

    # patient_id fully transformed, consistent within each group, and globally
    # unique (identifiers get a hard cross-group uniqueness guarantee).
    assert (out["patient_id"] != df["patient_id"]).all()
    orig_to_new: dict[str, str] = {}
    for o, n in zip(df["patient_id"], out["patient_id"]):
        orig_to_new.setdefault(o, n)
        assert orig_to_new[o] == n  # same original -> same synthetic within its group
    synth_ids = list(orig_to_new.values())
    assert len(set(synth_ids)) == len(synth_ids)  # no cross-group collisions
    assert not (set(synth_ids) & set(df["patient_id"]))  # never reuse a real id

    # DOB perturbed (changed) and consistent within each patient group.
    assert (out["date_of_birth"] != df["date_of_birth"]).all()
    for _pid, g in out.groupby(df["patient_id"]):
        assert g["date_of_birth"].nunique() == 1


def test_entity_driven_column_under_persona_routes_to_standalone():
    """Placement is irrelevant for entity-driven columns: a date_of_birth listed
    under a persona is still replaced on the standalone path, keeping its patterns.
    """
    from nemo_safe_synthesizer.pii_replacer.replacement import build_standalone_maps, extract_instances, run_replacement

    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="primary_person",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                    PiiColumnPlan(
                        column_name="date_of_birth",
                        entity_type=PiiEntity.date_of_birth,
                        patterns=["%m/%d/%Y"],
                    ),
                ],
            )
        ]
    )
    cfg = Config(
        locale="en_US",
        random_seed=7,
        persona_backend="faker",
        sdg_pgms_src="/tmp",
        managed_assets_path=None,
    )
    df = pd.DataFrame(
        {
            "first_name": ["Alice", "Bob"],
            "date_of_birth": ["01/15/1980", "02/20/1990"],
        }
    )
    # Persona path sees only first_name; DOB is built as a standalone map.
    instances = extract_instances(df, plan, cfg)
    assert all("date_of_birth" not in inst.field_cols for inst in instances)
    assert all("first_name" in inst.field_cols for inst in instances)
    maps = build_standalone_maps(df, plan, cfg)
    assert "date_of_birth" in maps
    out, _ = run_replacement(df, plan, cfg)
    for original, new in zip(df["date_of_birth"], out["date_of_birth"]):
        assert new != original
        assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(new))


def test_grouped_replacement_distinct_doctors_within_group(fixture_patient_df: pd.DataFrame):
    plan = PiiReplacementPlan(
        scope=PiiReplacementScope.group,
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                ],
            ),
            PersonaColumnSet(
                persona="doctor",
                columns_to_replace=[
                    PiiColumnPlan(column_name="provider_name", entity_type=PiiEntity.full_name),
                ],
            ),
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
        ],
    )
    validate_plan(fixture_patient_df, plan, data_config=DataParameters(group_training_examples_by="patient_id"))
    replacer = TabularPiiReplacer(
        PiiReplacerConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(fixture_patient_df)
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
        scope=PiiReplacementScope.record,
        persona_backed_columns=[
            PersonaColumnSet(
                persona="person",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                    PiiColumnPlan(column_name="provider_name", entity_type=PiiEntity.full_name),
                ],
            )
        ],
    )
    replacer = TabularPiiReplacer(
        PiiReplacerConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
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


def test_unique_id_entity_drives_standalone_replacement():
    from nemo_safe_synthesizer.pii_replacer.replacement import build_standalone_maps

    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="event_id", entity_type=PiiEntity.unique_identifier),
        ]
    )
    spec = plan.standalone_columns_to_replace[0]
    assert spec.entity_type == PiiEntity.unique_identifier
    df = pd.DataFrame({"event_id": ["E001", "E002", "E003"]})
    cfg = Config(
        locale="en_US",
        random_seed=7,
        persona_backend="managed",
        sdg_pgms_src="/tmp",
        managed_assets_path=None,
    )
    maps = build_standalone_maps(df, plan, cfg)
    assert "event_id" in maps
    assert maps["event_id"].kind == "flat"
    assert set(maps["event_id"].data) == {"E001", "E002", "E003"}


def test_standalone_identifier_scope_record_group_dataframe():
    """Repeated phones/IDs honor plan.scope the same way DOB does."""
    from nemo_safe_synthesizer.pii_replacer.replacement import run_replacement

    cfg = Config(
        locale="en_US",
        random_seed=11,
        persona_backend="faker",
        sdg_pgms_src="/tmp",
        managed_assets_path=None,
    )
    phone = "555-0100"
    df = pd.DataFrame(
        {
            "group_id": ["G1", "G1", "G2", "G2"],
            "phone": [phone, phone, phone, phone],
            "event_id": ["E1", "E1", "E1", "E1"],
        }
    )
    standalone = [
        PiiColumnPlan(column_name="phone", entity_type=PiiEntity.phone_number),
        PiiColumnPlan(column_name="event_id", entity_type=PiiEntity.unique_identifier),
    ]

    # dataframe: one original -> one synthetic everywhere
    plan_df = PiiReplacementPlan(scope=PiiReplacementScope.dataframe, standalone_columns_to_replace=standalone)
    out_df, _ = run_replacement(df, plan_df, cfg)
    assert out_df["phone"].nunique() == 1
    assert out_df["phone"].iloc[0] != phone
    assert out_df["event_id"].nunique() == 1

    # record: same original can get independent synthetics per row
    plan_rec = PiiReplacementPlan(scope=PiiReplacementScope.record, standalone_columns_to_replace=standalone)
    out_rec, details_rec = run_replacement(df, plan_rec, cfg)
    standalone_rec = cast(dict[str, ScopedValueMap], details_rec["standalone_maps"])
    assert standalone_rec["phone"].kind == "record"
    assert out_rec["phone"].nunique() == 4
    assert (out_rec["phone"] != phone).all()
    assert out_rec["event_id"].nunique() == 4

    # group: shared within group, independent across groups; synthetics injective
    plan_grp = PiiReplacementPlan(scope=PiiReplacementScope.group, standalone_columns_to_replace=standalone)
    out_grp, details_grp = run_replacement(df, plan_grp, cfg, group_key="group_id")
    standalone_grp = cast(dict[str, ScopedValueMap], details_grp["standalone_maps"])
    assert standalone_grp["phone"].kind == "group"
    g1 = out_grp.loc[df["group_id"] == "G1", "phone"]
    g2 = out_grp.loc[df["group_id"] == "G2", "phone"]
    assert g1.nunique() == 1 and g2.nunique() == 1
    assert g1.iloc[0] != g2.iloc[0]
    assert g1.iloc[0] != phone and g2.iloc[0] != phone
    synths = list(out_grp["phone"].unique()) + list(out_grp["event_id"].unique())
    assert len(synths) == len(set(synths))


def test_record_scope_warns_on_large_frames(monkeypatch, caplog):
    import logging

    import nemo_safe_synthesizer.pii_replacer.replacement as replacement
    from nemo_safe_synthesizer.pii_replacer.replacement import standalone

    monkeypatch.setattr(standalone, "_RECORD_SCOPE_COST_WARN_ROWS", 3)
    caplog.set_level(logging.WARNING)
    cfg = Config(
        locale="en_US",
        random_seed=0,
        persona_backend="faker",
        sdg_pgms_src="/tmp",
        managed_assets_path=None,
    )
    df = pd.DataFrame({"event_id": [f"E{i}" for i in range(5)]})
    plan = PiiReplacementPlan(
        scope=PiiReplacementScope.record,
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="event_id", entity_type=PiiEntity.unique_identifier),
        ],
    )
    replacement.build_standalone_maps(df, plan, cfg)
    messages = [r.getMessage() for r in caplog.records]
    assert any("scope=record" in m and "can be costly" in m for m in messages)

    # Below the threshold: no cost warning.
    caplog.clear()
    replacement.build_standalone_maps(df.head(3), plan, cfg)
    assert not any("can be costly" in r.getMessage() for r in caplog.records)


def test_duplicate_dataframe_index_is_reset_for_record_scope():
    df = pd.DataFrame(
        {
            "first_name": ["Alice", "Bob"],
            "notes": ["Alice visited", "Bob visited"],
        },
        index=[7, 7],
    )
    plan = PiiReplacementPlan(
        scope=PiiReplacementScope.record,
        persona_backed_columns=[
            PersonaColumnSet(
                persona="person",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                ],
            )
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
        ],
    )
    replacer = TabularPiiReplacer(
        PiiReplacerConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(),
    )
    replacer.transform_df(df)
    assert replacer.result is not None
    out = replacer.result.transformed_df
    assert out.index.is_unique
    name0 = str(out.loc[0, "first_name"])
    name1 = str(out.loc[1, "first_name"])
    note0 = str(out.loc[0, "notes"])
    note1 = str(out.loc[1, "notes"])
    assert name0 in note0
    assert name1 in note1
    assert "Alice" not in note0 and "Bob" not in note1


def test_timeseries_does_not_replace_group_key_or_timestamp():
    from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters

    n = 30
    df = pd.DataFrame(
        {
            "patient_id": [f"pmc-6{i:05d}-1" for i in range(n)],
            "event_ts": [f"2024-01-{(i % 28) + 1:02d}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
        }
    )
    config = PiiReplacerConfig(
        replacement_plan=AUTO_DISCOVERY,
        person=PiiPersonConfig(backend=PiiPersonBackend.faker),
    )
    data_config = DataParameters(group_training_examples_by="patient_id")
    ts = TimeSeriesParameters(is_timeseries=True, timestamp_column="event_ts")
    plan = resolve_plan(
        config,
        df,
        data_config=data_config,
        cfg=config_from_replace_pii(config),
        time_series=ts,
    )
    assert column_spec(plan.standalone_columns_to_replace, "patient_id") is None
    assert column_spec(plan.standalone_columns_to_replace, "event_ts") is None
    for persona in plan.persona_backed_columns:
        assert column_spec(persona.columns_to_replace, "patient_id") is None
        assert column_spec(persona.columns_to_replace, "event_ts") is None


def test_order_by_column_is_not_replaced():

    n = 30
    df = pd.DataFrame(
        {
            "patient_id": [f"pmc-6{i:05d}-1" for i in range(n)],
            "seq_id": [f"S{i:05d}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
        }
    )
    config = PiiReplacerConfig(
        replacement_plan=AUTO_DISCOVERY,
        person=PiiPersonConfig(backend=PiiPersonBackend.faker),
    )
    data_config = DataParameters(
        group_training_examples_by="patient_id",
        order_training_examples_by="seq_id",
    )
    plan = resolve_plan(
        config,
        df,
        data_config=data_config,
        cfg=config_from_replace_pii(config),
    )
    # Non-timeseries: group key is still replaced when discovery classifies it as an ID.
    assert column_spec(plan.standalone_columns_to_replace, "patient_id") is not None
    assert column_spec(plan.standalone_columns_to_replace, "seq_id") is None
    for persona in plan.persona_backed_columns:
        assert column_spec(persona.columns_to_replace, "seq_id") is None


def test_user_plan_with_protected_column_is_rejected():
    from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
    from nemo_safe_synthesizer.pii_replacer.planning import resolve_plan

    n = 30
    df = pd.DataFrame(
        {
            "patient_id": [f"pmc-6{i:05d}-1" for i in range(n)],
            "event_ts": [f"2024-01-{(i % 28) + 1:02d}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
        }
    )
    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="patient_id", entity_type=PiiEntity.unique_identifier),
            PiiColumnPlan(column_name="event_ts", entity_type=PiiEntity.unique_identifier),
        ],
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name)],
            )
        ],
    )
    config = PiiReplacerConfig(
        replacement_plan=plan,
        person=PiiPersonConfig(backend=PiiPersonBackend.faker),
    )
    data_config = DataParameters(group_training_examples_by="patient_id")
    ts = TimeSeriesParameters(is_timeseries=True, timestamp_column="event_ts")
    with pytest.raises(ParameterError, match="must not replace structural columns"):
        resolve_plan(
            config,
            df,
            data_config=data_config,
            cfg=config_from_replace_pii(config),
            time_series=ts,
        )
