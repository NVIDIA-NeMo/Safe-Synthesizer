# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.pii_replacement import (
    PersonaColumnSet,
    PersonaMatchColumn,
    PiiColumnPlan,
    PiiEntity,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    PiiReplacementScope,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.planning import PII_REPLACEMENT_PLAN_FILENAME, plan_section_help
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from nemo_safe_synthesizer.tooling.pii_plan_preview import (
    build_diagram_model,
    map_yaml_source_ranges,
    plan_to_preview_yaml,
    preview_pii_replacement_plan,
    preview_state_from_yaml,
)

SAMPLE_YAML = """\
scope: group
persona_backed_columns:
  - persona: patient
    columns_to_replace:
      - column_name: first_name
        entity_type: first_name
      - column_name: last_name
        entity_type: last_name
    match_persona_by:
      - persona_attribute: gender
        column_name: sex
  - persona: provider
    columns_to_replace:
      - column_name: provider_name
        entity_type: full_name
standalone_columns_to_replace:
  - column_name: patient_id
    entity_type: unique_identifier
"""


@pytest.fixture
def patient_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2"],
            "first_name": ["Aisha", "Aisha", "Ben"],
            "last_name": ["Liang", "Liang", "Ortiz"],
            "sex": ["Female", "Female", "Male"],
            "provider_name": ["Dr. A", "Dr. A", "Dr. B"],
        }
    )


@pytest.fixture
def data_config() -> DataParameters:
    return DataParameters(group_training_examples_by="patient_id")


def test_map_yaml_source_ranges_covers_cards_and_rows():
    ranges = map_yaml_source_ranges(SAMPLE_YAML)
    assert "scope" in ranges
    assert "persona_backed_columns[0]" in ranges
    assert "persona_backed_columns[0].persona" in ranges
    assert "persona_backed_columns[0].columns_to_replace[1]" in ranges
    assert "persona_backed_columns[0].match_persona_by[0]" in ranges
    assert "persona_backed_columns[1]" in ranges
    assert "standalone_columns_to_replace[0]" in ranges

    first = ranges["persona_backed_columns[0].columns_to_replace[0]"]
    snippet = SAMPLE_YAML[first["start"] : first["end"]]
    assert "first_name" in snippet
    assert SAMPLE_YAML[ranges["scope"]["start"] : ranges["scope"]["end"]].startswith("scope:")


def test_map_yaml_source_ranges_preserves_user_text_and_handles_invalid():
    assert map_yaml_source_ranges("") == {}
    assert map_yaml_source_ranges(":\n  -") == {}


def test_build_diagram_model_cards():
    plan = PiiReplacementPlan.from_yaml_str(SAMPLE_YAML)
    diagram = build_diagram_model(plan)
    assert diagram["scope"] == "group"
    assert len(diagram["cards"]) == 3
    patient, provider, standalone = diagram["cards"]
    assert patient["kind"] == "persona"
    assert patient["title"] == "patient"
    assert [c["label"] for c in patient["compartments"]] == [
        "Columns to replace",
        "Match persona by",
    ]
    assert patient["help"] == plan_section_help("persona_backed_columns")
    assert "never replaced" in patient["compartments"][1]["hint"]
    assert diagram["scope_help"] == plan_section_help("scope")
    assert patient["compartments"][0]["rows"][0]["primary"] == "first_name"
    assert patient["compartments"][1]["rows"][0]["secondary"] == "gender"
    assert provider["title"] == "provider"
    assert standalone["kind"] == "standalone"
    assert standalone["title"] == "Standalone columns"
    assert standalone["help"] == plan_section_help("standalone_columns_to_replace")
    assert standalone["compartments"][0]["rows"][0]["primary"] == "patient_id"


def test_build_diagram_model_empty_sections():
    plan = PiiReplacementPlan(scope=PiiReplacementScope.dataframe)
    diagram = build_diagram_model(plan)
    assert diagram["cards"][0]["kind"] == "standalone"
    assert diagram["cards"][0]["compartments"][0]["rows"] == []


def test_preview_state_validates_against_dataframe(patient_df, data_config):
    good = preview_state_from_yaml(SAMPLE_YAML, df=patient_df, data_config=data_config)
    assert good.valid
    assert good.error is None
    assert good.warnings == []
    assert good.plan is not None

    missing_col = SAMPLE_YAML.replace(
        "column_name: first_name\n        entity_type: first_name",
        "column_name: not_a_column\n        entity_type: first_name",
        1,
    )
    bad = preview_state_from_yaml(missing_col, df=patient_df, data_config=data_config, previous=good)
    assert not bad.valid
    assert "not_a_column" in (bad.error or "")
    assert bad.warnings == []
    assert bad.diagram == good.diagram


def test_preview_state_rejects_protected_structural_columns(patient_df, data_config):
    """Edits that list order-by / time-series keys under columns_to_replace fail like resolve_plan."""
    from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters

    order_by_config = DataParameters(
        group_training_examples_by="patient_id",
        order_training_examples_by="patient_id",
    )
    yaml_text = """\
standalone_columns_to_replace:
  - column_name: patient_id
    entity_type: unique_identifier
"""
    state = preview_state_from_yaml(yaml_text, df=patient_df, data_config=order_by_config)
    assert not state.valid
    assert "structural columns" in (state.error or "")

    ts = TimeSeriesParameters(is_timeseries=True, timestamp_column="provider_name")
    ts_yaml = """\
standalone_columns_to_replace:
  - column_name: provider_name
    entity_type: unique_identifier
"""
    ts_state = preview_state_from_yaml(
        ts_yaml,
        df=patient_df,
        data_config=data_config,
        time_series=ts,
    )
    assert not ts_state.valid
    assert "structural columns" in (ts_state.error or "")


def test_preview_state_surfaces_section_placement_warnings(patient_df, data_config):
    yaml_text = """\
scope: group
persona_backed_columns:
  - persona: patient
    columns_to_replace:
      - column_name: patient_id
        entity_type: unique_identifier
standalone_columns_to_replace:
  - column_name: first_name
    entity_type: first_name
"""
    state = preview_state_from_yaml(yaml_text, df=patient_df, data_config=data_config)
    assert state.valid
    assert state.error is None
    assert len(state.warnings) == 2
    assert any("unique_identifier" in msg for msg in state.warnings)
    assert any("first_name" in msg for msg in state.warnings)


def test_preview_state_keeps_last_valid_diagram_across_repeated_errors(patient_df, data_config):
    good = preview_state_from_yaml(SAMPLE_YAML, df=patient_df, data_config=data_config)
    first_bad = preview_state_from_yaml(
        SAMPLE_YAML.replace("column_name: first_name", "column_name: not_a_column", 1),
        df=patient_df,
        data_config=data_config,
        previous=good,
    )
    second_bad = preview_state_from_yaml("scope: [oops]", df=patient_df, data_config=data_config, previous=first_bad)
    assert not second_bad.valid
    assert second_bad.diagram == good.diagram


def test_preview_state_invalid_without_previous_uses_empty_diagram(patient_df, data_config):
    state = preview_state_from_yaml("scope: [oops]", df=patient_df, data_config=data_config)
    assert not state.valid
    assert state.error
    assert state.diagram["cards"][0]["kind"] == "standalone"


def test_preview_state_explains_yaml_indentation_errors(patient_df, data_config):
    """Over-indented keys under a list item get a clearer hint than raw PyYAML."""
    bad = """\
scope: group
standalone_columns_to_replace:
  - column_name: Street
      entity_type: street_address
"""
    state = preview_state_from_yaml(bad, df=patient_df, data_config=data_config)
    assert not state.valid
    assert state.error is not None
    assert "same indentation" in state.error
    assert "mapping values are not allowed here" in state.error
    assert "entity_type" in state.error


def test_plan_to_preview_yaml_round_trip_omits_defaults(tmp_path: Path, patient_df):
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name)],
                match_persona_by=[PersonaMatchColumn(persona_attribute="gender", column_name="sex")],
            )
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="patient_id", entity_type=PiiEntity.unique_identifier),
        ],
    )
    text = plan_to_preview_yaml(plan)
    data = yaml.safe_load(text)
    assert "scope" not in data  # default dataframe omitted
    assert data["persona_backed_columns"][0]["persona"] == "patient"
    path = tmp_path / "plan.yaml"
    path.write_text(text)
    assert preview_state_from_yaml(
        path.read_text(),
        df=patient_df,
        data_config=DataParameters(),
    ).valid


def _builder(patient_df: pd.DataFrame, save_path: Path) -> SafeSynthesizer:
    return (
        SafeSynthesizer(save_path=save_path)
        .with_data_source(patient_df)
        .with_data(group_training_examples_by="patient_id")
        .with_replace_pii(
            ReplacePiiConfig(
                replacement_plan=PiiReplacementPlan.from_yaml_str(SAMPLE_YAML),
                person=PiiPersonConfig(backend=PiiPersonBackend.faker),
            )
        )
    )


def test_preview_helper_syncs_plan_and_builder_integration(patient_df, data_config, tmp_path):
    pytest.importorskip("anywidget")

    synced: list[PiiReplacementPlan] = []
    widget = preview_pii_replacement_plan(
        SAMPLE_YAML,
        df=patient_df,
        data_config=data_config,
        on_plan=synced.append,
    )
    assert widget.diagram["cards"][0]["title"] == "patient"
    assert widget.error == ""
    assert list(widget.warnings) == []
    assert synced and synced[0].persona_backed_columns[0].persona == "patient"
    assert widget.current_plan is synced[0]

    widget.yaml_text = SAMPLE_YAML.replace(
        "column_name: first_name\n        entity_type: first_name",
        "column_name: not_a_column\n        entity_type: first_name",
        1,
    )
    state = widget.refresh()
    assert not state.valid
    assert "not_a_column" in widget.error
    assert list(widget.warnings) == []
    assert len(synced) == 1  # invalid render does not sync

    misplaced = """\
scope: group
persona_backed_columns:
  - persona: patient
    columns_to_replace:
      - column_name: patient_id
        entity_type: unique_identifier
standalone_columns_to_replace:
  - column_name: first_name
    entity_type: first_name
  - column_name: last_name
    entity_type: last_name
  - column_name: provider_name
    entity_type: full_name
"""
    widget.yaml_text = misplaced
    state = widget.refresh()
    assert state.valid
    assert widget.error == ""
    assert len(widget.warnings) >= 2
    assert "placement warning" in widget.status
    assert len(synced) == 2  # warnings still sync the plan

    widget.yaml_text = SAMPLE_YAML.replace("persona: patient", "persona: member", 1)
    state = widget.refresh()
    assert state.valid
    assert widget.current_plan is not None
    assert widget.current_plan.persona_backed_columns[0].persona == "member"
    assert synced[-1].persona_backed_columns[0].persona == "member"

    builder = _builder(patient_df, tmp_path)
    preview = builder.preview_replace_pii()
    assert builder._nss_config is not None and builder._nss_config.replace_pii is not None
    assert isinstance(builder._nss_config.replace_pii.replacement_plan, PiiReplacementPlan)
    assert builder._nss_config.replace_pii.replacement_plan.persona_backed_columns[0].persona == "patient"
    assert preview.current_plan is not None

    assert builder._workdir is not None
    plan_path = builder._workdir.run_dir / PII_REPLACEMENT_PLAN_FILENAME
    assert plan_path.exists()
    preview.yaml_text = SAMPLE_YAML.replace("persona: patient", "persona: member", 1)
    preview.refresh()
    assert "persona: member" in plan_path.read_text()


def test_previewed_plan_rearms_process_data_only_when_the_plan_changes(patient_df, tmp_path):
    """Editing the plan after process_data() must let a re-run apply the edit."""
    builder = _builder(patient_df, tmp_path)
    builder._resolve_nss_config()
    assert builder._nss_config is not None and builder._nss_config.replace_pii is not None
    applied = builder._nss_config.replace_pii.replacement_plan
    assert isinstance(applied, PiiReplacementPlan)

    # Stand in for a completed process_data() run.
    builder._data_processed = True
    builder._applied_pii_plan = applied

    builder._apply_previewed_plan(applied)
    assert builder._data_processed  # re-opening the preview alone is not an edit

    edited = applied.model_copy(deep=True)
    edited.persona_backed_columns[0].persona = "member"
    builder._apply_previewed_plan(edited)
    assert not builder._data_processed

    assert builder._workdir is not None
    saved = PiiReplacementPlan.from_yaml_str((builder._workdir.run_dir / PII_REPLACEMENT_PLAN_FILENAME).read_text())
    assert saved.persona_backed_columns[0].persona == "member"
