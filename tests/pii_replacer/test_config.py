# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config mapping, managed assets, and replacer result stats."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import (
    AUTO_DISCOVERY,
    PersonaColumnSet,
    PiiColumnPlan,
    PiiEntity,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    PiiReplacementScope,
    PiiReplacementSettings,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.defaults import NSS_MANAGED_ASSETS_PATH_ENV, default_managed_assets_path
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.entities import Config, config_from_replace_pii
from nemo_safe_synthesizer.pii_replacer.replacer import (
    TabularPiiReplacer,
    _persona_transform_method,
    _plan_column_counts,
    _replacement_plan_source,
)


def test_replace_pii_defaults():
    cfg = SafeSynthesizerParameters()
    assert cfg.replace_pii is not None
    assert cfg.replace_pii.replacement_plan == AUTO_DISCOVERY
    assert cfg.replace_pii.person.backend == PiiPersonBackend.managed


def test_replace_pii_config_rejects_llm_enhancement_true():
    """Direct construction must refuse the unsupported LLM flag."""
    with pytest.raises(ValidationError, match=r"replace_pii\.llm_enhancement"):
        ReplacePiiConfig(llm_enhancement=True)


def test_person_config_requires_sdg_pgms_src_for_pgm():
    with pytest.raises((ParameterError, ValidationError), match=r"sdg_pgms_src"):
        PiiPersonConfig(backend=PiiPersonBackend.pgm)


def test_person_config_default_leaves_sdg_pgms_src_unset():
    cfg = PiiPersonConfig()
    assert cfg.backend == PiiPersonBackend.managed
    assert cfg.sdg_pgms_src is None


def test_config_from_replace_pii_maps_user_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("PERSON_RANDOM_SEED", raising=False)
    assets = tmp_path / "assets"
    assets.mkdir()
    cfg = ReplacePiiConfig(
        replacement=PiiReplacementSettings(locale="en_GB", seed=7),
        person=PiiPersonConfig(backend=PiiPersonBackend.faker, managed_assets_path=str(assets)),
    )
    engine = config_from_replace_pii(cfg)
    assert isinstance(engine, Config)
    assert engine.locale == "en_GB"
    assert engine.random_seed == 7
    assert engine.persona_backend == "faker"
    assert engine.managed_assets_path == str(assets)
    assert engine.sdg_pgms_src is None


def test_config_from_replace_pii_seed_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("PERSON_RANDOM_SEED", "99")
    cfg = ReplacePiiConfig(replacement=PiiReplacementSettings(seed=None))
    assert config_from_replace_pii(cfg).random_seed == 99


def test_default_managed_assets_path(monkeypatch):
    monkeypatch.delenv(NSS_MANAGED_ASSETS_PATH_ENV, raising=False)
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


def test_managed_backend_smoke(tmp_path, fixture_patient_df: pd.DataFrame):
    """Managed sampling works when a locale parquet is present (no NGC install required)."""
    assets_root = tmp_path / "managed-assets"
    (assets_root / "datasets").mkdir(parents=True)
    # Minimal columns the managed sampler / name writer read from the parquet.
    pd.DataFrame(
        {
            "first_name": [f"SynFirst{i}" for i in range(40)],
            "last_name": [f"SynLast{i}" for i in range(40)],
            "sex": (["Female", "Male"] * 20),
            "ethnic_background": (["white", "black"] * 20),
        }
    ).to_parquet(assets_root / "datasets" / "en_US.parquet")

    plan = PiiReplacementPlan(
        scope=PiiReplacementScope.group,
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
        ReplacePiiConfig(
            replacement_plan=plan,
            replacement=PiiReplacementSettings(locale="en_US"),
            person=PiiPersonConfig(backend=PiiPersonBackend.managed, managed_assets_path=str(assets_root)),
        ),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(fixture_patient_df)
    assert replacer.result is not None
    out = replacer.result.transformed_df
    assert out["first_name"].tolist() != fixture_patient_df["first_name"].tolist()
    assert replacer.result.column_statistics["first_name"].transform_methods == {"en_US personas"}


def test_persona_transform_method_labels():
    assert _persona_transform_method("en_US", "managed") == "en_US personas"
    assert _persona_transform_method("en_US", "pgm") == "en_US PGM"
    assert _persona_transform_method("en_US", "faker") == "en_US Faker"


def test_column_statistics_transform_methods_and_entity_counts(fixture_patient_df: pd.DataFrame):
    plan = PiiReplacementPlan(
        scope=PiiReplacementScope.group,
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                ],
            ),
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
        ],
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(
            replacement_plan=plan,
            replacement=PiiReplacementSettings(locale="en_US"),
            person=PiiPersonConfig(backend=PiiPersonBackend.faker),
        ),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(fixture_patient_df)
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
        scope=PiiReplacementScope.group,
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[
                    PiiColumnPlan(
                        column_name="date_of_birth",
                        entity_type=PiiEntity.date_of_birth,
                        patterns=["%m/%d/%Y"],
                    ),
                ],
            ),
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="record_id", entity_type=PiiEntity.unique_identifier),
        ],
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(
            replacement_plan=plan,
            replacement=PiiReplacementSettings(locale="en_US"),
            person=PiiPersonConfig(backend=PiiPersonBackend.faker),
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


def test_plan_column_counts():
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                    PiiColumnPlan(column_name="last_name", entity_type=PiiEntity.last_name),
                ],
            ),
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
        ],
    )
    assert _plan_column_counts(plan) == (1, 2, 1)


def test_replacement_plan_source(tmp_path):
    plan_path = str(tmp_path / "plan.yaml")
    assert _replacement_plan_source(ReplacePiiConfig()) == "auto_discovery"
    assert _replacement_plan_source(ReplacePiiConfig(replacement_plan=plan_path)) == plan_path


def test_replace_pii_null_disables():
    cfg = SafeSynthesizerParameters.model_validate({"replace_pii": None})
    assert cfg.replace_pii is None


def test_managed_backend_says_so_when_it_falls_back_to_faker(caplog):
    """The two are interchangeable enough to continue, but not to switch in silence."""
    import logging

    from nemo_safe_synthesizer.pii_replacer.replacement import PersonaEngine

    caplog.set_level(logging.WARNING)
    cfg = ReplacePiiConfig(
        person=PiiPersonConfig(backend=PiiPersonBackend.managed, managed_assets_path="/nonexistent/assets")
    )
    engine = PersonaEngine(config_from_replace_pii(cfg), 1)

    assert engine.backend == "faker"
    assert any("generating personas with Faker instead" in record.getMessage() for record in caplog.records)


def test_replacement_plan_artifact(tmp_path, fixture_patient_df: pd.DataFrame):
    import yaml

    from nemo_safe_synthesizer.pii_replacer.planning import (
        PII_REPLACEMENT_PLAN_FILENAME,
        load_plan_from_path,
        save_plan_to_path,
    )

    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=AUTO_DISCOVERY, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(group_training_examples_by="patient_id"),
        workdir=tmp_path,
    )
    replacer.transform_df(fixture_patient_df)
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

    compact_plan_file = save_plan_to_path(
        PiiReplacementPlan(
            persona_backed_columns=[
                PersonaColumnSet(
                    persona="patient",
                    columns_to_replace=[
                        PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                    ],
                )
            ]
        ),
        tmp_path / "compact_plan.yaml",
    )
    compact_plan_data = yaml.safe_load(compact_plan_file.read_text())
    assert "match_persona_by" not in compact_plan_data["persona_backed_columns"][0]


def test_managed_fallback_stats_report_faker():
    """If the managed backend fails, the fallback should report Faker."""
    df = pd.DataFrame({"first_name": [f"First{i}" for i in range(20)]})
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="person",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                ],
            )
        ],
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(
            replacement_plan=plan,
            person=PiiPersonConfig(backend=PiiPersonBackend.managed, managed_assets_path="/nonexistent/assets"),
        ),
        data_config=DataParameters(),
    )
    replacer.transform_df(df)
    assert replacer.result is not None
    methods = replacer.result.column_statistics["first_name"].transform_methods
    assert methods == {"en_US Faker"}
