# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from nemo_safe_synthesizer.config.replace_pii import (
    ConditioningColumn,
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
)
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.planning import load_plan, save_plan


@pytest.mark.unit
class TestPlanIo:
    def test_save_then_load_round_trip(self, tmp_path: Path) -> None:
        plan = PiiReplacementPlan(columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)])

        path = save_plan(plan, tmp_path / "nested" / "plan.yaml")

        assert path == tmp_path / "nested" / "plan.yaml"
        assert yaml.safe_load(path.read_text())["schema_version"] == 3
        assert load_plan(path) == plan

    def test_save_uses_canonical_sparse_document_and_preserves_inference(self, tmp_path: Path) -> None:
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(column_name="first", entity_type=EntityType.FIRST_NAME),
                PiiColumnPlan(
                    column_name="email",
                    entity_type=EntityType.EMAIL,
                    depends_on=[
                        ConditioningColumn(column_name="first"),
                        ConditioningColumn(column_name="company", entity_type=EntityType.ORGANIZATION),
                    ],
                ),
            ],
            dependency_value_mappings={"company": {"Independent": None}},
        )
        path = save_plan(plan, tmp_path / "plan.yaml")

        assert yaml.safe_load(path.read_text()) == {
            "schema_version": 3,
            "columns_to_replace": [
                {"column_name": "first", "entity_type": "first_name"},
                {
                    "column_name": "email",
                    "entity_type": "email",
                    "depends_on": [
                        {"column_name": "first"},
                        {"column_name": "company", "entity_type": "organization"},
                    ],
                },
            ],
            "dependency_value_mappings": {"company": {"Independent": None}},
        }

        loaded = load_plan(path)
        assert loaded == plan
        inferred = loaded.columns_to_replace[1].depends_on[0]
        assert inferred.entity_type is EntityType.FIRST_NAME
        assert "entity_type" not in inferred.model_fields_set

        second_path = save_plan(loaded, tmp_path / "second.yaml")
        assert second_path.read_text() == path.read_text()

    def test_save_always_includes_core_fields(self, tmp_path: Path) -> None:
        path = save_plan(PiiReplacementPlan(), tmp_path / "plan.yaml")

        assert yaml.safe_load(path.read_text()) == {
            "schema_version": 3,
            "columns_to_replace": [],
            "dependency_value_mappings": {},
        }

    def test_load_treats_missing_schema_version_as_v3(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.yaml"
        path.write_text("columns_to_replace: []\n")

        assert load_plan(path) == PiiReplacementPlan()

    @pytest.mark.parametrize("schema_version", [1, 2, 0, -1])
    def test_load_rejects_unsupported_schema_version(self, tmp_path: Path, schema_version: int) -> None:
        path = tmp_path / "plan.yaml"
        path.write_text(f"schema_version: {schema_version}\ncolumns_to_replace: []\n")

        with pytest.raises(ParameterError, match=f"unsupported schema version {schema_version}.*supports version 3"):
            load_plan(path)

    @pytest.mark.parametrize("schema_version", [True, 1.0, "1", None])
    def test_load_rejects_non_integer_schema_version(self, tmp_path: Path, schema_version: object) -> None:
        path = tmp_path / "plan.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": schema_version,
                    "columns_to_replace": [],
                }
            )
        )

        with pytest.raises(ParameterError, match="schema_version must be an integer"):
            load_plan(path)

    def test_load_rejects_non_mapping_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.yaml"
        path.write_text("- not\n- a\n- mapping\n")

        with pytest.raises(ParameterError, match="must contain a mapping"):
            load_plan(path)

    def test_load_wraps_missing_file_error(self, tmp_path: Path) -> None:
        with pytest.raises(ParameterError, match="Could not read PII replacement plan file"):
            load_plan(tmp_path / "missing.yaml")

    def test_load_rejects_unknown_plan_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.yaml"
        path.write_text(yaml.safe_dump({"scope": "group"}))

        with pytest.raises(ParameterError, match="Unknown configuration field 'scope'"):
            load_plan(path)
