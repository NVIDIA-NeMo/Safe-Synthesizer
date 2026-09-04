# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from nemo_safe_synthesizer.config.replace_pii import EntityType, PiiColumnPlan, PiiReplacementPlan
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

    def test_load_treats_missing_schema_version_as_v3(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.yaml"
        path.write_text("scope: dataframe\ncolumns_to_replace: []\n")

        assert load_plan(path) == PiiReplacementPlan()

    @pytest.mark.parametrize("schema_version", [1, 2, 0, -1])
    def test_load_rejects_unsupported_schema_version(self, tmp_path: Path, schema_version: int) -> None:
        path = tmp_path / "plan.yaml"
        path.write_text(f"schema_version: {schema_version}\nscope: dataframe\ncolumns_to_replace: []\n")

        with pytest.raises(ParameterError, match=f"unsupported schema version {schema_version}.*supports version 3"):
            load_plan(path)

    @pytest.mark.parametrize("schema_version", [True, 1.0, "1", None])
    def test_load_rejects_non_integer_schema_version(self, tmp_path: Path, schema_version: object) -> None:
        path = tmp_path / "plan.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": schema_version,
                    "scope": "dataframe",
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
