# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from nemo_safe_synthesizer.config.replace_pii import EntityType, PiiColumnPlan, PiiReplacementPlan
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.planning import load_plan, save_plan


@pytest.mark.unit
class TestPlanIo:
    def test_save_then_load_round_trip(self, tmp_path: Path) -> None:
        plan = PiiReplacementPlan(columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)])

        path = save_plan(plan, tmp_path / "nested" / "plan.yaml")

        assert path == tmp_path / "nested" / "plan.yaml"
        assert load_plan(path) == plan

    def test_load_rejects_non_mapping_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.yaml"
        path.write_text("- not\n- a\n- mapping\n")

        with pytest.raises(ParameterError, match="must contain a mapping"):
            load_plan(path)

    def test_load_wraps_missing_file_error(self, tmp_path: Path) -> None:
        with pytest.raises(ParameterError, match="Could not read PII replacement plan file"):
            load_plan(tmp_path / "missing.yaml")
