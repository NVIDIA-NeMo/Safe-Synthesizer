# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    LLMConfig,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiReplacementScope,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.planning import (
    PlanDiscoverer,
    PlanDiscoveryInput,
    PlanEnhancer,
    load_plan,
    resolve_plan,
)


class RecordingDiscoverer(PlanDiscoverer):
    def __init__(self, plan: PiiReplacementPlan | None = None) -> None:
        self.plan = plan
        self.inputs: list[PlanDiscoveryInput] = []

    def discover(self, discovery_input: PlanDiscoveryInput) -> PiiReplacementPlan:
        self.inputs.append(discovery_input)
        return self.plan or PiiReplacementPlan(scope=discovery_input.scope)


class RecordingEnhancer(PlanEnhancer):
    def __init__(self, plan: PiiReplacementPlan) -> None:
        self.plan = plan
        self.calls: list[tuple[PlanDiscoveryInput, PiiReplacementPlan]] = []

    def enhance(
        self,
        discovery_input: PlanDiscoveryInput,
        baseline: PiiReplacementPlan,
    ) -> PiiReplacementPlan:
        self.calls.append((discovery_input, baseline))
        return self.plan


@pytest.fixture
def fixture_patient_df() -> pd.DataFrame:
    """Return grouped patient rows for replacement-plan discovery tests."""
    return pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "event_index": [1, 2, 1, 2],
            "name": ["Ada Lovelace", "Ada Lovelace", "Grace Hopper", "Grace Hopper"],
            "email": ["ada@example.com", "ada@example.com", "grace@example.com", "grace@example.com"],
        }
    )


@pytest.mark.unit
class TestResolvePlan:
    def test_auto_discovery_uses_empty_heuristic_by_default(self, fixture_patient_df: pd.DataFrame) -> None:
        plan = resolve_plan(fixture_patient_df, ReplacePiiConfig(), DataParameters())

        assert plan == PiiReplacementPlan(scope=PiiReplacementScope.DATAFRAME)

    def test_llm_enhancer_receives_heuristic_baseline(self, fixture_patient_df: pd.DataFrame) -> None:
        baseline = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="name", entity_type=EntityType.FULL_NAME)]
        )
        enhanced = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)]
        )
        discoverer = RecordingDiscoverer(baseline)
        enhancer = RecordingEnhancer(enhanced)

        result = resolve_plan(
            fixture_patient_df,
            ReplacePiiConfig(llm=LLMConfig()),
            DataParameters(),
            discoverer=discoverer,
            enhancer=enhancer,
        )

        assert result is enhanced
        assert len(discoverer.inputs) == 1
        assert enhancer.calls == [(discoverer.inputs[0], baseline)]

    def test_configured_llm_without_enhancer_fails(self, fixture_patient_df: pd.DataFrame) -> None:
        with pytest.raises(ParameterError, match="no LLM plan enhancer is available"):
            resolve_plan(
                fixture_patient_df,
                ReplacePiiConfig(llm=LLMConfig()),
                DataParameters(),
            )

    def test_inline_plan_bypasses_discovery_but_retains_llm_for_replacement(
        self,
        fixture_patient_df: pd.DataFrame,
    ) -> None:
        plan = PiiReplacementPlan(columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)])
        discoverer = RecordingDiscoverer()
        enhancer = RecordingEnhancer(PiiReplacementPlan())
        config = ReplacePiiConfig(replacement_plan=plan, llm=LLMConfig())

        result = resolve_plan(
            fixture_patient_df,
            config,
            DataParameters(),
            discoverer=discoverer,
            enhancer=enhancer,
        )

        assert result is plan
        assert config.llm is not None
        assert discoverer.inputs == []
        assert enhancer.calls == []

    def test_plan_file_bypasses_discovery(self, fixture_patient_df: pd.DataFrame, tmp_path: Path) -> None:
        path = tmp_path / "plan.yaml"
        path.write_text("scope: dataframe\ncolumns_to_replace:\n  - column_name: email\n    entity_type: email\n")
        discoverer = RecordingDiscoverer()

        result = resolve_plan(
            fixture_patient_df,
            ReplacePiiConfig(replacement_plan=str(path)),
            DataParameters(),
            discoverer=discoverer,
        )

        assert result.columns_to_replace[0].column_name == "email"
        assert discoverer.inputs == []

    def test_profiles_are_bounded_deterministic_and_keep_group_metadata_separate(
        self,
        fixture_patient_df: pd.DataFrame,
    ) -> None:
        discoverer = RecordingDiscoverer()
        data_config = DataParameters(
            group_training_examples_by="patient_id",
            order_training_examples_by="event_index",
        )

        resolve_plan(fixture_patient_df, ReplacePiiConfig(), data_config, discoverer=discoverer)
        first_input = discoverer.inputs[0]
        resolve_plan(
            fixture_patient_df.sample(frac=1, random_state=7),
            ReplacePiiConfig(),
            data_config,
            discoverer=discoverer,
        )
        second_input = discoverer.inputs[1]

        first_profiles = {profile.column_name: profile for profile in first_input.column_profiles}
        second_profiles = {profile.column_name: profile for profile in second_input.column_profiles}
        assert first_input.scope is PiiReplacementScope.GROUP
        assert first_input.group_column == "patient_id"
        assert first_input.protected_columns == frozenset({"event_index"})
        assert first_profiles["name"].samples == second_profiles["name"].samples

    def test_output_path_persists_the_final_plan(self, fixture_patient_df: pd.DataFrame, tmp_path: Path) -> None:
        path = tmp_path / "pii_replacement_plan.yaml"

        plan = resolve_plan(
            fixture_patient_df,
            ReplacePiiConfig(),
            DataParameters(),
            output_path=path,
        )

        assert path.exists()
        assert load_plan(path) == plan
