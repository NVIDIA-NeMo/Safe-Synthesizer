# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    ConditioningColumn,
    EntityType,
    LLMConfig,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiReplacementSettings,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.llm_client import InferenceSettings
from nemo_safe_synthesizer.pii_replacer.planning import (
    PlanDiscoverer,
    PlanDiscoveryInput,
    PlanEnhancer,
    resolve_plan,
    resolve_replacement_config,
)
from nemo_safe_synthesizer.pii_replacer.planning.resolver import row_counter_columns

from ..conftest import FakeLocalServer


class RecordingDiscoverer(PlanDiscoverer):
    def __init__(self, plan: PiiReplacementPlan | None = None) -> None:
        self.plan = plan
        self.inputs: list[PlanDiscoveryInput] = []

    def discover(self, discovery_input: PlanDiscoveryInput) -> PiiReplacementPlan:
        self.inputs.append(discovery_input)
        return self.plan or PiiReplacementPlan()


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


class PassThroughEnhancer(PlanEnhancer):
    def enhance(self, discovery_input: PlanDiscoveryInput, baseline: PiiReplacementPlan) -> PiiReplacementPlan:
        return baseline


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
    def test_auto_discovery_uses_the_llm_by_default(
        self,
        fixture_patient_df: pd.DataFrame,
        fixture_fake_local_servers: list[FakeLocalServer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)]
        )
        enhancer = RecordingEnhancer(expected)
        monkeypatch.setattr(
            "nemo_safe_synthesizer.pii_replacer.planning.llm.LLMPlanEnhancer",
            lambda config, settings=None: enhancer,
        )

        result = resolve_plan(fixture_patient_df, ReplacePiiConfig(), DataParameters())

        assert result is expected
        assert len(enhancer.calls) == 1
        assert len(fixture_fake_local_servers) == 1

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

    def test_configured_llm_constructs_default_enhancer(
        self,
        fixture_patient_df: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)]
        )
        enhancer = RecordingEnhancer(expected)
        monkeypatch.setenv("NSS_INFERENCE_ENDPOINT", "https://inference.example.com/v1")
        monkeypatch.setattr(
            "nemo_safe_synthesizer.pii_replacer.planning.llm.LLMPlanEnhancer",
            lambda config, settings=None: enhancer,
        )

        result = resolve_plan(
            fixture_patient_df,
            ReplacePiiConfig(llm=LLMConfig(model_id="served/model")),
            DataParameters(),
        )

        assert result is expected
        assert len(enhancer.calls) == 1

    def test_inline_plan_bypasses_discovery_but_retains_llm_for_replacement(
        self,
        fixture_patient_df: pd.DataFrame,
    ) -> None:
        plan = PiiReplacementPlan(columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)])
        discoverer = RecordingDiscoverer()
        enhancer = RecordingEnhancer(PiiReplacementPlan())
        config = ReplacePiiConfig(replacement=PiiReplacementSettings(plan=plan), llm=LLMConfig())

        result = resolve_plan(
            fixture_patient_df,
            config,
            DataParameters(),
            discoverer=discoverer,
            enhancer=enhancer,
        )

        assert result is plan
        assert discoverer.inputs == []
        assert enhancer.calls == []

    def test_profiles_select_a_stable_bounded_set_in_dataframe_order(self) -> None:
        dataframe = pd.DataFrame(
            {
                "patient_id": [index // 2 for index in range(12)],
                "event_index": list(range(12)),
                "name": [f"person-{index}" for index in range(12)],
            }
        )
        shuffled = dataframe.sample(frac=1, random_state=7)
        discoverer = RecordingDiscoverer()
        data_config = DataParameters(
            group_training_examples_by="patient_id",
            order_training_examples_by="event_index",
        )

        resolve_plan(dataframe, ReplacePiiConfig(), data_config, discoverer=discoverer, enhancer=PassThroughEnhancer())
        first_input = discoverer.inputs[0]
        resolve_plan(
            shuffled,
            ReplacePiiConfig(),
            data_config,
            discoverer=discoverer,
            enhancer=PassThroughEnhancer(),
        )
        second_input = discoverer.inputs[1]

        first_profiles = {profile.column_name: profile for profile in first_input.column_profiles}
        second_profiles = {profile.column_name: profile for profile in second_input.column_profiles}
        assert first_input.group_column == "patient_id"
        assert first_input.protected_columns == frozenset({"event_index"})
        first_samples = first_profiles["name"].samples
        second_samples = second_profiles["name"].samples
        assert len(first_samples) == 8
        assert set(first_samples) == set(second_samples)
        assert first_samples == tuple(value for value in dataframe["name"] if value in set(first_samples))
        assert second_samples == tuple(value for value in shuffled["name"] if value in set(second_samples))

    def test_discovery_input_uses_no_group_column_for_record_consistency(
        self,
        fixture_patient_df: pd.DataFrame,
    ) -> None:
        discoverer = RecordingDiscoverer()

        resolve_plan(
            fixture_patient_df,
            ReplacePiiConfig(),
            DataParameters(),
            discoverer=discoverer,
            enhancer=PassThroughEnhancer(),
        )

        assert discoverer.inputs[0].group_column is None

    def test_auto_discovery_skips_row_counter_columns(self) -> None:
        dataframe = pd.DataFrame(
            {"Unnamed: 0": [2, 0, 1], "customer_id": [1001, 1002, 1003], "name": ["Ada", "Grace", "Alan"]}
        )
        discovered = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(column_name=column, entity_type=entity_type)
                for column, entity_type in (
                    ("Unnamed: 0", EntityType.UNIQUE_IDENTIFIER),
                    ("customer_id", EntityType.UNIQUE_IDENTIFIER),
                    ("name", EntityType.FULL_NAME),
                )
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(),
            DataParameters(),
            discoverer=RecordingDiscoverer(discovered),
            enhancer=PassThroughEnhancer(),
        )

        assert [spec.column_name for spec in plan.columns_to_replace] == ["customer_id", "name"]

    def test_explicit_plans_may_replace_row_counter_columns(self) -> None:
        dataframe = pd.DataFrame({"row": [1, 2, 3]})
        plan = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="row", entity_type=EntityType.UNIQUE_IDENTIFIER)]
        )

        resolved = resolve_plan(
            dataframe, ReplacePiiConfig(replacement=PiiReplacementSettings(plan=plan)), DataParameters()
        )

        assert [spec.column_name for spec in resolved.columns_to_replace] == ["row"]

    @pytest.mark.parametrize(
        ("values", "is_counter"),
        [
            ([3, 0, 2, 1], True),
            ([1.0, 2.0, 3.0, 4.0], True),
            ([0, 1, 3, 4], False),
            ([17, 18, 19, 20], False),
            ([0, 1, 1, 2], False),
            ([True, False, True, False], False),
        ],
    )
    def test_row_counter_detection_uses_values_only(self, values: list[object], is_counter: bool) -> None:
        assert (row_counter_columns(pd.DataFrame({"column": values})) == {"column"}) is is_counter


@pytest.mark.unit
class TestResolvePlanWithLocalServer:
    def test_server_runs_for_enhancement_and_stops_before_validation(
        self,
        fixture_patient_df: pd.DataFrame,
        fixture_fake_local_servers: list[FakeLocalServer],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        expected = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)]
        )
        seen: dict[str, object] = {}

        def make_enhancer(config: LLMConfig, settings: InferenceSettings | None = None) -> PlanEnhancer:
            seen["settings"] = settings
            seen["running_during_enhance"] = fixture_fake_local_servers[0].running
            return RecordingEnhancer(expected)

        def record_validation(*args: object, **kwargs: object) -> None:
            seen["running_during_validation"] = fixture_fake_local_servers[0].running

        monkeypatch.setattr("nemo_safe_synthesizer.pii_replacer.planning.llm.LLMPlanEnhancer", make_enhancer)
        monkeypatch.setattr("nemo_safe_synthesizer.pii_replacer.planning.resolver.validate_plan", record_validation)

        result = resolve_plan(
            fixture_patient_df,
            ReplacePiiConfig(llm=LLMConfig()),
            DataParameters(),
        )

        assert result is expected
        assert seen["settings"] == fixture_fake_local_servers[0].inference_settings(max_workers=LLMConfig().max_workers)
        assert seen["running_during_enhance"] is True
        assert seen["running_during_validation"] is False
        assert not fixture_fake_local_servers[0].running

    def test_server_stops_when_enhancement_fails(
        self,
        fixture_patient_df: pd.DataFrame,
        fixture_fake_local_servers: list[FakeLocalServer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FailingEnhancer(PlanEnhancer):
            def enhance(self, discovery_input: PlanDiscoveryInput, baseline: PiiReplacementPlan) -> PiiReplacementPlan:
                raise RuntimeError("inference failed")

        monkeypatch.setattr(
            "nemo_safe_synthesizer.pii_replacer.planning.llm.LLMPlanEnhancer",
            lambda config, settings=None: FailingEnhancer(),
        )

        with pytest.raises(RuntimeError, match="inference failed"):
            resolve_plan(fixture_patient_df, ReplacePiiConfig(llm=LLMConfig()), DataParameters())

        assert len(fixture_fake_local_servers) == 1
        assert not fixture_fake_local_servers[0].running

    def test_explicit_plan_starts_no_server(
        self,
        fixture_patient_df: pd.DataFrame,
        fixture_fake_local_servers: list[FakeLocalServer],
    ) -> None:
        config = ReplacePiiConfig(
            replacement=PiiReplacementSettings(
                plan=PiiReplacementPlan(
                    columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)]
                )
            )
        )

        resolve_plan(fixture_patient_df, config, DataParameters())

        assert fixture_fake_local_servers == []

    def test_caller_supplied_enhancer_starts_no_server(
        self,
        fixture_patient_df: pd.DataFrame,
        fixture_fake_local_servers: list[FakeLocalServer],
    ) -> None:
        resolve_plan(
            fixture_patient_df,
            ReplacePiiConfig(llm=LLMConfig()),
            DataParameters(),
            enhancer=RecordingEnhancer(PiiReplacementPlan()),
        )

        assert fixture_fake_local_servers == []


class _RecordingLlm(PlanEnhancer):
    """Stands in for LLMPlanEnhancer; records whether the managed server was running for each LLM step."""

    def __init__(self, plan: PiiReplacementPlan, servers: list[FakeLocalServer]) -> None:
        self.plan = plan
        self.servers = servers
        self.running_during: dict[str, bool] = {}

    def enhance(self, discovery_input: PlanDiscoveryInput, baseline: PiiReplacementPlan) -> PiiReplacementPlan:
        self.running_during["enhance"] = self.servers[-1].running
        return self.plan

    def discover_data_to_sampler_value_mapping(
        self,
        dataframe: pd.DataFrame,
        plan: PiiReplacementPlan,
        catalog: object,
    ) -> dict[str, dict[str, list[str]]]:
        self.running_during["map"] = self.servers[-1].running
        return {"sex": {"X": ["female", "male"]}}


def _plan_needing_a_value_mapping() -> PiiReplacementPlan:
    return PiiReplacementPlan(
        columns_to_replace=[
            PiiColumnPlan(
                column_name="name",
                entity_type=EntityType.FIRST_NAME,
                depends_on=[ConditioningColumn(column_name="sex", entity_type=EntityType.GENDER)],
            )
        ]
    )


@pytest.mark.unit
class TestResolveReplacementConfigWithLocalServer:
    def test_plan_enhancement_and_value_mapping_share_one_server(
        self,
        fixture_fake_local_servers: list[FakeLocalServer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataframe = pd.DataFrame({"name": ["Ada", "Alan"], "sex": ["Female", "X"]})
        llm = _RecordingLlm(_plan_needing_a_value_mapping(), fixture_fake_local_servers)
        monkeypatch.setattr(
            "nemo_safe_synthesizer.pii_replacer.planning.llm.LLMPlanEnhancer", lambda config, settings=None: llm
        )

        resolved = resolve_replacement_config(
            dataframe,
            ReplacePiiConfig(llm=LLMConfig()),
            DataParameters(),
            sampler_value_catalog={EntityType.GENDER: ("female", "male")},
        )

        assert len(fixture_fake_local_servers) == 1
        assert llm.running_during == {"enhance": True, "map": True}
        assert not fixture_fake_local_servers[0].running
        assert resolved.inline_plan is not None
        assert resolved.inline_plan.data_to_sampler_value_mapping == {"sex": {"X": ["female", "male"]}}

    def test_value_mapping_alone_runs_in_its_own_server_session(
        self,
        fixture_fake_local_servers: list[FakeLocalServer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataframe = pd.DataFrame({"name": ["Ada", "Alan"], "sex": ["Female", "X"]})
        llm = _RecordingLlm(_plan_needing_a_value_mapping(), fixture_fake_local_servers)
        monkeypatch.setattr(
            "nemo_safe_synthesizer.pii_replacer.planning.llm.LLMPlanEnhancer", lambda config, settings=None: llm
        )

        resolve_replacement_config(
            dataframe,
            ReplacePiiConfig(llm=LLMConfig()),
            DataParameters(),
            enhancer=RecordingEnhancer(_plan_needing_a_value_mapping()),
            sampler_value_catalog={EntityType.GENDER: ("female", "male")},
        )

        assert len(fixture_fake_local_servers) == 1
        assert llm.running_during == {"map": True}
        assert not fixture_fake_local_servers[0].running
