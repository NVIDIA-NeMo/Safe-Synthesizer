# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    ConditioningColumn,
    DataToSamplerValueMapping,
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.planning import (
    PlanDiscoverer,
    PlanDiscoveryInput,
    PlanEnhancer,
    resolve_replacement_config,
)
from nemo_safe_synthesizer.pii_replacer.planning.dependency_mappings import mapping_inputs


def _plan(data_to_sampler_value_mapping: DataToSamplerValueMapping | None = None) -> PiiReplacementPlan:
    return PiiReplacementPlan(
        columns_to_replace=[
            PiiColumnPlan(
                column_name="first_name",
                entity_type=EntityType.FIRST_NAME,
                depends_on=[
                    ConditioningColumn(column_name="sex", entity_type=EntityType.GENDER),
                    ConditioningColumn(column_name="race", entity_type=EntityType.ETHNIC_BACKGROUND),
                ],
            )
        ],
        data_to_sampler_value_mapping=data_to_sampler_value_mapping or {},
    )


class _StaticPlanDiscoverer(PlanDiscoverer):
    def discover(self, discovery_input: PlanDiscoveryInput) -> PiiReplacementPlan:
        del discovery_input
        return _plan()


class _PassThroughEnhancer(PlanEnhancer):
    def enhance(self, discovery_input: PlanDiscoveryInput, baseline: PiiReplacementPlan) -> PiiReplacementPlan:
        return baseline


@pytest.mark.unit
class TestDependencyMappings:
    def test_mapping_inputs_report_non_string_values_in_their_mapping_key_form(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada", "Grace"], "sex": [1.0, 2.0], "race": ["White"] * 2})

        inputs = mapping_inputs(
            dataframe,
            _plan(),
            {EntityType.GENDER: ("female", "male"), EntityType.ETHNIC_BACKGROUND: ("white",)},
        )

        assert [(item.column_name, item.source_values) for item in inputs] == [("sex", ("1.0", "2.0"))]

    def test_mapping_inputs_omit_case_insensitive_identity_values(self) -> None:
        dataframe = pd.DataFrame(
            {
                "first_name": ["Ada", "Grace"],
                "sex": ["Female", "Non-binary"],
                "race": ["White", "Asian"],
            }
        )

        inputs = mapping_inputs(
            dataframe,
            _plan(),
            {
                EntityType.GENDER: ("female", "male"),
                EntityType.ETHNIC_BACKGROUND: ("east asian", "white"),
            },
        )

        assert [(item.column_name, item.source_values) for item in inputs] == [
            ("sex", ("Non-binary",)),
            ("race", ("Asian",)),
        ]

    def test_auto_mapping_rejects_dependency_values_over_128_characters(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["x" * 129], "race": ["White"]})

        with pytest.raises(ParameterError, match="values longer than 128 characters"):
            mapping_inputs(
                dataframe,
                _plan(),
                {
                    EntityType.GENDER: ("female", "male"),
                    EntityType.ETHNIC_BACKGROUND: ("white",),
                },
            )

    def test_identity_only_auto_mapping_resolves_to_an_empty_inline_mapping(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["Female"], "race": ["WHITE"]})

        resolved = resolve_replacement_config(
            dataframe,
            ReplacePiiConfig(),
            DataParameters(),
            discoverer=_StaticPlanDiscoverer(),
            enhancer=_PassThroughEnhancer(),
            sampler_value_catalog={
                EntityType.GENDER: ("female", "male"),
                EntityType.ETHNIC_BACKGROUND: ("white",),
            },
        )

        assert resolved.inline_plan is not None
        assert resolved.inline_plan.data_to_sampler_value_mapping == {}

    def test_explicit_plan_uses_case_insensitive_matches_without_discovery(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["Female"], "race": ["White"]})

        resolved = resolve_replacement_config(
            dataframe,
            ReplacePiiConfig(replacement_plan=_plan()),
            DataParameters(),
            sampler_value_catalog={
                EntityType.GENDER: ("female", "male"),
                EntityType.ETHNIC_BACKGROUND: ("white",),
            },
        )

        assert resolved.inline_plan is not None
        assert resolved.inline_plan.data_to_sampler_value_mapping == {}

    def test_explicit_plan_rejects_unmatched_values_without_entries(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada", "Grace"], "sex": ["X", "Female"], "race": ["White"] * 2})

        with pytest.raises(ParameterError, match=r"'sex' \(1\)"):
            resolve_replacement_config(
                dataframe,
                ReplacePiiConfig(replacement_plan=_plan()),
                DataParameters(),
                sampler_value_catalog={
                    EntityType.GENDER: ("female", "male"),
                    EntityType.ETHNIC_BACKGROUND: ("white",),
                },
            )

    def test_manual_mapping_is_authoritative_and_validated_by_dependency_column(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["Woman"], "race": ["White"]})
        config = ReplacePiiConfig(
            replacement_plan=_plan({"sex": {"Woman": ["female"]}}),
        )

        resolved = resolve_replacement_config(
            dataframe,
            config,
            DataParameters(),
            sampler_value_catalog={
                EntityType.GENDER: ("female", "male"),
                EntityType.ETHNIC_BACKGROUND: ("white",),
            },
        )

        assert resolved.inline_plan is not None
        assert resolved.inline_plan.data_to_sampler_value_mapping == {"sex": {"Woman": ["female"]}}

    @pytest.mark.parametrize(
        ("mappings", "error"),
        [
            ({"unknown": {"Woman": ["female"]}}, "not used as a dependency"),
            ({"sex": {"Woman": ["unknown"]}}, "values not supported"),
        ],
    )
    def test_manual_mapping_rejects_invalid_plan_or_sampler_bindings(
        self,
        mappings: DataToSamplerValueMapping,
        error: str,
    ) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["Woman"], "race": ["White"]})
        config = ReplacePiiConfig(
            replacement_plan=_plan(mappings),
        )

        with pytest.raises(ParameterError, match=error):
            resolve_replacement_config(
                dataframe,
                config,
                DataParameters(),
                sampler_value_catalog={EntityType.GENDER: ("female", "male")},
            )

    def test_mapping_for_a_column_the_sampler_cannot_condition_on_is_kept_with_a_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["Female"], "race": ["Asian"]})
        mappings: DataToSamplerValueMapping = {"race": {"Asian": ["east asian"]}}

        resolved = resolve_replacement_config(
            dataframe,
            ReplacePiiConfig(replacement_plan=_plan(mappings)),
            DataParameters(),
            sampler_value_catalog={EntityType.GENDER: ("female", "male")},
        )

        assert resolved.inline_plan is not None
        assert resolved.inline_plan.data_to_sampler_value_mapping == mappings
        assert "Ignoring data_to_sampler_value_mapping for a column the configured sampler cannot condition on" in (
            caplog.text
        )

    def test_built_in_variants_resolve_without_an_llm_or_manual_entries(self) -> None:
        dataframe = pd.DataFrame(
            {"first_name": ["Ada", "Grace", "Alan", "Mae"], "sex": ["F", "Woman", "M", "female"], "race": ["White"] * 4}
        )

        resolved = resolve_replacement_config(
            dataframe,
            ReplacePiiConfig(),
            DataParameters(),
            discoverer=_StaticPlanDiscoverer(),
            enhancer=_PassThroughEnhancer(),
            sampler_value_catalog={
                EntityType.GENDER: ("female", "male"),
                EntityType.ETHNIC_BACKGROUND: ("white",),
            },
        )

        assert resolved.inline_plan is not None
        assert resolved.inline_plan.data_to_sampler_value_mapping == {
            "sex": {"F": ["female"], "Woman": ["female"], "M": ["male"]}
        }

    def test_built_in_variants_map_to_localized_persona_sex_values(self) -> None:
        dataframe = pd.DataFrame(
            {
                "first_name": ["Ada", "Grace", "Alan", "Mae"],
                "sex": ["F", "Female", "Man", "Homme"],
                "race": ["White"] * 4,
            }
        )

        resolved = resolve_replacement_config(
            dataframe,
            ReplacePiiConfig(),
            DataParameters(),
            discoverer=_StaticPlanDiscoverer(),
            enhancer=_PassThroughEnhancer(),
            sampler_value_catalog={
                EntityType.GENDER: ("femme", "homme"),
                EntityType.ETHNIC_BACKGROUND: ("white",),
            },
        )

        assert resolved.inline_plan is not None
        assert resolved.inline_plan.data_to_sampler_value_mapping == {
            "sex": {"F": ["femme"], "Female": ["femme"], "Man": ["homme"]}
        }

    def test_plan_entries_take_precedence_over_built_in_variants(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada", "Alan"], "sex": ["F", "M"], "race": ["White"] * 2})

        resolved = resolve_replacement_config(
            dataframe,
            ReplacePiiConfig(replacement_plan=_plan({"sex": {"f": ["female", "male"]}})),
            DataParameters(),
            sampler_value_catalog={
                EntityType.GENDER: ("female", "male"),
                EntityType.ETHNIC_BACKGROUND: ("white",),
            },
        )

        assert resolved.inline_plan is not None
        assert resolved.inline_plan.data_to_sampler_value_mapping == {"sex": {"M": ["male"], "f": ["female", "male"]}}

    def test_mapping_inputs_reject_high_cardinality_dependency_columns(self) -> None:
        dataframe = pd.DataFrame(
            {"first_name": ["Ada"] * 1001, "sex": [f"value-{i}" for i in range(1001)], "race": ["White"] * 1001}
        )

        with pytest.raises(ParameterError, match="'sex' has more than 1000 distinct values"):
            mapping_inputs(
                dataframe,
                _plan(),
                {EntityType.GENDER: ("female", "male"), EntityType.ETHNIC_BACKGROUND: ("white",)},
            )
