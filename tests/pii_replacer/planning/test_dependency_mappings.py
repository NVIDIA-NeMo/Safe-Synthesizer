# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    ConditioningColumn,
    DependencyValueMappings,
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.planning import (
    PlanDiscoverer,
    PlanDiscoveryInput,
    resolve_replacement_config,
)
from nemo_safe_synthesizer.pii_replacer.planning.dependency_mappings import mapping_inputs


def _plan(dependency_value_mappings: DependencyValueMappings | None = None) -> PiiReplacementPlan:
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
        dependency_value_mappings=dependency_value_mappings or {},
    )


class _StaticPlanDiscoverer(PlanDiscoverer):
    def discover(self, discovery_input: PlanDiscoveryInput) -> PiiReplacementPlan:
        del discovery_input
        return _plan()


@pytest.mark.unit
class TestDependencyMappings:
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

        with pytest.raises(ParameterError, match="no longer than 128 characters"):
            mapping_inputs(
                dataframe,
                _plan(),
                {
                    EntityType.GENDER: ("female", "male"),
                    EntityType.ETHNIC_BACKGROUND: ("white",),
                },
            )

    def test_auto_mapping_without_llm_fails_only_for_unmatched_values(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["Woman"], "race": ["White"]})
        config = ReplacePiiConfig()

        with pytest.raises(ParameterError, match="configure replace_pii.llm, or generate and edit"):
            resolve_replacement_config(
                dataframe,
                config,
                DataParameters(),
                discoverer=_StaticPlanDiscoverer(),
                dependency_labels={
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
            dependency_labels={
                EntityType.GENDER: ("female", "male"),
                EntityType.ETHNIC_BACKGROUND: ("white",),
            },
        )

        assert resolved.inline_plan is not None
        assert resolved.inline_plan.dependency_value_mappings == {}

    def test_explicit_plan_does_not_separately_auto_discover_mappings(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["Woman"], "race": ["White"]})

        resolved = resolve_replacement_config(
            dataframe,
            ReplacePiiConfig(replacement_plan=_plan()),
            DataParameters(),
            dependency_labels={
                EntityType.GENDER: ("female", "male"),
                EntityType.ETHNIC_BACKGROUND: ("white",),
            },
        )

        assert resolved.inline_plan is not None
        assert resolved.inline_plan.dependency_value_mappings == {}

    def test_manual_mapping_is_authoritative_and_validated_by_dependency_column(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["Woman"], "race": ["White"]})
        config = ReplacePiiConfig(
            replacement_plan=_plan({"sex": {"Woman": ["female"]}}),
        )

        resolved = resolve_replacement_config(
            dataframe,
            config,
            DataParameters(),
            dependency_labels={
                EntityType.GENDER: ("female", "male"),
                EntityType.ETHNIC_BACKGROUND: ("white",),
            },
        )

        assert resolved.inline_plan is not None
        assert resolved.inline_plan.dependency_value_mappings == {"sex": {"Woman": ["female"]}}

    @pytest.mark.parametrize(
        ("mappings", "error"),
        [
            ({"unknown": {"Woman": ["female"]}}, "not used as a dependency"),
            ({"sex": {"Woman": ["unknown"]}}, "labels not supported"),
        ],
    )
    def test_manual_mapping_rejects_invalid_plan_or_sampler_bindings(
        self,
        mappings: dict[str, dict[str, list[str] | None]],
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
                dependency_labels={EntityType.GENDER: ("female", "male")},
            )
