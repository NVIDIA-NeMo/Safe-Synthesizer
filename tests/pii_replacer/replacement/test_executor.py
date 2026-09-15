# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    ConditioningColumn,
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiSamplerBackend,
)
from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.pii_replacer.replacement.compiler import compile_plan
from nemo_safe_synthesizer.pii_replacer.replacement.executor import (
    ReplacementExecutionResult,
    StructuredReplacementExecutor,
    resolve_base_seed,
)
from nemo_safe_synthesizer.pii_replacer.replacement.generation import ReplacementGenerationRequest


class _RecordingGenerator:
    backend = PiiSamplerBackend.FAKER

    def __init__(self) -> None:
        self.requests: list[ReplacementGenerationRequest] = []

    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        dependencies = {
            entity_type: value.normalized_value
            for entity_type, value in request.effective_dependency_tuple
            if value is not None
        }
        if request.entity_type is EntityType.EMAIL:
            prefix = dependencies.get(EntityType.FIRST_NAME) or dependencies.get(EntityType.ORGANIZATION) or "user"
            return f"{prefix.casefold()}-{request.seed}@synthetic.test"
        return f"synthetic-{request.entity_type.value}-{request.seed}"


class _ConstantGenerator(_RecordingGenerator):
    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        return "Synthetic"


class _OriginalThenReplacementGenerator(_RecordingGenerator):
    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        return request.original_value if len(self.requests) == 1 else "Synthetic"


def _target(
    column_name: str,
    entity_type: EntityType,
    *dependencies: tuple[str, EntityType | None],
) -> PiiColumnPlan:
    return PiiColumnPlan(
        column_name=column_name,
        entity_type=entity_type,
        depends_on=[
            ConditioningColumn(column_name=name)
            if dependency_type is None
            else ConditioningColumn(column_name=name, entity_type=dependency_type)
            for name, dependency_type in dependencies
        ],
    )


def _plan(*targets: PiiColumnPlan) -> PiiReplacementPlan:
    return PiiReplacementPlan(columns_to_replace=list(targets))


def _execute(
    dataframe: pd.DataFrame,
    plan: PiiReplacementPlan,
    generator: _RecordingGenerator,
    *,
    group_column: str | None = None,
    dependency_value_mappings: dict[str, dict[str, list[str] | None]] | None = None,
) -> ReplacementExecutionResult:
    return StructuredReplacementExecutor(
        plan,
        generator,
        group_column=group_column,
        base_seed=42,
        dependency_value_mappings=dependency_value_mappings,
    ).execute(dataframe)


@pytest.mark.unit
class TestPlanCompiler:
    def test_stable_topological_order_preserves_independent_plan_order(self) -> None:
        plan = _plan(
            _target("email", EntityType.EMAIL, ("first", None)),
            _target("unrelated", EntityType.UNIQUE_IDENTIFIER),
            _target("first", EntityType.FIRST_NAME),
        )

        assert [spec.column_name for spec in compile_plan(plan)] == ["unrelated", "first", "email"]

    def test_defensively_rejects_a_cycle_when_model_validation_is_bypassed(self) -> None:
        first = PiiColumnPlan.model_construct(
            column_name="first",
            entity_type=EntityType.FIRST_NAME,
            pattern=None,
            depends_on=[ConditioningColumn.model_construct(column_name="full", entity_type=EntityType.FULL_NAME)],
        )
        full = PiiColumnPlan.model_construct(
            column_name="full",
            entity_type=EntityType.FULL_NAME,
            pattern=None,
            depends_on=[ConditioningColumn.model_construct(column_name="first", entity_type=EntityType.FIRST_NAME)],
        )

        with pytest.raises(ParameterError, match="cycle involving"):
            compile_plan(PiiReplacementPlan.model_construct(columns_to_replace=[first, full]))


@pytest.mark.unit
class TestStructuredReplacementExecutor:
    def test_resolves_sampler_labels_from_the_dependency_source_column(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["Woman"]})
        plan = _plan(_target("first_name", EntityType.FIRST_NAME, ("sex", EntityType.GENDER)))
        generator = _RecordingGenerator()

        _execute(
            dataframe,
            plan,
            generator,
            dependency_value_mappings={"sex": {"Woman": ["female"]}},
        )

        assert generator.requests[0].resolved_dependency_labels == ((EntityType.GENDER, ("female",)),)

    def test_reads_upstream_replacements_and_preserves_duplicate_indexes(self) -> None:
        dataframe = pd.DataFrame(
            {"first": ["Ada", "Grace"], "email": ["ada@example.com", "grace@example.com"]},
            index=[7, 7],
        )
        original = dataframe.copy(deep=True)
        plan = _plan(
            _target("email", EntityType.EMAIL, ("first", None)),
            _target("first", EntityType.FIRST_NAME),
        )
        generator = _RecordingGenerator()

        result = _execute(dataframe, plan, generator)

        pd.testing.assert_frame_equal(dataframe, original)
        assert result.dataframe.index.tolist() == [7, 7]
        assert [request.entity_type for request in generator.requests] == [
            EntityType.FIRST_NAME,
            EntityType.FIRST_NAME,
            EntityType.EMAIL,
            EntityType.EMAIL,
        ]
        dependency_values = [request.effective_dependency_tuple[0][1] for request in generator.requests[2:]]
        assert all(value is not None for value in dependency_values)
        assert [value.normalized_value for value in dependency_values if value is not None] == result.dataframe[
            "first"
        ].tolist()

    def test_record_scope_uses_row_position_and_preserves_nulls(self) -> None:
        dataframe = pd.DataFrame({"identifier": ["same", "same", None]}, index=[1, 1, 1])
        plan = _plan(_target("identifier", EntityType.UNIQUE_IDENTIFIER))
        generator = _RecordingGenerator()

        result = _execute(dataframe, plan, generator)

        assert result.dataframe["identifier"].iloc[0] != result.dataframe["identifier"].iloc[1]
        assert pd.isna(result.dataframe["identifier"].iloc[2])
        assert result.generation_statistics.generated_replacement_count == 2
        assert result.column_statistics["identifier"].detected_entity_counts == {"unique_identifier": 2}

    def test_natural_attributes_may_repeat_across_independent_mappings(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada", "Grace"]})
        plan = _plan(_target("first_name", EntityType.FIRST_NAME))

        result = _execute(dataframe, plan, _ConstantGenerator())

        assert result.dataframe["first_name"].tolist() == ["Synthetic", "Synthetic"]
        assert result.generation_statistics.generated_replacement_count == 2

    def test_unchanged_candidate_is_resampled_with_a_new_seed(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"]})
        plan = _plan(_target("first_name", EntityType.FIRST_NAME))
        generator = _OriginalThenReplacementGenerator()

        result = _execute(dataframe, plan, generator)

        assert result.dataframe["first_name"].tolist() == ["Synthetic"]
        assert len(generator.requests) == 2
        assert generator.requests[0].seed != generator.requests[1].seed

    def test_identifier_replacements_remain_collision_resistant(self) -> None:
        dataframe = pd.DataFrame({"identifier": ["USER-1", "USER-2"]})
        plan = _plan(_target("identifier", EntityType.UNIQUE_IDENTIFIER))

        with pytest.raises(GenerationError, match="could not generate a distinct value"):
            _execute(dataframe, plan, _ConstantGenerator())

    def test_group_scope_snapshots_replaced_group_keys_and_reports_dependency_drift(self) -> None:
        dataframe = pd.DataFrame(
            {
                "patient": ["A", "A", "B"],
                "organization": ["First Org", "Different Org", "Third Org"],
                "email": ["same@example.com"] * 3,
            },
            index=[4, 4, 4],
        )
        plan = _plan(
            _target("patient", EntityType.UNIQUE_IDENTIFIER),
            _target("email", EntityType.EMAIL, ("organization", EntityType.ORGANIZATION)),
        )
        generator = _RecordingGenerator()

        result = _execute(dataframe, plan, generator, group_column="patient")

        assert result.dataframe["patient"].iloc[0] == result.dataframe["patient"].iloc[1]
        assert result.dataframe["patient"].iloc[0] != result.dataframe["patient"].iloc[2]
        assert result.dataframe["email"].iloc[0] == result.dataframe["email"].iloc[1]
        assert result.dataframe["email"].iloc[0] != result.dataframe["email"].iloc[2]
        assert len(result.dependency_drifts) == 1
        assert result.dependency_drifts[0].as_log_extra() == {
            "target_column": "email",
            "conditioner_entity_types": ["organization"],
            "conflict_count": 1,
        }
        assert result.generation_statistics.generated_replacement_count == 4

    def test_free_text_target_fails_before_mutating_the_input(self) -> None:
        dataframe = pd.DataFrame({"notes": ["Ada called"]})
        original = dataframe.copy(deep=True)
        plan = _plan(_target("notes", EntityType.FREE_TEXT))

        with pytest.raises(GenerationError, match="detector and span-resolution"):
            _execute(dataframe, plan, _RecordingGenerator())

        pd.testing.assert_frame_equal(dataframe, original)


@pytest.mark.unit
class TestResolveBaseSeed:
    def test_precedence_is_explicit_then_environment_then_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERSON_RANDOM_SEED", "17")

        assert resolve_base_seed(9) == 9
        assert resolve_base_seed(None) == 17
        monkeypatch.delenv("PERSON_RANDOM_SEED")
        assert resolve_base_seed(None) == 42

    def test_invalid_environment_seed_is_a_parameter_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERSON_RANDOM_SEED", "not-an-integer")

        with pytest.raises(ParameterError, match="PERSON_RANDOM_SEED must be an integer"):
            resolve_base_seed(None)
