# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

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
from nemo_safe_synthesizer.pii_replacer.replacement.types import DetectedSpan, DetectionCell, DetectionCellId


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


class _CompositeNameGenerator(_RecordingGenerator):
    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        if request.entity_type is EntityType.FULL_NAME:
            return "Grace Hopper"
        return super().generate(request)


class _CompositeAddressGenerator(_RecordingGenerator):
    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        if request.entity_type is EntityType.STREET_ADDRESS:
            return "9 New Road, Boston, MA"
        return super().generate(request)


class _StaticDetector:
    def __init__(self, spans: list[DetectedSpan]) -> None:
        self.spans = spans
        self.cells: list[DetectionCell] = []

    def detect(self, cells: Sequence[DetectionCell]) -> tuple[DetectedSpan, ...]:
        self.cells.extend(cells)
        return tuple(self.spans)


def _target(
    column_name: str,
    entity_type: EntityType,
    *dependencies: tuple[str, EntityType | None],
    pattern: str | None = None,
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
        pattern=pattern,
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
    free_text_detector: _StaticDetector | None = None,
    capture_replacement_map: bool = False,
) -> ReplacementExecutionResult:
    return StructuredReplacementExecutor(
        plan,
        generator,
        group_column=group_column,
        base_seed=42,
        dependency_value_mappings=dependency_value_mappings,
        free_text_detector=free_text_detector,
        capture_replacement_map=capture_replacement_map,
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
    def test_replacement_map_is_opt_in(self) -> None:
        dataframe = pd.DataFrame({"identifier": ["USER-1"]})
        plan = _plan(_target("identifier", EntityType.UNIQUE_IDENTIFIER))

        default_result = _execute(dataframe, plan, _RecordingGenerator())
        captured_result = _execute(
            dataframe,
            plan,
            _RecordingGenerator(),
            capture_replacement_map=True,
        )

        assert default_result.replacement_map is None
        assert captured_result.replacement_map is not None
        record = captured_result.replacement_map.structured[0]
        assert record.row_position == 0
        assert record.column_name == "identifier"
        assert record.entity_type is EntityType.UNIQUE_IDENTIFIER
        assert record.scope == "record"
        assert record.original_value == "USER-1"
        assert record.replacement_value == captured_result.dataframe.at[0, "identifier"]
        assert "USER-1" not in repr(record)

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

    def test_free_text_replaces_only_accepted_spans_in_one_pass(self) -> None:
        dataframe = pd.DataFrame({"notes": ["Ada met Ada at ada@example.com"]})
        original = dataframe.copy(deep=True)
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector(
            [
                DetectedSpan(DetectionCellId(0, "notes"), 0, 3, EntityType.FIRST_NAME, "gliner", 0.9),
                DetectedSpan(DetectionCellId(0, "notes"), 15, 30, EntityType.EMAIL, "regex"),
            ]
        )
        generator = _RecordingGenerator()

        result = _execute(
            dataframe,
            plan,
            generator,
            free_text_detector=detector,
            capture_replacement_map=True,
        )

        pd.testing.assert_frame_equal(dataframe, original)
        assert result.dataframe["notes"].iloc[0].startswith("synthetic-first_name-")
        assert " met Ada at " in result.dataframe["notes"].iloc[0]
        assert "ada@example.com" not in result.dataframe["notes"].iloc[0]
        assert result.column_statistics["notes"].detected_entity_counts == {"first_name": 1, "email": 1}
        assert result.column_statistics["notes"].detected_entity_values == {
            "first_name": {"Ada"},
            "email": {"ada@example.com"},
        }
        assert result.replacement_map is not None
        first, email = result.replacement_map.free_text
        assert first.model_dump(exclude={"original_value", "replacement_value"}) == {
            "row_position": 0,
            "column_name": "notes",
            "start": 0,
            "end": 3,
            "entity_type": EntityType.FIRST_NAME,
            "detection_source": "gliner",
            "score": 0.9,
            "scope": "record",
        }
        assert first.original_value == "Ada"
        assert first.replacement_value in str(result.dataframe.at[0, "notes"])
        assert email.start == 15
        assert email.end == 30
        assert email.detection_source == "regex"
        assert email.score is None

    def test_equal_detected_values_reuse_one_replacement_across_free_text_columns(self) -> None:
        dataframe = pd.DataFrame({"primary": ["Ada"], "secondary": ["Call Ada"]})
        plan = _plan(
            _target("primary", EntityType.FREE_TEXT),
            _target("secondary", EntityType.FREE_TEXT),
        )
        detector = _StaticDetector(
            [
                DetectedSpan(DetectionCellId(0, "primary"), 0, 3, EntityType.FIRST_NAME, "gliner", 0.9),
                DetectedSpan(DetectionCellId(0, "secondary"), 5, 8, EntityType.FIRST_NAME, "gliner", 0.9),
            ]
        )
        generator = _RecordingGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        replacement = result.dataframe["primary"].iloc[0]
        assert result.dataframe["secondary"].iloc[0] == f"Call {replacement}"
        assert len(generator.requests) == 1
        assert result.generation_statistics.generated_replacement_count == 1

    def test_independently_detected_span_reuses_an_exact_structured_mapping(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "notes": ["Ada met Grace"]})
        plan = _plan(
            _target("notes", EntityType.FREE_TEXT),
            _target("first_name", EntityType.FIRST_NAME),
        )
        detector = _StaticDetector(
            [DetectedSpan(DetectionCellId(0, "notes"), 0, 3, EntityType.FIRST_NAME, "gliner", 0.9)]
        )
        generator = _RecordingGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        replacement = result.dataframe["first_name"].iloc[0]
        assert result.dataframe["notes"].iloc[0] == f"{replacement} met Grace"
        assert len(generator.requests) == 1
        assert result.generation_statistics.generated_replacement_count == 1

    def test_detected_name_component_reuses_parent_mapping_from_validated_pattern(self) -> None:
        dataframe = pd.DataFrame({"full_name": ["John Smith"], "notes": ["Ask Smith to call"]})
        plan = _plan(
            _target("notes", EntityType.FREE_TEXT),
            _target("full_name", EntityType.FULL_NAME, pattern="{First} {Last}"),
        )
        detector = _StaticDetector(
            [DetectedSpan(DetectionCellId(0, "notes"), 4, 9, EntityType.LAST_NAME, "gliner", 0.9)]
        )
        generator = _CompositeNameGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        assert result.dataframe["full_name"].iloc[0] == "Grace Hopper"
        assert result.dataframe["notes"].iloc[0] == "Ask Hopper to call"
        assert len(generator.requests) == 1
        assert result.generation_statistics.generated_replacement_count == 1

    def test_detected_street_component_reuses_parent_with_validated_dependency_suffix(self) -> None:
        dataframe = pd.DataFrame(
            {
                "address": ["1 Main Street, Boston, MA"],
                "city": ["Boston"],
                "state": ["MA"],
                "notes": ["Ship to 1 Main Street tomorrow"],
            }
        )
        plan = _plan(
            _target("notes", EntityType.FREE_TEXT),
            _target(
                "address",
                EntityType.STREET_ADDRESS,
                ("city", EntityType.CITY),
                ("state", EntityType.STATE),
            ),
        )
        detector = _StaticDetector(
            [DetectedSpan(DetectionCellId(0, "notes"), 8, 21, EntityType.STREET_ADDRESS, "gliner", 0.9)]
        )
        generator = _CompositeAddressGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        assert result.dataframe["address"].iloc[0] == "9 New Road, Boston, MA"
        assert result.dataframe["notes"].iloc[0] == "Ship to 9 New Road tomorrow"
        assert len(generator.requests) == 1
        assert result.generation_statistics.generated_replacement_count == 1


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
