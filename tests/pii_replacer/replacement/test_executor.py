# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    ConditioningColumn,
    DataToSamplerValueMapping,
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiReplacementSettings,
    PiiSamplerBackend,
    PiiSamplerConfig,
)
from nemo_safe_synthesizer.errors import DataError, GenerationError, ParameterError
from nemo_safe_synthesizer.pii_replacer.replacement.compiler import compile_plan
from nemo_safe_synthesizer.pii_replacer.replacement.executor import (
    ReplacementExecutionResult,
    StructuredReplacementExecutor,
    resolve_base_seed,
)
from nemo_safe_synthesizer.pii_replacer.replacement.generation import ReplacementGenerationRequest
from nemo_safe_synthesizer.pii_replacer.replacement.generators import FakerReplacementGenerator
from nemo_safe_synthesizer.pii_replacer.replacement.types import (
    DetectedSpan,
    DetectionCell,
    DetectionCellId,
    FreeTextDetections,
    ProtectedSpan,
)


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


class _ParameterErrorGenerator(_RecordingGenerator):
    def generate(self, request: ReplacementGenerationRequest) -> str:
        raise ParameterError("the configured sampler cannot produce this value")


class _SwapGenerator(_RecordingGenerator):
    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        return {"1.5": "2.5", "2.5": "1.5"}[request.original_value]


class _DerivedEmailGenerator(_RecordingGenerator):
    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        return {"ana.ruiz@acme.com": "zoe.park@acme.com", "ben.cole@acme.com": "ana.ruiz@acme.com"}[
            request.original_value
        ]


class _EchoGenerator(_RecordingGenerator):
    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        return request.original_value


def _warning_counts(caplog: pytest.LogCaptureFixture, field: str) -> list[object]:
    return [getattr(record, field) for record in caplog.records if hasattr(record, field)]


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
    def __init__(self, spans: list[DetectedSpan], protected: Sequence[ProtectedSpan] = ()) -> None:
        self.spans = spans
        self.protected = tuple(protected)
        self.cells: list[DetectionCell] = []

    def detect(self, cells: Sequence[DetectionCell]) -> FreeTextDetections:
        self.cells.extend(cells)
        return FreeTextDetections(tuple(self.spans), self.protected)


def _span(
    row_position: int, column_name: str, text: str, value: str, entity_type: EntityType, occurrence: int = 0
) -> DetectedSpan:
    """Build a GLiNER span for the ``occurrence``-th appearance of ``value`` in ``text``."""
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(value, start + 1)
    return DetectedSpan(
        DetectionCellId(row_position, column_name), start, start + len(value), entity_type, "gliner", 0.95
    )


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


def _plan(
    *targets: PiiColumnPlan,
    data_to_sampler_value_mapping: DataToSamplerValueMapping | None = None,
) -> PiiReplacementPlan:
    return PiiReplacementPlan(
        columns_to_replace=list(targets),
        data_to_sampler_value_mapping=data_to_sampler_value_mapping or {},
    )


def _execute(
    dataframe: pd.DataFrame,
    plan: PiiReplacementPlan,
    generator: _RecordingGenerator,
    *,
    group_column: str | None = None,
    free_text_detector: _StaticDetector | None = None,
    capture_replacement_map: bool = False,
    find_and_replace: bool = False,
) -> ReplacementExecutionResult:
    return StructuredReplacementExecutor(
        plan,
        generator,
        group_column=group_column,
        base_seed=42,
        free_text_detector=free_text_detector,
        capture_replacement_map=capture_replacement_map,
        find_and_replace=find_and_replace,
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

    def test_resolves_sampler_values_from_the_dependency_source_column(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["Woman"]})
        plan = _plan(
            _target("first_name", EntityType.FIRST_NAME, ("sex", EntityType.GENDER)),
            data_to_sampler_value_mapping={"sex": {"Woman": ["female"]}},
        )
        generator = _RecordingGenerator()

        _execute(dataframe, plan, generator)

        assert generator.requests[0].resolved_dependency_values == ((EntityType.GENDER, ("female",)),)

    def test_matches_mapping_keys_for_non_string_dependency_values(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada", "Grace"], "sex": [1.0, 2.0]})
        plan = _plan(
            _target("first_name", EntityType.FIRST_NAME, ("sex", EntityType.GENDER)),
            data_to_sampler_value_mapping={"sex": {"1.0": ["female"], "2.0": ["male"]}},
        )
        generator = _RecordingGenerator()

        _execute(dataframe, plan, generator)

        assert [request.resolved_dependency_values for request in generator.requests] == [
            ((EntityType.GENDER, ("female",)),),
            ((EntityType.GENDER, ("male",)),),
        ]

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
        dataframe = pd.DataFrame({"first_name": ["Ada", "Ada", None]}, index=[1, 1, 1])
        plan = _plan(_target("first_name", EntityType.FIRST_NAME))
        generator = _RecordingGenerator()

        result = _execute(dataframe, plan, generator)

        assert result.dataframe["first_name"].iloc[0] != result.dataframe["first_name"].iloc[1]
        assert pd.isna(result.dataframe["first_name"].iloc[2])
        assert result.generation_statistics.generated_replacement_count == 2
        assert result.column_statistics["first_name"].detected_entity_counts == {"first_name": 2}

    def test_identifier_replacements_repeat_with_a_warning_once_the_format_runs_out(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        # 100 rows share 5 two-digit IDs. Each row is replaced independently, and only 95
        # two-digit values are not originals, so the last 5 replacements repeat earlier ones.
        dataframe = pd.DataFrame({"hospitalid": [f"{10 + row % 5}" for row in range(100)]})
        plan = _plan(_target("hospitalid", EntityType.UNIQUE_IDENTIFIER))

        result = _execute(dataframe, plan, _RecordingGenerator())

        replaced = result.dataframe["hospitalid"].tolist()
        assert len(set(replaced)) == 95
        assert not set(replaced) & {"10", "11", "12", "13", "14"}
        repeats = [record for record in caplog.records if "format has no unused values left" in record.getMessage()]
        assert [getattr(record, "repeated_count", None) for record in repeats] == [5]

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

    def test_identifier_replacements_avoid_repeats_before_accepting_one(self) -> None:
        dataframe = pd.DataFrame({"identifier": ["USER-1", "USER-2"]})
        plan = _plan(_target("identifier", EntityType.UNIQUE_IDENTIFIER))
        generator = _ConstantGenerator()

        result = _execute(dataframe, plan, generator)

        # The second row retries every attempt for an unused value before repeating "Synthetic".
        assert result.dataframe["identifier"].tolist() == ["Synthetic", "Synthetic"]
        assert len(generator.requests) == 11

    def test_group_scope_reuses_name_replacements_within_a_group_only(self) -> None:
        dataframe = pd.DataFrame({"patient": ["A", "A", "B"], "first_name": ["Ada", "Ada", "Ada"]})
        plan = _plan(_target("first_name", EntityType.FIRST_NAME))

        result = _execute(dataframe, plan, _RecordingGenerator(), group_column="patient")

        replaced = result.dataframe["first_name"].tolist()
        assert replaced[0] == replaced[1]
        assert replaced[0] != replaced[2]

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
        assert result.generation_statistics.dependency_drift_count == 1

    def test_group_scope_rejects_rows_without_a_group_identifier(self) -> None:
        dataframe = pd.DataFrame({"patient": ["A", None], "email": ["same@example.com"] * 2})
        plan = _plan(_target("email", EntityType.EMAIL))

        with pytest.raises(DataError, match="'patient' has 1 missing value"):
            _execute(dataframe, plan, _RecordingGenerator(), group_column="patient")

    def test_dense_row_counter_identifiers_are_replaced_from_a_wider_range(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        dataframe = pd.DataFrame({"row": list(range(1000))})
        plan = _plan(_target("row", EntityType.UNIQUE_IDENTIFIER))

        result = _execute(dataframe, plan, _RecordingGenerator())

        replaced = result.dataframe["row"].tolist()
        assert len(set(replaced)) == 1000
        assert not set(replaced) & {str(value) for value in range(1000)}
        assert all(value.isdigit() and len(value) <= 4 for value in replaced)
        assert "Widened replacement identifiers" in caplog.text

    def test_patterned_identifiers_take_every_unused_mask_value(self) -> None:
        dataframe = pd.DataFrame({"user": [f"USER-{value}" for value in range(5)]})
        plan = _plan(PiiColumnPlan(column_name="user", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="USER-#"))

        result = _execute(dataframe, plan, _RecordingGenerator())

        assert sorted(result.dataframe["user"]) == [f"USER-{value}" for value in range(5, 10)]

    def test_small_mask_repeats_replacements_but_never_reuses_an_original(self) -> None:
        dataframe = pd.DataFrame({"user": [f"USER-{value}" for value in range(8)]})
        plan = _plan(PiiColumnPlan(column_name="user", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="USER-#"))

        result = _execute(dataframe, plan, _RecordingGenerator())

        assert set(result.dataframe["user"]) == {"USER-8", "USER-9"}

    def test_mask_whose_every_value_is_an_original_reuses_other_originals_with_a_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        originals = [f"USER-{value}" for value in range(10)]
        dataframe = pd.DataFrame({"user": originals})
        plan = _plan(PiiColumnPlan(column_name="user", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="USER-#"))

        result = _execute(dataframe, plan, _RecordingGenerator())

        replaced = result.dataframe["user"].tolist()
        assert all(new != old for new, old in zip(replaced, originals, strict=True))
        assert _warning_counts(caplog, "reused_original_count") == [10]

    def test_mask_that_can_only_produce_the_original_is_a_configuration_error(self) -> None:
        dataframe = pd.DataFrame({"user": ["USER-0"]})
        plan = _plan(PiiColumnPlan(column_name="user", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="USER-[0]"))

        with pytest.raises(ParameterError, match="cannot produce any value other than the original"):
            _execute(dataframe, plan, _RecordingGenerator())

    def test_float_identifiers_reuse_another_original_only_as_a_last_resort(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        dataframe = pd.DataFrame({"id": [1.5, 2.5]})
        plan = _plan(_target("id", EntityType.UNIQUE_IDENTIFIER))

        result = _execute(dataframe, plan, _SwapGenerator())

        assert result.dataframe["id"].tolist() == ["2.5", "1.5"]
        assert _warning_counts(caplog, "reused_original_count") == [2]

    def test_derived_email_equal_to_another_rows_original_is_kept_with_a_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        # Ben's replacement email is fully determined by his replaced name, and it is Ana's real address.
        dataframe = pd.DataFrame({"email": ["ana.ruiz@acme.com", "ben.cole@acme.com"]})
        plan = _plan(_target("email", EntityType.EMAIL))

        result = _execute(dataframe, plan, _DerivedEmailGenerator())

        assert result.dataframe["email"].tolist() == ["zoe.park@acme.com", "ana.ruiz@acme.com"]
        assert _warning_counts(caplog, "reused_original_count") == [1]

    def test_unpatterned_email_is_built_from_the_replaced_names_it_depends_on(self) -> None:
        # Mixed local-part formats leave the planned email without a pattern; the dependency must still apply.
        dataframe = pd.DataFrame(
            {
                "first_name": ["Brenda", "Bobby", "Patrick"],
                "last_name": ["Rivera", "Hughes", "Adams"],
                "email": ["brenda.rivera@yahoo.com", "hughes_b@gmail.com", "padams@hotmail.com"],
            }
        )
        plan = _plan(
            _target(
                "email", EntityType.EMAIL, ("first_name", EntityType.FIRST_NAME), ("last_name", EntityType.LAST_NAME)
            ),
            _target("first_name", EntityType.FIRST_NAME),
            _target("last_name", EntityType.LAST_NAME),
        )
        generator = FakerReplacementGenerator(
            settings=PiiReplacementSettings(),
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
        )

        replaced = (
            StructuredReplacementExecutor(plan, generator, group_column=None, base_seed=42).execute(dataframe).dataframe
        )

        for first, last, domain, email in zip(
            replaced["first_name"],
            replaced["last_name"],
            ["yahoo.com", "gmail.com", "hotmail.com"],
            replaced["email"],
            strict=True,
        ):
            assert re.fullmatch(
                rf"{re.escape(first.lower())}\.{re.escape(last.lower())}\d{{2}}@{re.escape(domain)}", email
            )
        assert not set(replaced["first_name"]) & set(dataframe["first_name"])

    def test_replacement_equal_to_its_own_original_still_fails(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Jeremy"]})
        plan = _plan(_target("first_name", EntityType.FIRST_NAME))

        with pytest.raises(GenerationError, match="differs from the original"):
            _execute(dataframe, plan, _EchoGenerator())

    def test_free_text_group_scope_rejects_rows_without_a_group_identifier(self) -> None:
        dataframe = pd.DataFrame({"patient": ["A", None], "notes": ["Ada called", "Ada called"]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector([])

        with pytest.raises(DataError, match="'patient' has 1 missing value"):
            _execute(dataframe, plan, _RecordingGenerator(), group_column="patient", free_text_detector=detector)
        assert detector.cells == []

    def test_project_errors_from_the_generator_are_not_rewrapped(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"]})
        plan = _plan(_target("first_name", EntityType.FIRST_NAME))

        with pytest.raises(ParameterError, match="configured sampler cannot"):
            _execute(dataframe, plan, _ParameterErrorGenerator())

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

    def test_free_text_identifier_reuses_the_structured_unique_identifier_replacement(self) -> None:
        dataframe = pd.DataFrame({"customer_id": ["cust-00412"], "notes": ["Ticket for cust-00412"]})
        plan = _plan(_target("customer_id", EntityType.UNIQUE_IDENTIFIER), _target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector(
            [_span(0, "notes", "Ticket for cust-00412", "cust-00412", EntityType.UNIQUE_IDENTIFIER)]
        )
        generator = _RecordingGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        assert result.dataframe.at[0, "notes"] == f"Ticket for {result.dataframe.at[0, 'customer_id']}"
        assert len(generator.requests) == 1

    def test_free_text_identifier_reuses_an_allocated_integer_replacement(self) -> None:
        dataframe = pd.DataFrame({"api_key": [12345, 67890], "notes": ["Key 12345 issued", "No key"]})
        plan = _plan(_target("api_key", EntityType.API_KEY), _target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector([DetectedSpan(DetectionCellId(0, "notes"), 4, 9, EntityType.API_KEY, "gliner", 0.9)])
        generator = _RecordingGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        assert result.dataframe.at[0, "notes"] == f"Key {result.dataframe.at[0, 'api_key']} issued"
        assert generator.requests == []

    def test_reused_free_text_replacement_takes_each_occurrence_letter_case(self) -> None:
        dataframe = pd.DataFrame({"notes": ["Ada, ADA, ada"]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector(
            [
                DetectedSpan(DetectionCellId(0, "notes"), start, start + 3, EntityType.FIRST_NAME, "gliner", 0.9)
                for start in (0, 5, 10)
            ]
        )
        generator = _RecordingGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        first, shouted, quiet = str(result.dataframe.at[0, "notes"]).split(", ")
        assert shouted == first.upper()
        assert quiet == first.lower()
        assert len(generator.requests) == 1

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

    @pytest.mark.parametrize(
        "text",
        [
            "Kenneth Gonzales grew up in Tennessee. Kenneth enjoys skiing.",
            "Kenneth enjoys skiing. Kenneth Gonzales grew up.",
        ],
    )
    def test_free_text_full_name_and_bare_first_name_share_a_replacement_in_either_order(self, text: str) -> None:
        dataframe = pd.DataFrame({"notes": [text]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        bare_occurrence = 1 if text.startswith("Kenneth Gonzales") else 0
        detector = _StaticDetector(
            [
                _span(0, "notes", text, "Kenneth Gonzales", EntityType.FULL_NAME),
                _span(0, "notes", text, "Kenneth", EntityType.FIRST_NAME, occurrence=bare_occurrence),
            ]
        )
        generator = _RecordingGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector, capture_replacement_map=True)

        assert result.replacement_map is not None
        replaced = {record.entity_type: record.replacement_value for record in result.replacement_map.free_text}
        assert replaced[EntityType.FULL_NAME].split(" ")[0] == replaced[EntityType.FIRST_NAME]
        assert [request.entity_type for request in generator.requests] == [EntityType.FIRST_NAME, EntityType.LAST_NAME]

    def test_free_text_name_parts_are_shared_across_rows_in_group_scope(self) -> None:
        dataframe = pd.DataFrame({"patient": ["A", "A"], "notes": ["Kenneth Gonzales visited", "Gonzales called"]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector(
            [
                _span(0, "notes", "Kenneth Gonzales visited", "Kenneth Gonzales", EntityType.FULL_NAME),
                _span(1, "notes", "Gonzales called", "Gonzales", EntityType.LAST_NAME),
            ]
        )

        result = _execute(
            dataframe,
            plan,
            _RecordingGenerator(),
            group_column="patient",
            free_text_detector=detector,
            capture_replacement_map=True,
        )

        assert result.replacement_map is not None
        full, last = result.replacement_map.free_text
        assert full.replacement_value.split(" ")[1] == last.replacement_value

    def test_free_text_three_word_name_shares_its_middle_name(self) -> None:
        text = "Mary Ann Smith said Ann agreed"
        dataframe = pd.DataFrame({"notes": [text]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector(
            [
                _span(0, "notes", text, "Mary Ann Smith", EntityType.FULL_NAME),
                _span(0, "notes", text, "Ann", EntityType.MIDDLE_NAME, occurrence=1),
            ]
        )

        result = _execute(
            dataframe, plan, _RecordingGenerator(), free_text_detector=detector, capture_replacement_map=True
        )

        assert result.replacement_map is not None
        full, middle = result.replacement_map.free_text
        assert full.replacement_value.split(" ")[1] == middle.replacement_value

    def test_free_text_full_name_with_a_title_is_generated_whole(self) -> None:
        text = "Dr. Kenneth Gonzales called"
        dataframe = pd.DataFrame({"notes": [text]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector([_span(0, "notes", text, "Dr. Kenneth Gonzales", EntityType.FULL_NAME)])
        generator = _RecordingGenerator()

        _execute(dataframe, plan, generator, free_text_detector=detector)

        assert [request.entity_type for request in generator.requests] == [EntityType.FULL_NAME]

    def test_detected_first_name_reuses_an_unpatterned_structured_full_name(self) -> None:
        dataframe = pd.DataFrame({"full_name": ["Kenneth Gonzales"], "notes": ["Kenneth called"]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT), _target("full_name", EntityType.FULL_NAME))
        detector = _StaticDetector([_span(0, "notes", "Kenneth called", "Kenneth", EntityType.FIRST_NAME)])
        generator = _CompositeNameGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        assert result.dataframe["notes"].iloc[0] == "Grace called"
        assert len(generator.requests) == 1

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
class TestKnownValueMatching:
    @pytest.mark.parametrize("find_and_replace", [True, False])
    def test_structured_value_the_detector_missed_is_replaced_only_with_find_and_replace(
        self, find_and_replace: bool
    ) -> None:
        dataframe = pd.DataFrame({"first_name": ["Kenneth"], "notes": ["kenneth called back"]})
        plan = _plan(_target("first_name", EntityType.FIRST_NAME), _target("notes", EntityType.FREE_TEXT))

        result = _execute(
            dataframe,
            plan,
            _RecordingGenerator(),
            free_text_detector=_StaticDetector([]),
            capture_replacement_map=True,
            find_and_replace=find_and_replace,
        )

        assert result.replacement_map is not None
        if not find_and_replace:
            assert result.dataframe.at[0, "notes"] == "kenneth called back"
            assert result.replacement_map.free_text == ()
            return
        assert result.dataframe.at[0, "notes"] == f"{result.dataframe.at[0, 'first_name']} called back"
        (record,) = result.replacement_map.free_text
        assert (record.entity_type, record.detection_source, record.score) == (
            EntityType.FIRST_NAME,
            "structured_value",
            None,
        )

    def test_detected_values_replace_missed_occurrences_in_the_same_record_only(self) -> None:
        dataframe = pd.DataFrame(
            {
                "notes": ["Kenneth Gonzales visited", "Gonzales called"],
                "summary": ["Gonzales paid; Kenneth Gonzales left", "Kenneth left"],
            }
        )
        plan = _plan(_target("notes", EntityType.FREE_TEXT), _target("summary", EntityType.FREE_TEXT))
        detector = _StaticDetector(
            [_span(0, "notes", "Kenneth Gonzales visited", "Kenneth Gonzales", EntityType.FULL_NAME)]
        )

        result = _execute(
            dataframe,
            plan,
            _RecordingGenerator(),
            free_text_detector=detector,
            capture_replacement_map=True,
            find_and_replace=True,
        )

        full = str(result.dataframe.at[0, "notes"]).removesuffix(" visited")
        last = full.split(" ")[1]
        assert result.dataframe.at[0, "summary"] == f"{last} paid; {full} left"
        assert result.dataframe.iloc[1].tolist() == ["Gonzales called", "Kenneth left"]
        assert result.replacement_map is not None
        assert [(record.entity_type, record.detection_source) for record in result.replacement_map.free_text] == [
            (EntityType.FULL_NAME, "gliner"),
            (EntityType.LAST_NAME, "detected_value"),
            (EntityType.FULL_NAME, "detected_value"),
        ]

    def test_group_scope_searches_values_known_anywhere_in_the_group(self) -> None:
        dataframe = pd.DataFrame(
            {
                "patient": ["A", "A", "B"],
                "first_name": ["Kenneth", "Ada", "Ada"],
                "notes": ["", "Kenneth called", "Kenneth called"],
            }
        )
        plan = _plan(_target("first_name", EntityType.FIRST_NAME), _target("notes", EntityType.FREE_TEXT))

        result = _execute(
            dataframe,
            plan,
            _RecordingGenerator(),
            group_column="patient",
            free_text_detector=_StaticDetector([]),
            find_and_replace=True,
        )

        assert result.dataframe["notes"].tolist() == [
            "",
            f"{result.dataframe.at[0, 'first_name']} called",
            "Kenneth called",
        ]

    def test_search_matches_whole_words_and_skips_short_and_common_values(self) -> None:
        text = "Will will call Al. Gonzales's file, Gonzaless, GONZALES"
        dataframe = pd.DataFrame(
            {"first_name": ["Will"], "middle_name": ["Al"], "last_name": ["Gonzales"], "notes": [text]}
        )
        plan = _plan(
            _target("first_name", EntityType.FIRST_NAME),
            _target("middle_name", EntityType.MIDDLE_NAME),
            _target("last_name", EntityType.LAST_NAME),
            _target("notes", EntityType.FREE_TEXT),
        )

        result = _execute(
            dataframe, plan, _RecordingGenerator(), free_text_detector=_StaticDetector([]), find_and_replace=True
        )

        last = str(result.dataframe.at[0, "last_name"])
        assert result.dataframe.at[0, "notes"] == f"Will will call Al. {last}'s file, Gonzaless, {last.upper()}"

    def test_search_skips_matches_inside_protected_spans(self) -> None:
        text = "Has Parkinson's disease. Parkinson called."
        dataframe = pd.DataFrame({"last_name": ["Parkinson"], "notes": [text]})
        plan = _plan(_target("last_name", EntityType.LAST_NAME), _target("notes", EntityType.FREE_TEXT))
        condition = ProtectedSpan(DetectionCellId(0, "notes"), text.index("Parkinson"), text.index("."))

        result = _execute(
            dataframe,
            plan,
            _RecordingGenerator(),
            free_text_detector=_StaticDetector([], protected=[condition]),
            find_and_replace=True,
        )

        assert (
            result.dataframe.at[0, "notes"] == f"Has Parkinson's disease. {result.dataframe.at[0, 'last_name']} called."
        )

    def test_longer_match_beats_a_detected_span_and_equal_length_keeps_the_detected_span(self) -> None:
        text = "Kenneth Gonzales met Jordan"
        dataframe = pd.DataFrame({"full_name": ["Kenneth Gonzales"], "first_name": ["Jordan"], "notes": [text]})
        plan = _plan(
            _target("full_name", EntityType.FULL_NAME),
            _target("first_name", EntityType.FIRST_NAME),
            _target("notes", EntityType.FREE_TEXT),
        )
        detector = _StaticDetector(
            [
                _span(0, "notes", text, "Kenneth", EntityType.FIRST_NAME),
                _span(0, "notes", text, "Jordan", EntityType.LAST_NAME),
            ]
        )

        result = _execute(
            dataframe,
            plan,
            _RecordingGenerator(),
            free_text_detector=detector,
            capture_replacement_map=True,
            find_and_replace=True,
        )

        assert result.replacement_map is not None
        full, jordan = result.replacement_map.free_text
        assert (full.original_value, full.entity_type, full.detection_source) == (
            "Kenneth Gonzales",
            EntityType.FULL_NAME,
            "structured_value",
        )
        assert full.replacement_value == result.dataframe.at[0, "full_name"]
        assert (jordan.entity_type, jordan.detection_source) == (EntityType.LAST_NAME, "gliner")


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
