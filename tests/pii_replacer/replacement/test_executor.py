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
from nemo_safe_synthesizer.pii_replacer.replacement.value_checks import keeps_original_name_part


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


class _ScriptedNameGenerator(_RecordingGenerator):
    def __init__(self, *names: str) -> None:
        super().__init__()
        self._names = names

    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        return self._names[min(len(self.requests), len(self._names)) - 1]


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


class _NamePartGenerator(_RecordingGenerator):
    def __init__(
        self,
        *,
        first_name: str = "Grace",
        middle_name: str = "Mary",
        last_name: str = "Hopper",
        full_name: str = "Grace Hopper",
    ) -> None:
        super().__init__()
        self._names = {
            EntityType.FIRST_NAME: first_name,
            EntityType.MIDDLE_NAME: middle_name,
            EntityType.LAST_NAME: last_name,
            EntityType.FULL_NAME: full_name,
        }

    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        return self._names[request.entity_type]


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
        requested = {cell.cell_id for cell in cells}
        return FreeTextDetections(
            tuple(span for span in self.spans if span.cell_id in requested),
            tuple(span for span in self.protected if span.cell_id in requested),
        )


class _StreetGenerator(_RecordingGenerator):
    """Return a bare street when the original locality is kept, otherwise a recognizable whole value."""

    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        return "9 New Road" if request.keep_original_locality else "Whole Replacement"


class _FailingDetector(_StaticDetector):
    def detect(self, cells: Sequence[DetectionCell]) -> FreeTextDetections:
        raise GenerationError("GLiNER2 free-text PII model could not be loaded")


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

    def test_repeated_identifiers_only_need_to_differ_from_their_own_original(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        # 100 rows share 5 two-digit IDs, so the column is not all distinct and replacements may repeat.
        originals = [f"{10 + row % 5}" for row in range(100)]
        dataframe = pd.DataFrame({"hospitalid": originals})
        plan = _plan(_target("hospitalid", EntityType.UNIQUE_IDENTIFIER))

        result = _execute(dataframe, plan, _RecordingGenerator())

        replaced = result.dataframe["hospitalid"].tolist()
        assert all(new != old and len(new) <= 2 for new, old in zip(replaced, originals, strict=True))
        assert _warning_counts(caplog, "repeated_count") == []

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

    def test_candidate_equal_to_the_original_ignoring_case_is_resampled(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"]})
        plan = _plan(_target("first_name", EntityType.FIRST_NAME))
        generator = _ScriptedNameGenerator("ADA", "Kathleen")

        result = _execute(dataframe, plan, generator)

        assert result.dataframe["first_name"].tolist() == ["Kathleen"]
        assert len(generator.requests) == 2

    def test_identifier_replacements_avoid_repeats_before_accepting_one(self) -> None:
        dataframe = pd.DataFrame({"identifier": ["USER-1", "USER-2"]})
        plan = _plan(_target("identifier", EntityType.UNIQUE_IDENTIFIER))
        generator = _ConstantGenerator()

        result = _execute(dataframe, plan, generator)

        # The second row retries every attempt for an unused value before repeating "Synthetic",
        # written in capitals like the originals.
        assert result.dataframe["identifier"].tolist() == ["SYNTHETIC", "SYNTHETIC"]
        assert len(generator.requests) == 11

    def test_values_differing_only_in_letter_case_share_a_replacement_in_their_own_case(self) -> None:
        dataframe = pd.DataFrame({"patient": ["A", "A", "A", "B"], "first_name": ["Ada", "ADA", "ada", "Ada"]})
        plan = _plan(_target("first_name", EntityType.FIRST_NAME))
        generator = _ScriptedNameGenerator("Kathleen", "Grace")

        result = _execute(dataframe, plan, generator, group_column="patient")

        assert result.dataframe["first_name"].tolist() == ["Kathleen", "KATHLEEN", "kathleen", "Grace"]
        assert result.generation_statistics.generated_replacement_count == 2

    def test_group_scope_reuses_name_replacements_within_a_group_only(self) -> None:
        dataframe = pd.DataFrame({"patient": ["A", "A", "B"], "first_name": ["Ada", "Ada", "Ada"]})
        plan = _plan(_target("first_name", EntityType.FIRST_NAME))

        result = _execute(dataframe, plan, _RecordingGenerator(), group_column="patient")

        replaced = result.dataframe["first_name"].tolist()
        assert replaced[0] == replaced[1]
        assert replaced[0] != replaced[2]

    def test_column_statistics_report_per_column_fallbacks_without_serializing_originals(self) -> None:
        dataframe = pd.DataFrame(
            {
                "patient": ["A", "A", "B"],
                "organization": ["First Org", "Different Org", "Third Org"],
                "email": ["same@example.com"] * 3,
                "ssn": ["111-11-1111", "222-22-2222", "333-33-3333"],
            }
        )
        plan = _plan(
            _target("email", EntityType.EMAIL, ("organization", EntityType.ORGANIZATION)),
            _target("ssn", EntityType.SSN),
        )

        result = _execute(dataframe, plan, _ConstantGenerator(), group_column="patient")

        email, ssn = result.column_statistics["email"], result.column_statistics["ssn"]
        assert email.dependency_drift_count == 1
        assert ssn.repeated_replacement_count == 2
        assert ssn.detected_entity_values == {"ssn": {"111-11-1111", "222-22-2222", "333-33-3333"}}
        assert "detected_entity_values" not in ssn.model_dump()
        assert "111-11-1111" not in repr(ssn)

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

    def test_a_distinct_numeric_key_is_replaced_from_a_range_wide_enough_to_stay_distinct(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        dataframe = pd.DataFrame({"row": list(range(1000))})
        plan = _plan(_target("row", EntityType.UNIQUE_IDENTIFIER))

        result = _execute(dataframe, plan, _RecordingGenerator())

        replaced = result.dataframe["row"].tolist()
        assert len(set(replaced)) == 1000
        assert all(new != str(old) for new, old in zip(replaced, range(1000), strict=True))
        assert all(value.isdigit() and len(value) <= 4 for value in replaced)
        assert "Widened replacement identifiers" in caplog.text

    def test_patterned_distinct_identifiers_get_distinct_mask_values(self) -> None:
        originals = [f"USER-{value}" for value in range(5)]
        dataframe = pd.DataFrame({"user": originals})
        plan = _plan(PiiColumnPlan(column_name="user", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="USER-#"))

        result = _execute(dataframe, plan, _RecordingGenerator())

        replaced = result.dataframe["user"].tolist()
        assert len(set(replaced)) == 5
        assert all(re.fullmatch(r"USER-\d", new) and new != old for new, old in zip(replaced, originals, strict=True))

    def test_mask_with_room_keeps_a_distinct_column_distinct(self) -> None:
        originals = [f"USER-{value}" for value in range(8)]
        dataframe = pd.DataFrame({"user": originals})
        plan = _plan(PiiColumnPlan(column_name="user", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="USER-#"))

        result = _execute(dataframe, plan, _RecordingGenerator())

        replaced = result.dataframe["user"].tolist()
        assert len(set(replaced)) == 8
        assert all(new != old for new, old in zip(replaced, originals, strict=True))

    def test_mask_with_no_spare_values_never_returns_the_own_original(self) -> None:
        originals = [f"USER-{value}" for value in range(10)]
        dataframe = pd.DataFrame({"user": originals})
        plan = _plan(PiiColumnPlan(column_name="user", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="USER-#"))

        result = _execute(dataframe, plan, _RecordingGenerator())

        replaced = result.dataframe["user"].tolist()
        assert all(new != old for new, old in zip(replaced, originals, strict=True))

    def test_mask_that_can_only_produce_the_original_is_a_configuration_error(self) -> None:
        dataframe = pd.DataFrame({"user": ["USER-0"]})
        plan = _plan(PiiColumnPlan(column_name="user", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="USER-[0]"))

        with pytest.raises(ParameterError, match="cannot produce any value other than the original"):
            _execute(dataframe, plan, _RecordingGenerator())

    def test_an_identifier_may_take_another_rows_original(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)
        dataframe = pd.DataFrame({"id": [1.5, 2.5]})
        plan = _plan(_target("id", EntityType.UNIQUE_IDENTIFIER))

        result = _execute(dataframe, plan, _SwapGenerator())

        assert result.dataframe["id"].tolist() == ["2.5", "1.5"]
        assert caplog.records == []

    def test_derived_email_equal_to_another_rows_original_is_accepted(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)
        # Ben's replacement email is fully determined by his replaced name, and it is Ana's real address.
        dataframe = pd.DataFrame({"email": ["ana.ruiz@acme.com", "ben.cole@acme.com"]})
        plan = _plan(_target("email", EntityType.EMAIL))

        result = _execute(dataframe, plan, _DerivedEmailGenerator())

        assert result.dataframe["email"].tolist() == ["zoe.park@acme.com", "ana.ruiz@acme.com"]
        assert caplog.records == []

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
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
            locale="en_US",
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

    def test_full_name_replacement_that_keeps_the_original_first_name_is_retried(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        dataframe = pd.DataFrame({"full_name": ["John Smith"]})

        result = _execute(
            dataframe,
            _plan(_target("full_name", EntityType.FULL_NAME)),
            _ScriptedNameGenerator("John Brown", "Mary Jones"),
        )

        assert result.dataframe["full_name"].tolist() == ["Mary Jones"]
        assert _warning_counts(caplog, "kept_name_part_count") == []

    def test_full_name_that_can_only_keep_a_name_part_is_used_with_a_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A dependency can fix the first name, as when it is copied from a linked first-name column.
        caplog.set_level(logging.WARNING)
        dataframe = pd.DataFrame({"full_name": ["John Smith"]})

        result = _execute(
            dataframe, _plan(_target("full_name", EntityType.FULL_NAME)), _ScriptedNameGenerator("John Brown")
        )

        assert result.dataframe["full_name"].tolist() == ["John Brown"]
        assert _warning_counts(caplog, "kept_name_part_count") == [1]

    @pytest.mark.parametrize(
        ("original", "candidate", "pattern", "keeps"),
        [
            ("John Smith", "John Brown", None, True),
            ("John Smith", "Mary Smith", None, True),
            ("John Smith", "Mary Brown", None, False),
            ("Smith, John", "Brown, John", "{Last}, {First}", True),
            ("Dr. Ana Ruiz", "Dr. Maria Lopez", "Dr. {First} {Last}", False),
            ("J. Smith", "J. Brown", None, False),
        ],
    )
    def test_keeps_original_name_part_compares_first_and_last_names(
        self,
        original: str,
        candidate: str,
        pattern: str | None,
        keeps: bool,
    ) -> None:
        assert keeps_original_name_part(original, candidate, pattern) is keeps

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

    def test_free_text_replacement_equal_to_the_original_ignoring_case_is_resampled(self) -> None:
        dataframe = pd.DataFrame({"notes": ["ada called"]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector([_span(0, "notes", "ada called", "ada", EntityType.FIRST_NAME)])
        generator = _ScriptedNameGenerator("Ada", "Kathleen")

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        assert result.dataframe.at[0, "notes"] == "kathleen called"

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

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Dr Ada Lovelace", "Dr Grace Hopper"),
            ("dr. Ada Lovelace", "dr. Grace Hopper"),
            ("Prof Ada Byron Lovelace", "Prof Grace Mary Hopper"),
            ("John Smith Jr", "Grace Hopper Jr"),
            ("Mr. John Smith III", "Mr. Grace Hopper III"),
        ],
    )
    def test_free_text_full_name_keeps_titles_and_suffixes(self, name: str, expected: str) -> None:
        text = f"{name} called"
        dataframe = pd.DataFrame({"notes": [text]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector([_span(0, "notes", text, name, EntityType.FULL_NAME)])

        result = _execute(dataframe, plan, _NamePartGenerator(), free_text_detector=detector)

        assert result.dataframe["notes"].iloc[0] == f"{expected} called"

    def test_free_text_full_name_with_a_suffix_shares_its_surname(self) -> None:
        text = "John Smith Jr called. Smith agreed."
        dataframe = pd.DataFrame({"notes": [text]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector(
            [
                _span(0, "notes", text, "John Smith Jr", EntityType.FULL_NAME),
                _span(0, "notes", text, "Smith", EntityType.LAST_NAME, occurrence=1),
            ]
        )
        generator = _NamePartGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        assert result.dataframe["notes"].iloc[0] == "Grace Hopper Jr called. Hopper agreed."
        assert [request.entity_type for request in generator.requests] == [EntityType.FIRST_NAME, EntityType.LAST_NAME]

    def test_structured_full_name_with_a_suffix_registers_its_surname(self) -> None:
        dataframe = pd.DataFrame({"full_name": ["John Smith Jr"], "notes": ["Ask Smith to call"]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT), _target("full_name", EntityType.FULL_NAME))
        detector = _StaticDetector([_span(0, "notes", "Ask Smith to call", "Smith", EntityType.LAST_NAME)])

        result = _execute(dataframe, plan, _CompositeNameGenerator(), free_text_detector=detector)

        assert result.dataframe["notes"].iloc[0] == "Ask Hopper to call"

    @pytest.mark.parametrize(("initial", "expected"), [("L", "H"), ("L.", "H."), ("l", "h")])
    def test_free_text_initial_takes_the_first_letter_of_the_matching_replaced_name(
        self, initial: str, expected: str
    ) -> None:
        text = f"Ada Lovelace called. Ada {initial} agreed."
        dataframe = pd.DataFrame({"notes": [text]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector(
            [
                _span(0, "notes", text, "Ada Lovelace", EntityType.FULL_NAME),
                # "Ada L" is also a prefix of "Ada Lovelace", so take its last occurrence.
                _span(0, "notes", text, f"Ada {initial}", EntityType.FULL_NAME, text.count(f"Ada {initial}") - 1),
            ]
        )
        generator = _NamePartGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        assert result.dataframe["notes"].iloc[0] == f"Grace Hopper called. Grace {expected} agreed."
        assert [request.entity_type for request in generator.requests] == [EntityType.FIRST_NAME, EntityType.LAST_NAME]

    def test_free_text_initial_without_a_matching_name_uses_a_generated_name_part(self) -> None:
        text = "Ada K. Lovelace called"
        dataframe = pd.DataFrame({"notes": [text]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector([_span(0, "notes", text, "Ada K. Lovelace", EntityType.FULL_NAME)])
        generator = _NamePartGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        assert result.dataframe["notes"].iloc[0] == "Grace M. Hopper called"
        assert [(request.entity_type, request.original_value) for request in generator.requests] == [
            (EntityType.FIRST_NAME, "Ada"),
            (EntityType.MIDDLE_NAME, "K"),
            (EntityType.LAST_NAME, "Lovelace"),
        ]

    def test_free_text_initial_always_changes_its_letter(self) -> None:
        text = "Ada Lovelace called. Ada L. agreed."
        dataframe = pd.DataFrame({"notes": [text]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT))
        detector = _StaticDetector(
            [
                _span(0, "notes", text, "Ada Lovelace", EntityType.FULL_NAME),
                _span(0, "notes", text, "Ada L.", EntityType.FULL_NAME),
            ]
        )

        result = _execute(dataframe, plan, _NamePartGenerator(last_name="Lopez"), free_text_detector=detector)

        replaced = str(result.dataframe["notes"].iloc[0])
        assert re.fullmatch(r"Grace Lopez called\. Grace [A-KM-Z]\. agreed\.", replaced), replaced

    def test_structured_name_initial_is_not_registered(self) -> None:
        dataframe = pd.DataFrame({"full_name": ["Ada K Lovelace"], "notes": ["Ada and K called"]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT), _target("full_name", EntityType.FULL_NAME))
        detector = _StaticDetector(
            [
                _span(0, "notes", "Ada and K called", "Ada", EntityType.FIRST_NAME),
                _span(0, "notes", "Ada and K called", "K", EntityType.MIDDLE_NAME),
            ]
        )
        generator = _NamePartGenerator(full_name="Grace Mary Hopper", middle_name="Zoe")

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        # The all-capitals "K" gets its fresh replacement in capitals.
        assert result.dataframe["notes"].iloc[0] == "Grace and ZOE called"

    def test_detected_first_name_reuses_an_unpatterned_structured_full_name(self) -> None:
        dataframe = pd.DataFrame({"full_name": ["Kenneth Gonzales"], "notes": ["Kenneth called"]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT), _target("full_name", EntityType.FULL_NAME))
        detector = _StaticDetector([_span(0, "notes", "Kenneth called", "Kenneth", EntityType.FIRST_NAME)])
        generator = _CompositeNameGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        assert result.dataframe["notes"].iloc[0] == "Grace called"
        assert len(generator.requests) == 1


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
class TestStreetAddressLocality:
    def test_full_address_keeps_the_locality_after_the_detected_street(self) -> None:
        value = "489 Park Blvd, Mesa, IA 76237"
        dataframe = pd.DataFrame({"address": [value, value]})
        detector = _StaticDetector([_span(0, "address", value, "489 Park Blvd", EntityType.STREET_ADDRESS)])
        generator = _StreetGenerator()

        result = _execute(
            dataframe, _plan(_target("address", EntityType.STREET_ADDRESS)), generator, free_text_detector=detector
        )

        assert result.dataframe["address"].tolist() == ["9 New Road, Mesa, IA 76237"] * 2
        # Detection runs once per distinct value; every row with that value keeps its locality.
        assert [cell.text for cell in detector.cells] == [value]
        assert [request.original_value for request in generator.requests] == ["489 Park Blvd"] * 2
        assert all(request.keep_original_locality for request in generator.requests)

    def test_grouping_addresses_with_the_same_locality_get_distinct_streets(self) -> None:
        first, second = "489 Park Blvd, Mesa, IA 76237", "12 Elm St, Mesa, IA 76237"
        dataframe = pd.DataFrame({"address": [first, second]})
        detector = _StaticDetector(
            [
                _span(0, "address", first, "489 Park Blvd", EntityType.STREET_ADDRESS),
                _span(1, "address", second, "12 Elm St", EntityType.STREET_ADDRESS),
            ]
        )
        # The second group is first offered the street the first group already got.
        generator = _ScriptedNameGenerator("9 New Road", "9 New Road", "7 Oak Lane")

        result = _execute(
            dataframe,
            _plan(_target("address", EntityType.STREET_ADDRESS)),
            generator,
            group_column="address",
            free_text_detector=detector,
        )

        assert result.dataframe["address"].tolist() == ["9 New Road, Mesa, IA 76237", "7 Oak Lane, Mesa, IA 76237"]

    @pytest.mark.parametrize(
        ("value", "detected"),
        [
            pytest.param("489 Park Blvd, Mesa, IA 76237", [], id="no-span"),
            pytest.param(
                "Unit 5, 42 George Street, Sydney NSW 2000",
                [("42 George Street", EntityType.STREET_ADDRESS)],
                id="street-not-at-start",
            ),
            pytest.param(
                "1420 Fillmore St Apt 3B", [("1420 Fillmore St Apt 3B", EntityType.STREET_ADDRESS)], id="whole-value"
            ),
            pytest.param("489 Park Blvd, Mesa", [("489 Park Bl", EntityType.STREET_ADDRESS)], id="ends-mid-word"),
            pytest.param("489 Park Blvd ,", [("489 Park Blvd", EntityType.STREET_ADDRESS)], id="no-locality-text"),
            pytest.param(
                "489 Park Blvd, Mesa, call 202-555-0101",
                [("489 Park Blvd", EntityType.STREET_ADDRESS), ("202-555-0101", EntityType.PHONE_NUMBER)],
                id="another-entity",
            ),
        ],
    )
    def test_values_without_one_leading_street_span_are_replaced_whole(
        self,
        value: str,
        detected: list[tuple[str, EntityType]],
    ) -> None:
        dataframe = pd.DataFrame({"address": [value]})
        detector = _StaticDetector([_span(0, "address", value, text, entity_type) for text, entity_type in detected])
        generator = _StreetGenerator()

        result = _execute(
            dataframe, _plan(_target("address", EntityType.STREET_ADDRESS)), generator, free_text_detector=detector
        )

        assert result.dataframe["address"].iloc[0] == "Whole Replacement"
        assert generator.requests[0].original_value == value

    def test_detected_street_in_free_text_reuses_the_structured_street_without_linked_columns(self) -> None:
        address = "489 Park Blvd, Mesa, IA 76237"
        notes = "Ship to 489 Park Blvd today"
        dataframe = pd.DataFrame({"address": [address], "notes": [notes]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT), _target("address", EntityType.STREET_ADDRESS))
        detector = _StaticDetector(
            [
                _span(0, "address", address, "489 Park Blvd", EntityType.STREET_ADDRESS),
                _span(0, "notes", notes, "489 Park Blvd", EntityType.STREET_ADDRESS),
            ]
        )
        generator = _StreetGenerator()

        result = _execute(dataframe, plan, generator, free_text_detector=detector)

        assert result.dataframe["address"].iloc[0] == "9 New Road, Mesa, IA 76237"
        assert result.dataframe["notes"].iloc[0] == "Ship to 9 New Road today"
        assert len(generator.requests) == 1

    def test_find_and_replace_replaces_a_structured_street_the_detector_missed_in_free_text(self) -> None:
        address = "489 Park Blvd, Mesa, IA 76237"
        dataframe = pd.DataFrame({"address": [address], "notes": ["Ship to 489 Park Blvd today"]})
        plan = _plan(_target("notes", EntityType.FREE_TEXT), _target("address", EntityType.STREET_ADDRESS))
        detector = _StaticDetector([_span(0, "address", address, "489 Park Blvd", EntityType.STREET_ADDRESS)])

        result = _execute(dataframe, plan, _StreetGenerator(), free_text_detector=detector, find_and_replace=True)

        assert result.dataframe["notes"].iloc[0] == "Ship to 9 New Road today"

    def test_detection_failure_replaces_addresses_whole_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        dataframe = pd.DataFrame({"address": ["489 Park Blvd, Mesa, IA 76237"]})
        generator = _StreetGenerator()

        with caplog.at_level(logging.WARNING):
            result = _execute(
                dataframe,
                _plan(_target("address", EntityType.STREET_ADDRESS)),
                generator,
                free_text_detector=_FailingDetector([]),
            )

        assert result.dataframe["address"].iloc[0] == "Whole Replacement"
        assert any("Street-address values were replaced whole" in record.getMessage() for record in caplog.records)

    def test_without_a_detector_addresses_are_replaced_whole(self) -> None:
        dataframe = pd.DataFrame({"address": ["489 Park Blvd, Mesa, IA 76237"]})

        result = _execute(dataframe, _plan(_target("address", EntityType.STREET_ADDRESS)), _StreetGenerator())

        assert result.dataframe["address"].iloc[0] == "Whole Replacement"


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
