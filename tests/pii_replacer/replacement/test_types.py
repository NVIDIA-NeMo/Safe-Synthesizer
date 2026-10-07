# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import pytest

from nemo_safe_synthesizer.config.replace_pii import EntityType
from nemo_safe_synthesizer.errors import InternalError
from nemo_safe_synthesizer.pii_replacer.replacement.types import (
    CanonicalValue,
    DetectedSpan,
    DetectionCell,
    DetectionCellId,
    DetectionSource,
    FreeTextMappingKey,
    GroupDependencyDrift,
    GroupMappingKey,
    GroupMappingProvenance,
    RecordMappingKey,
    detected_text,
    free_text_mapping_key,
)


def _cell() -> DetectionCell:
    return DetectionCell(
        cell_id=DetectionCellId(row_position=1, column_name="notes"),
        text="Email ada@example.com today",
        allowed_entity_types=frozenset({EntityType.EMAIL}),
    )


def _span(
    *,
    cell_id: DetectionCellId | None = None,
    start: int = 6,
    end: int = 21,
    entity_type: EntityType = EntityType.EMAIL,
    source: DetectionSource = "regex",
    score: float | None = None,
) -> DetectedSpan:
    return DetectedSpan(
        cell_id=cell_id or DetectionCellId(row_position=1, column_name="notes"),
        start=start,
        end=end,
        entity_type=entity_type,
        source=source,
        score=score,
    )


@pytest.mark.unit
class TestDetectionContracts:
    def test_contracts_are_immutable_and_hide_cell_text_from_repr(self) -> None:
        cell = _cell()
        span = _span()

        with pytest.raises(FrozenInstanceError):
            setattr(cell, "text", "changed")
        with pytest.raises(FrozenInstanceError):
            setattr(span, "end", 10)
        assert "ada@example.com" not in repr(cell)
        assert "ada@example.com" not in repr(span)

    @pytest.mark.parametrize(
        ("start", "end"),
        [(-1, 2), (0, 0), (2, 1)],
    )
    def test_span_requires_nonempty_half_open_offsets(self, start: int, end: int) -> None:
        with pytest.raises(InternalError, match="0 <= start < end"):
            _span(start=start, end=end)

    def test_span_is_validated_against_the_complete_original_cell(self) -> None:
        cell = _cell()

        assert detected_text(cell, _span()) == "ada@example.com"
        with pytest.raises(InternalError, match="exceeds the original cell length"):
            detected_text(cell, _span(end=len(cell.text) + 1))
        with pytest.raises(InternalError, match="cell_id does not match"):
            detected_text(cell, _span(cell_id=DetectionCellId(0, "notes")))
        with pytest.raises(InternalError, match="entity_type is not allowed"):
            detected_text(cell, _span(entity_type=EntityType.FULL_NAME))

    def test_detector_result_requires_a_normalized_entity_type(self) -> None:
        with pytest.raises(InternalError, match="normalized EntityType"):
            _span(entity_type="email")  # ty: ignore[invalid-argument-type] -- deliberate invalid input

    def test_cell_requires_an_immutable_allowed_entity_collection(self) -> None:
        with pytest.raises(InternalError, match="must be a frozenset"):
            DetectionCell(
                DetectionCellId(0, "notes"),
                "text",
                {EntityType.EMAIL},  # ty: ignore[invalid-argument-type] -- deliberate invalid input
            )

    @pytest.mark.parametrize("score", [-0.1, 1.1, True])
    def test_span_score_must_be_a_unit_interval_number(self, score: float) -> None:
        with pytest.raises(InternalError, match="score must be"):
            _span(score=score)


@pytest.mark.unit
class TestMappingContracts:
    def test_repeated_free_text_value_in_one_row_reuses_mapping_across_columns(self) -> None:
        first_cell = DetectionCell(
            cell_id=DetectionCellId(row_position=3, column_name="notes_primary"),
            text="Ada met Ada",
            allowed_entity_types=frozenset({EntityType.FIRST_NAME}),
        )
        second_cell = DetectionCell(
            cell_id=DetectionCellId(row_position=3, column_name="notes_secondary"),
            text="Call Ada",
            allowed_entity_types=frozenset({EntityType.FIRST_NAME}),
        )
        first_span = DetectedSpan(first_cell.cell_id, 0, 3, EntityType.FIRST_NAME, "gliner", 0.9)
        repeated_span = DetectedSpan(first_cell.cell_id, 8, 11, EntityType.FIRST_NAME, "gliner", 0.8)
        other_column_span = DetectedSpan(second_cell.cell_id, 5, 8, EntityType.FIRST_NAME, "gliner", 0.7)

        first_key = free_text_mapping_key(first_cell, first_span, scope_identity=first_cell.cell_id.row_position)
        repeated_key = free_text_mapping_key(first_cell, repeated_span, scope_identity=first_cell.cell_id.row_position)
        other_column_key = free_text_mapping_key(
            second_cell,
            other_column_span,
            scope_identity=second_cell.cell_id.row_position,
        )

        assert first_key == repeated_key == other_column_key
        assert first_key == FreeTextMappingKey(3, EntityType.FIRST_NAME, "Ada")
        assert "Ada" not in repr(first_key)

    def test_canonical_structured_value_is_type_tagged_and_hides_pii(self) -> None:
        integer = CanonicalValue(type_tag="integer", normalized_value="1")
        string = CanonicalValue(type_tag="string", normalized_value="1")

        assert integer != string
        assert "1" not in repr(integer)

    def test_record_identity_uses_stable_row_position_without_dependencies(self) -> None:
        original_value = CanonicalValue(type_tag="string", normalized_value="ada@example.com")
        base = RecordMappingKey(
            target_column="email",
            row_position=0,
            canonical_original_value=original_value,
        )
        duplicate_index_peer = RecordMappingKey(
            target_column="email",
            row_position=1,
            canonical_original_value=original_value,
        )

        assert base != duplicate_index_peer
        assert "ada@example.com" not in repr(base)

    def test_group_identity_excludes_dependencies_and_records_first_provenance_separately(self) -> None:
        original_value = CanonicalValue(type_tag="string", normalized_value="ada@example.com")
        first_dependency = CanonicalValue(type_tag="string", normalized_value="example")
        later_dependency = CanonicalValue(type_tag="string", normalized_value="different")
        key = GroupMappingKey(
            target_column="email",
            original_group_identity="patient-1",
            canonical_original_value=original_value,
        )
        first = GroupMappingProvenance(((EntityType.ORGANIZATION, first_dependency),))
        later = GroupMappingProvenance(((EntityType.ORGANIZATION, later_dependency),))

        assert key == GroupMappingKey("email", "patient-1", original_value)
        assert first != later
        assert "patient-1" not in repr(key)
        assert "ada@example.com" not in repr(key)
        assert "example" not in repr(first)

    def test_group_dependency_drift_payload_is_aggregate_and_pii_free(self) -> None:
        drift = GroupDependencyDrift(
            target_column="email",
            conditioner_entity_types=frozenset({EntityType.ORGANIZATION, EntityType.FIRST_NAME}),
            conflict_count=3,
        )

        assert drift.as_log_extra() == {
            "target_column": "email",
            "conditioner_entity_types": ["first_name", "organization"],
            "conflict_count": 3,
        }

    def test_group_dependency_drift_requires_a_conflict(self) -> None:
        with pytest.raises(InternalError, match="conflict_count must be positive"):
            GroupDependencyDrift("email", frozenset({EntityType.ORGANIZATION}), 0)
        with pytest.raises(InternalError, match="conflict_count must be an integer"):
            GroupDependencyDrift("email", frozenset({EntityType.ORGANIZATION}), True)

    def test_group_dependency_drift_requires_normalized_entity_types(self) -> None:
        with pytest.raises(InternalError, match="normalized EntityType"):
            GroupDependencyDrift(
                "email",
                frozenset({"organization"}),  # ty: ignore[invalid-argument-type] -- deliberate invalid input
                1,
            )

    def test_mapping_keys_require_canonical_original_values(self) -> None:
        with pytest.raises(InternalError, match="must be a CanonicalValue"):
            RecordMappingKey("email", 0, "ada@example.com")  # ty: ignore[invalid-argument-type] -- deliberate invalid input
        with pytest.raises(InternalError, match="must be a CanonicalValue"):
            GroupMappingKey("email", "patient-1", "ada@example.com")  # ty: ignore[invalid-argument-type] -- deliberate invalid input

    @pytest.mark.parametrize("scope_identity", [float("nan"), ["unhashable"]])
    def test_scope_identities_must_be_hashable_and_not_nan(self, scope_identity: object) -> None:
        original_value = CanonicalValue(type_tag="string", normalized_value="ada@example.com")

        with pytest.raises(InternalError, match="hashable and not NaN"):
            FreeTextMappingKey(scope_identity, EntityType.EMAIL, "ada@example.com")
        with pytest.raises(InternalError, match="hashable and not NaN"):
            GroupMappingKey("email", scope_identity, original_value)

    def test_free_text_mapping_key_requires_a_normalized_entity_type(self) -> None:
        with pytest.raises(InternalError, match="normalized EntityType"):
            FreeTextMappingKey(0, "email", "ada@example.com")  # ty: ignore[invalid-argument-type] -- deliberate invalid input

    def test_group_provenance_requires_well_formed_dependencies(self) -> None:
        with pytest.raises(InternalError, match="must contain"):
            GroupMappingProvenance((("organization", None),))  # ty: ignore[invalid-argument-type] -- deliberate invalid input
