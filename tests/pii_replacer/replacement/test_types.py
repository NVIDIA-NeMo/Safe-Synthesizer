# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, fields
from typing import cast

import pytest

from nemo_safe_synthesizer.config.replace_pii import EntityType
from nemo_safe_synthesizer.pii_replacer.replacement.types import (
    DetectedSpan,
    DetectionCell,
    DetectionCellId,
    DetectionSource,
    FreeTextMappingKey,
    GroupDependencyDrift,
    GroupMappingKey,
    GroupMappingProvenance,
    NonGroupMappingKey,
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
    def test_cell_identity_is_positional_and_contains_no_raw_value(self) -> None:
        first = DetectionCellId(row_position=0, column_name="notes")
        duplicate_index_peer = DetectionCellId(row_position=1, column_name="notes")

        assert first != duplicate_index_peer
        assert {field.name for field in fields(DetectionCellId)} == {"row_position", "column_name"}

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
        with pytest.raises(ValueError, match="0 <= start < end"):
            _span(start=start, end=end)

    def test_span_is_validated_against_the_complete_original_cell(self) -> None:
        cell = _cell()

        assert detected_text(cell, _span()) == "ada@example.com"
        with pytest.raises(ValueError, match="exceeds the original cell length"):
            detected_text(cell, _span(end=len(cell.text) + 1))
        with pytest.raises(ValueError, match="cell_id does not match"):
            detected_text(cell, _span(cell_id=DetectionCellId(0, "notes")))
        with pytest.raises(ValueError, match="entity_type is not allowed"):
            detected_text(cell, _span(entity_type=EntityType.FULL_NAME))

    def test_detector_result_has_normalized_type_and_provenance_but_no_replacement(self) -> None:
        span = _span(source="gliner", score=0.91)

        assert span.entity_type is EntityType.EMAIL
        assert span.source == "gliner"
        assert span.score == 0.91
        assert "replacement" not in {field.name for field in fields(DetectedSpan)}

    def test_detector_result_requires_a_normalized_entity_type(self) -> None:
        with pytest.raises(TypeError, match="normalized EntityType"):
            _span(entity_type=cast(EntityType, "email"))

    def test_cell_requires_an_immutable_allowed_entity_collection(self) -> None:
        with pytest.raises(TypeError, match="must be a frozenset"):
            DetectionCell(
                DetectionCellId(0, "notes"),
                "text",
                cast(frozenset[EntityType], {EntityType.EMAIL}),
            )


@pytest.mark.unit
class TestMappingContracts:
    def test_repeated_free_text_value_in_one_cell_reuses_the_same_mapping(self) -> None:
        cell = DetectionCell(
            cell_id=DetectionCellId(row_position=3, column_name="notes"),
            text="Ada met Ada",
            allowed_entity_types=frozenset({EntityType.FIRST_NAME}),
        )
        first_span = DetectedSpan(cell.cell_id, 0, 3, EntityType.FIRST_NAME, "gliner", 0.9)
        second_span = DetectedSpan(cell.cell_id, 8, 11, EntityType.FIRST_NAME, "gliner", 0.8)

        first_key = free_text_mapping_key(cell, first_span, scope_identity=cell.cell_id.row_position)
        second_key = free_text_mapping_key(cell, second_span, scope_identity=cell.cell_id.row_position)

        assert first_key == second_key
        assert first_key == FreeTextMappingKey("notes", 3, EntityType.FIRST_NAME, "Ada")
        assert "Ada" not in repr(first_key)

    def test_non_group_identity_includes_effective_dependencies(self) -> None:
        base = NonGroupMappingKey(
            target_column="email",
            scope_identity=0,
            canonical_original_value="ada@example.com",
            effective_dependency_tuple=((EntityType.ORGANIZATION, "example"),),
        )
        changed_dependency = NonGroupMappingKey(
            target_column="email",
            scope_identity=0,
            canonical_original_value="ada@example.com",
            effective_dependency_tuple=((EntityType.ORGANIZATION, "different"),),
        )

        assert base != changed_dependency
        assert "ada@example.com" not in repr(base)
        assert "example" not in repr(base)

    def test_group_identity_excludes_dependencies_and_records_first_provenance_separately(self) -> None:
        key = GroupMappingKey(
            target_column="email",
            original_group_identity="patient-1",
            canonical_original_value="ada@example.com",
        )
        first = GroupMappingProvenance(((EntityType.ORGANIZATION, "example"),))
        later = GroupMappingProvenance(((EntityType.ORGANIZATION, "different"),))

        assert key == GroupMappingKey("email", "patient-1", "ada@example.com")
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
        assert {field.name for field in fields(GroupDependencyDrift)} == {
            "target_column",
            "conditioner_entity_types",
            "conflict_count",
        }

    def test_group_dependency_drift_requires_a_conflict(self) -> None:
        with pytest.raises(ValueError, match="conflict_count must be positive"):
            GroupDependencyDrift("email", frozenset({EntityType.ORGANIZATION}), 0)
