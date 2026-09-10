# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts shared by PII replacement detection, mapping, and generation adapters.

The execution engine composes these types to keep positional identity separate
from sensitive values and to keep sensitive mapping inputs out of diagnostics.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from ...config.replace_pii import EntityType

DetectionSource: TypeAlias = Literal["gliner", "regex"]
CanonicalValue: TypeAlias = str
EffectiveDependencyTuple: TypeAlias = tuple[tuple[EntityType, CanonicalValue | None], ...]


@dataclass(frozen=True, slots=True)
class DetectionCellId:
    """PII-free positional identity for one dataframe cell.

    ``row_position`` refers to stable input row order rather than the dataframe
    index, which may contain duplicates. ``column_name`` identifies the target
    column without carrying the raw cell value.
    """

    row_position: int
    column_name: str

    def __post_init__(self) -> None:
        if type(self.row_position) is not int:
            raise TypeError("detection cell row_position must be an integer")
        if self.row_position < 0:
            raise ValueError("detection cell row_position must be nonnegative")
        if not isinstance(self.column_name, str):
            raise TypeError("detection cell column_name must be a string")


@dataclass(frozen=True, slots=True)
class DetectionCell:
    """Original cell text and the entity types its plan permits detecting."""

    cell_id: DetectionCellId
    text: str = field(repr=False)
    allowed_entity_types: frozenset[EntityType]

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, DetectionCellId):
            raise TypeError("cell_id must be a DetectionCellId")
        if not isinstance(self.text, str):
            raise TypeError("detection cell text must be a string")
        if not isinstance(self.allowed_entity_types, frozenset):
            raise TypeError("allowed_entity_types must be a frozenset")
        if not all(isinstance(entity_type, EntityType) for entity_type in self.allowed_entity_types):
            raise TypeError("allowed_entity_types must contain normalized EntityType values")


@dataclass(frozen=True, slots=True)
class DetectedSpan:
    """One detector-produced half-open span in the complete original cell.

    The type contains normalized entity type and detector provenance, but no
    original text or replacement value. Use ``detected_text`` to validate the
    span against its ``DetectionCell`` and extract the exact occurrence.
    """

    cell_id: DetectionCellId
    start: int
    end: int
    entity_type: EntityType
    source: DetectionSource
    score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, DetectionCellId):
            raise TypeError("cell_id must be a DetectionCellId")
        if type(self.start) is not int or type(self.end) is not int:
            raise TypeError("detected span offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("detected span offsets must satisfy 0 <= start < end")
        if not isinstance(self.entity_type, EntityType):
            raise TypeError("detected span entity_type must be a normalized EntityType")
        if self.source not in {"gliner", "regex"}:
            raise ValueError("detected span source must be 'gliner' or 'regex'")


def detected_text(cell: DetectionCell, span: DetectedSpan) -> str:
    """Validate ``span`` against ``cell`` and return its exact detected text.

    Offsets are interpreted against the original complete cell and remain
    half-open. Error messages deliberately omit the cell text.
    """
    if span.cell_id != cell.cell_id:
        raise ValueError("detected span cell_id does not match its detection cell")
    if span.end > len(cell.text):
        raise ValueError("detected span end exceeds the original cell length")
    if span.entity_type not in cell.allowed_entity_types:
        raise ValueError("detected span entity_type is not allowed for its detection cell")
    return cell.text[span.start : span.end]


@dataclass(frozen=True, slots=True)
class NonGroupMappingKey:
    """Mapping identity for record and dataframe scopes.

    Identity includes the target column, scope identity, canonical original
    value, and effective dependency tuple. Sensitive values are excluded from
    ``repr`` so accidental diagnostics do not disclose them.
    """

    target_column: str
    scope_identity: Hashable = field(repr=False)
    canonical_original_value: CanonicalValue = field(repr=False)
    effective_dependency_tuple: EffectiveDependencyTuple = field(repr=False)


@dataclass(frozen=True, slots=True)
class FreeTextMappingKey:
    """Mapping identity for a detected value inside a free-text cell.

    The replacement executor looks up every accepted span by this key before
    calling the replacement generator. Repeated occurrences of the same entity
    and canonical value in one row and column therefore reuse one replacement,
    independently of propagation mappings. ``scope_identity`` may widen that
    reuse for group or dataframe scope.
    """

    target_column: str
    scope_identity: Hashable = field(repr=False)
    entity_type: EntityType
    canonical_original_value: CanonicalValue = field(repr=False)


def free_text_mapping_key(
    cell: DetectionCell,
    span: DetectedSpan,
    *,
    scope_identity: Hashable,
) -> FreeTextMappingKey:
    """Build the cache key for one accepted free-text detection."""
    return FreeTextMappingKey(
        target_column=cell.cell_id.column_name,
        scope_identity=scope_identity,
        entity_type=span.entity_type,
        canonical_original_value=detected_text(cell, span),
    )


@dataclass(frozen=True, slots=True)
class GroupMappingKey:
    """Mapping identity for group scope.

    Effective dependencies are intentionally absent from identity: the first
    occurrence in stable positional row order establishes the replacement for
    this target, original group, and canonical original value.
    """

    target_column: str
    original_group_identity: Hashable = field(repr=False)
    canonical_original_value: CanonicalValue = field(repr=False)


@dataclass(frozen=True, slots=True)
class GroupMappingProvenance:
    """Effective dependencies used by the first occurrence of a group mapping."""

    effective_dependency_tuple: EffectiveDependencyTuple = field(repr=False)


@dataclass(frozen=True, slots=True)
class GroupDependencyDrift:
    """PII-free aggregate warning for dependency drift within a group mapping."""

    target_column: str
    conditioner_entity_types: frozenset[EntityType]
    conflict_count: int

    def __post_init__(self) -> None:
        if self.conflict_count <= 0:
            raise ValueError("group dependency drift conflict_count must be positive")

    def as_log_extra(self) -> dict[str, object]:
        """Return the complete safe structured warning payload."""
        return {
            "target_column": self.target_column,
            "conditioner_entity_types": sorted(entity_type.value for entity_type in self.conditioner_entity_types),
            "conflict_count": self.conflict_count,
        }
