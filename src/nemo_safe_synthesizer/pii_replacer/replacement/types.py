# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts shared by PII replacement detection, mapping, and generation adapters.

The execution engine composes these types to keep positional identity separate
from sensitive values and to keep sensitive mapping inputs out of diagnostics.
The replacement executor is the only producer of these values, so contract
violations raise ``InternalError``.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, get_args

from ...config.replace_pii import EntityType
from ...errors import InternalError

__all__ = [
    "CanonicalValue",
    "DetectedSpan",
    "DetectionCell",
    "DetectionCellId",
    "DetectionSource",
    "EffectiveDependencyTuple",
    "FreeTextMappingKey",
    "GroupDependencyDrift",
    "GroupMappingKey",
    "GroupMappingProvenance",
    "RecordMappingKey",
    "detected_text",
    "free_text_mapping_key",
    "require_effective_dependency_tuple",
]

DetectionSource: TypeAlias = Literal["gliner", "regex"]
_DETECTION_SOURCES: frozenset[str] = frozenset(get_args(DetectionSource))


def _require_row_position(row_position: object, owner: str) -> None:
    if type(row_position) is not int:
        raise InternalError(f"{owner} row_position must be an integer")
    if row_position < 0:
        raise InternalError(f"{owner} row_position must be nonnegative")


def _require_str(value: object, description: str) -> None:
    if not isinstance(value, str):
        raise InternalError(f"{description} must be a string")


def _require_entity_type(value: object, description: str) -> None:
    if not isinstance(value, EntityType):
        raise InternalError(f"{description} must be a normalized EntityType")


def _require_entity_types(values: object, description: str) -> None:
    if not isinstance(values, frozenset):
        raise InternalError(f"{description} must be a frozenset")
    if not all(isinstance(value, EntityType) for value in values):
        raise InternalError(f"{description} must contain normalized EntityType values")


def _require_canonical_value(value: object, description: str) -> None:
    if not isinstance(value, CanonicalValue):
        raise InternalError(f"{description} must be a CanonicalValue")


def _require_scope_identity(value: object, description: str) -> None:
    # NaN is hashable but never equal to itself, so it would silently defeat mapping reuse.
    if not isinstance(value, Hashable) or (isinstance(value, float) and math.isnan(value)):
        raise InternalError(f"{description} must be hashable and not NaN")


@dataclass(frozen=True, slots=True)
class CanonicalValue:
    """Type-tagged, normalized identity for an original structured scalar.

    The replacement executor creates this value only after excluding missing
    values. The tag prevents unlike values such as integer ``1`` and string
    ``"1"`` from sharing a mapping. Canonicalization means stable typed
    serialization, not text cleanup: it must not trim, case-fold, or otherwise
    alter string content. Equivalent Python, NumPy, or pandas scalars must
    produce the same pair; different semantic scalar types must not.

    Example:
        CanonicalValue(type_tag="integer", normalized_value="1")  # from 1, np.int64(1), ...
        CanonicalValue(type_tag="string", normalized_value="1")  # from "1"; a different identity
    """

    type_tag: str
    """Nonempty identifier of the scalar's semantic type."""

    normalized_value: str = field(repr=False)
    """Deterministic, locale-independent string form of the scalar.

    Excluded from ``repr`` because it may contain PII.
    """

    def __post_init__(self) -> None:
        _require_str(self.type_tag, "canonical value type_tag")
        if not self.type_tag:
            raise InternalError("canonical value type_tag must be a nonempty string")
        _require_str(self.normalized_value, "canonical normalized_value")


EffectiveDependencyTuple: TypeAlias = tuple[tuple[EntityType, CanonicalValue | None], ...]


def require_effective_dependency_tuple(value: object, description: str) -> None:
    """Raise ``InternalError`` unless ``value`` is a well-formed ``EffectiveDependencyTuple``."""
    if not isinstance(value, tuple):
        raise InternalError(f"{description} must be a tuple")
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], EntityType)
            or not (item[1] is None or isinstance(item[1], CanonicalValue))
        ):
            raise InternalError(f"{description} must contain (EntityType, CanonicalValue | None) pairs")


@dataclass(frozen=True, slots=True)
class DetectionCellId:
    """PII-free positional identity for one dataframe cell.

    Example:
        DetectionCellId(row_position=3, column_name="notes")  # fourth input row, even if its index label is 7
    """

    row_position: int
    """Stable input row order rather than the dataframe index, which may contain duplicates."""

    column_name: str
    """Target column, identified without carrying the raw cell value."""

    def __post_init__(self) -> None:
        _require_row_position(self.row_position, "detection cell")
        _require_str(self.column_name, "detection cell column_name")


@dataclass(frozen=True, slots=True)
class DetectionCell:
    """Original cell text and the entity types its plan permits detecting.

    Example:
        DetectionCell(
            cell_id=DetectionCellId(3, "notes"),
            text="Email ada@example.com today",
            allowed_entity_types=frozenset({EntityType.EMAIL, EntityType.FIRST_NAME}),
        )
    """

    cell_id: DetectionCellId
    """Positional identity of the cell."""

    text: str = field(repr=False)
    """Complete original cell text. Excluded from ``repr`` because it may contain PII."""

    allowed_entity_types: frozenset[EntityType]
    """Entity types the plan permits detecting in this cell."""

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, DetectionCellId):
            raise InternalError("cell_id must be a DetectionCellId")
        _require_str(self.text, "detection cell text")
        _require_entity_types(self.allowed_entity_types, "allowed_entity_types")


@dataclass(frozen=True, slots=True)
class DetectedSpan:
    """One detector-produced half-open span in the complete original cell.

    The type contains normalized entity type and detector provenance, but no
    original text or replacement value. Use ``detected_text`` to validate the
    span against its ``DetectionCell`` and extract the exact occurrence.

    Example:
        # "ada@example.com" in "Email ada@example.com today", found by the email regex
        DetectedSpan(DetectionCellId(3, "notes"), start=6, end=21, entity_type=EntityType.EMAIL, source="regex")
    """

    cell_id: DetectionCellId
    """Positional identity of the cell containing the span."""

    start: int
    """Inclusive start offset into the complete original cell text."""

    end: int
    """Exclusive end offset into the complete original cell text."""

    entity_type: EntityType
    """Normalized entity type assigned by the detector."""

    source: DetectionSource
    """Detector that produced the span."""

    score: float | None = None
    """Detector confidence in ``[0, 1]``, or ``None`` for detectors without scores."""

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, DetectionCellId):
            raise InternalError("cell_id must be a DetectionCellId")
        if type(self.start) is not int or type(self.end) is not int:
            raise InternalError("detected span offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise InternalError(
                f"detected span offsets must satisfy 0 <= start < end, got start={self.start}, end={self.end}"
            )
        _require_entity_type(self.entity_type, "detected span entity_type")
        if self.source not in _DETECTION_SOURCES:
            raise InternalError(
                f"detected span source must be one of {sorted(_DETECTION_SOURCES)}, got {self.source!r}"
            )
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, int | float):
                raise InternalError("detected span score must be a number or None")
            if not 0 <= self.score <= 1:
                raise InternalError(f"detected span score must be between 0 and 1, got {self.score}")


def detected_text(cell: DetectionCell, span: DetectedSpan) -> str:
    """Validate ``span`` against ``cell`` and return its exact detected text.

    Offsets are interpreted against the original complete cell and remain
    half-open. Error messages deliberately omit the cell text.
    """
    if span.cell_id != cell.cell_id:
        raise InternalError("detected span cell_id does not match its detection cell")
    if span.end > len(cell.text):
        raise InternalError("detected span end exceeds the original cell length")
    if span.entity_type not in cell.allowed_entity_types:
        raise InternalError("detected span entity_type is not allowed for its detection cell")
    return cell.text[span.start : span.end]


@dataclass(frozen=True, slots=True)
class RecordMappingKey:
    """Mapping identity for a structured value in record scope.

    Stable positional row identity keeps duplicate dataframe indexes safe.
    Dependencies affect generation but not mapping identity because every row
    has one effective dependency tuple for a given target.

    Example:
        # Every occurrence of "ada@example.com" in row 0's email column reuses one replacement.
        RecordMappingKey("email", 0, CanonicalValue("string", "ada@example.com"))
    """

    target_column: str
    """Column whose value is replaced."""

    row_position: int
    """Stable input row position that scopes the mapping."""

    canonical_original_value: CanonicalValue = field(repr=False)
    """Canonical original value. Excluded from ``repr`` because it may contain PII."""

    def __post_init__(self) -> None:
        _require_str(self.target_column, "record mapping target_column")
        _require_row_position(self.row_position, "record mapping")
        _require_canonical_value(self.canonical_original_value, "record mapping canonical_original_value")


@dataclass(frozen=True, slots=True)
class FreeTextMappingKey:
    """Mapping identity for a detected value within the configured scope.

    The replacement executor looks up every accepted span by this key before
    calling the replacement generator. Repeated occurrences of the same entity
    and original value in any planned free-text column therefore reuse one
    replacement within the scope, independently of propagation mappings.
    Matching ignores case, so ``Margaret`` and ``MARGARET`` share a key; the
    key keeps the exact text of the occurrence that created it for generation.
    When the shared replacement is written into a later occurrence, it takes
    that occurrence's letter case: all capitals, all lowercase, or title case.

    Example:
        # In row 3, "Ada" in one note and "ADA" in another get the same replacement.
        FreeTextMappingKey(scope_identity=3, entity_type=EntityType.FIRST_NAME, original_value="Ada")
    """

    scope_identity: Hashable = field(repr=False)
    """Stable row position in record scope, or the group identity to widen reuse across rows.

    Excluded from ``repr`` because a group identity may contain PII.
    """

    entity_type: EntityType
    """Normalized entity type of the detected value."""

    original_value: str = field(repr=False, compare=False)
    """Exact detected text passed to the generator. Not part of identity; excluded from ``repr`` because it is PII."""

    folded_value: str = field(init=False, repr=False)
    """Case-folded ``original_value`` that identifies the key. Excluded from ``repr`` because it is PII."""

    def __post_init__(self) -> None:
        _require_scope_identity(self.scope_identity, "free-text mapping scope_identity")
        _require_entity_type(self.entity_type, "free-text mapping entity_type")
        _require_str(self.original_value, "free-text mapping original_value")
        object.__setattr__(self, "folded_value", self.original_value.casefold())


def free_text_mapping_key(
    cell: DetectionCell,
    span: DetectedSpan,
    *,
    scope_identity: Hashable,
) -> FreeTextMappingKey:
    """Build the cache key for one accepted free-text detection."""
    return FreeTextMappingKey(
        scope_identity=scope_identity,
        entity_type=span.entity_type,
        original_value=detected_text(cell, span),
    )


@dataclass(frozen=True, slots=True)
class GroupMappingKey:
    """Mapping identity for group scope.

    Effective dependencies are intentionally absent from identity: the first
    occurrence in stable positional row order establishes the replacement for
    this target, original group, and canonical original value.

    Example:
        # Every row of patient-1 with "ada@example.com" in the email column reuses one replacement.
        GroupMappingKey("email", "patient-1", CanonicalValue("string", "ada@example.com"))
    """

    target_column: str
    """Column whose value is replaced."""

    original_group_identity: Hashable = field(repr=False)
    """Original grouping-column value. Excluded from ``repr`` because it may contain PII."""

    canonical_original_value: CanonicalValue = field(repr=False)
    """Canonical original value. Excluded from ``repr`` because it may contain PII."""

    def __post_init__(self) -> None:
        _require_str(self.target_column, "group mapping target_column")
        _require_scope_identity(self.original_group_identity, "group mapping original_group_identity")
        _require_canonical_value(self.canonical_original_value, "group mapping canonical_original_value")


@dataclass(frozen=True, slots=True)
class GroupMappingProvenance:
    """Effective dependencies used by the first occurrence of a group mapping.

    Later occurrences with different dependencies reuse the first replacement and
    are counted as ``GroupDependencyDrift``.

    Example:
        # The group's email replacement was generated from the replaced company name "Example Corp".
        GroupMappingProvenance(((EntityType.ORGANIZATION, CanonicalValue("string", "Example Corp")),))
    """

    effective_dependency_tuple: EffectiveDependencyTuple = field(repr=False)
    """Dependency values that conditioned the first replacement.

    Excluded from ``repr`` because they may contain PII.
    """

    def __post_init__(self) -> None:
        require_effective_dependency_tuple(self.effective_dependency_tuple, "group mapping effective_dependency_tuple")


@dataclass(frozen=True, slots=True)
class GroupDependencyDrift:
    """PII-free aggregate warning for dependency drift within a group mapping.

    Example:
        # Three later rows reused a group's email replacement although their company differed.
        GroupDependencyDrift("email", frozenset({EntityType.ORGANIZATION}), conflict_count=3)
    """

    target_column: str
    """Column whose group mapping saw drifting dependencies."""

    conditioner_entity_types: frozenset[EntityType]
    """Entity types of the dependencies that drifted."""

    conflict_count: int
    """Number of later occurrences whose dependencies differed from the first occurrence."""

    def __post_init__(self) -> None:
        _require_str(self.target_column, "group dependency drift target_column")
        _require_entity_types(self.conditioner_entity_types, "group dependency drift conditioner_entity_types")
        if type(self.conflict_count) is not int:
            raise InternalError("group dependency drift conflict_count must be an integer")
        if self.conflict_count <= 0:
            raise InternalError(f"group dependency drift conflict_count must be positive, got {self.conflict_count}")

    def as_log_extra(self) -> dict[str, object]:
        """Return the complete safe structured warning payload."""
        return {
            "target_column": self.target_column,
            "conditioner_entity_types": sorted(entity_type.value for entity_type in self.conditioner_entity_types),
            "conflict_count": self.conflict_count,
        }
