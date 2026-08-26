# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed internal contracts for detection and plan emission.

Public plan and config types stay in ``config.replace_pii``. These dataclasses
are engine-only boundaries for structured detection before flat plan emission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

import pandas as pd

DemographicAttribute: TypeAlias = Literal["gender", "ethnic_background"]


@dataclass
class ColumnEvidence:
    """Per-column name/value evidence gathered before ``EntitySpec`` allocation."""

    col: str
    """Column name in the source dataframe."""
    series: pd.Series
    """Column values used for pattern coverage and DOB / value checks."""
    name_label: str | None
    """Entity label inferred from the column header, or ``None`` when no name match."""
    value_entity: str | None
    """Dominant entity label inferred from cell values, or ``None`` when unstructured."""
    analysis: dict[str, object]
    """Pattern analysis dict from ``analyze_column_patterns`` (entity, coverage, …)."""
    demo_label: str | None
    """Demographic label inferred from the header (gender, ethnic_background, …)."""


@dataclass
class DetectedField:
    """One replaceable column belonging to a same-person bundle."""

    column: str
    """Dataframe column name."""
    pattern: str | None = None
    """Dominant value template or strftime format, or ``None`` when omitted."""

    def to_dict(self) -> dict[str, object]:
        return {"column": self.column, "pattern": self.pattern}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DetectedField:
        pattern = raw.get("pattern")
        return cls(
            column=str(raw["column"]),
            pattern=None if pattern is None else str(pattern),
        )


@dataclass
class SamePersonBundle:
    """Columns that describe one person (persona-backed fields + demographics).

    Heuristics discovery does not cluster columns into roles. When every
    persona-backed entity type appears at most once, fields go here as a single
    bundle. Duplicate entity types flatten to standalone columns instead (see
    ``DiscoveryResult.person_link_ambiguous``). LLM mode may emit multiple bundles later.
    """

    bundle_id: str
    """Bundle identifier. Heuristics always uses ``\"person\"``; LLM mode may set roles."""
    fields: dict[str, DetectedField] = field(default_factory=dict)
    """Map of entity label to column (+ optional pattern) for replaceable fields."""
    demographics: dict[DemographicAttribute, str] = field(default_factory=dict)
    """Read-only demographic columns that may condition replacements for this person."""

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "demographics": dict(self.demographics),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SamePersonBundle:
        fields_raw = cast(Mapping[str, object], raw.get("fields") or {})
        demo_raw = cast(Mapping[str, object], raw.get("demographics") or {})
        return cls(
            bundle_id=str(raw["bundle_id"]),
            fields={str(k): DetectedField.from_dict(cast(Mapping[str, object], v)) for k, v in fields_raw.items()},
            demographics={cast(DemographicAttribute, str(k)): str(v) for k, v in demo_raw.items()},
        )


@dataclass
class DetectedStandalone:
    """One standalone or entity-driven column from structured detection."""

    column: str
    """Dataframe column name."""
    entity: str
    """Entity label driving standalone replacement (e.g. ``ssn``, ``date_of_birth``)."""
    pattern: str | None = None
    """Dominant value template or strftime format, or ``None`` when omitted."""

    def to_dict(self) -> dict[str, object]:
        return {"column": self.column, "entity": self.entity, "pattern": self.pattern}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DetectedStandalone:
        pattern = raw.get("pattern")
        return cls(
            column=str(raw["column"]),
            entity=str(raw["entity"]),
            pattern=None if pattern is None else str(pattern),
        )


@dataclass
class DiscoveryResult:
    """Authoritative structured-detection output before plan emission.

    Example:
        discovery = detection.detect_structured_columns(df, cfg)
    """

    same_person_bundles: list[SamePersonBundle] = field(default_factory=list)
    """Same-person column bundles with demographic conditioners.

    Heuristics emits at most one bundle when entity types are unique. When the
    same persona entity type appears on multiple columns, bundles stay empty and
    those columns are listed under ``standalone_columns`` with
    ``person_link_ambiguous=True``.
    """
    standalone_columns: list[DetectedStandalone] = field(default_factory=list)
    """Columns replaced independently of any same-person bundle."""
    identified_not_replaced: list[str] = field(default_factory=list)
    """Column names detected but excluded from replacement (identify-only temporals)."""
    free_text_columns: list[str] = field(default_factory=list)
    """Prose columns selected for free-text PII propagation."""
    person_link_ambiguous: bool = False
    """True when duplicate persona entity types prevented same-person linking."""
    conditioning_demographics: dict[DemographicAttribute, str] = field(default_factory=dict)
    """Gender / ethnicity columns kept for depends_on hints when linking is ambiguous."""

    def to_dict(self) -> dict[str, object]:
        return {
            "same_person_bundles": [b.to_dict() for b in self.same_person_bundles],
            "standalone_columns": [s.to_dict() for s in self.standalone_columns],
            "identified_not_replaced": list(self.identified_not_replaced),
            "free_text_columns": list(self.free_text_columns),
            "person_link_ambiguous": self.person_link_ambiguous,
            "conditioning_demographics": dict(self.conditioning_demographics),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DiscoveryResult:
        bundles_raw = cast(Sequence[Mapping[str, object]], raw.get("same_person_bundles") or [])
        standalone_raw = cast(Sequence[Mapping[str, object]], raw.get("standalone_columns") or [])
        identified = cast(Sequence[object], raw.get("identified_not_replaced") or [])
        free_text = cast(Sequence[object], raw.get("free_text_columns") or [])
        demos_raw = cast(Mapping[str, object], raw.get("conditioning_demographics") or {})
        return cls(
            same_person_bundles=[SamePersonBundle.from_dict(b) for b in bundles_raw],
            standalone_columns=[DetectedStandalone.from_dict(s) for s in standalone_raw],
            identified_not_replaced=[str(c) for c in identified],
            free_text_columns=[str(c) for c in free_text],
            person_link_ambiguous=bool(raw.get("person_link_ambiguous", False)),
            conditioning_demographics={
                cast(DemographicAttribute, str(k)): str(v) for k, v in demos_raw.items()
            },
        )
