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
StructuralGrain: TypeAlias = Literal["key", "group", "record"]


@dataclass
class ColumnEvidence:
    """Per-column name/value evidence gathered before ``EntitySpec`` allocation."""

    col: str
    """Column name in the source dataframe."""
    series: pd.Series
    """Column values used for pattern and content analysis."""
    name_label: str | None
    """Entity label inferred from the column header, or ``None`` when no name match."""
    value_entity: str | None
    """Best-covered entity the header allows, or ``None`` when content is unstructured."""
    analysis: dict[str, object]
    """Pattern analysis dict from ``analyze_column_patterns`` (entity, coverage, …)."""
    demo_label: str | None
    """Demographic label inferred from the header (gender, ethnic_background, …)."""
    grain: StructuralGrain = "record"
    """Structural grain within a training group (``key`` / ``group`` / ``record``).

    Distinct from plan replacement ``scope``. Group-constant and record-varying
    fields must not share one identity bundle.
    """


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
    """A bundle of columns that describe one person (role prefix and/or name agreement)."""

    bundle_id: str
    """Bundle identifier (e.g. ``patient``, ``provider_2``)."""
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
            fields={
                str(k): DetectedField.from_dict(cast(Mapping[str, object], v)) for k, v in fields_raw.items()
            },
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
        discovery = detection.detect_structured_columns(df, stats, cfg)
    """

    same_person_bundles: list[SamePersonBundle] = field(default_factory=list)
    """Bundles of columns that describe the same person, with demographic conditioners."""
    standalone_columns: list[DetectedStandalone] = field(default_factory=list)
    """Columns replaced independently of any same-person bundle."""
    identified_not_replaced: list[str] = field(default_factory=list)
    """Column names detected but excluded from replacement (identify-only temporals)."""
    free_text_columns: list[str] = field(default_factory=list)
    """Prose columns selected for free-text PII propagation."""

    def to_dict(self) -> dict[str, object]:
        return {
            "same_person_bundles": [b.to_dict() for b in self.same_person_bundles],
            "standalone_columns": [s.to_dict() for s in self.standalone_columns],
            "identified_not_replaced": list(self.identified_not_replaced),
            "free_text_columns": list(self.free_text_columns),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DiscoveryResult:
        bundles_raw = cast(Sequence[Mapping[str, object]], raw.get("same_person_bundles") or [])
        standalone_raw = cast(Sequence[Mapping[str, object]], raw.get("standalone_columns") or [])
        identified = cast(Sequence[object], raw.get("identified_not_replaced") or [])
        free_text = cast(Sequence[object], raw.get("free_text_columns") or [])
        return cls(
            same_person_bundles=[SamePersonBundle.from_dict(b) for b in bundles_raw],
            standalone_columns=[DetectedStandalone.from_dict(s) for s in standalone_raw],
            identified_not_replaced=[str(c) for c in identified],
            free_text_columns=[str(c) for c in free_text],
        )
