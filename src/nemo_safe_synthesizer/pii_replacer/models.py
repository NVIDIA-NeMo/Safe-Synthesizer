# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed internal pipeline contracts for detection, planning, and replacement.

Public plan and config types stay in ``config.replace_pii``. These dataclasses
are engine-only boundaries so LLM hooks and package seams do not harden raw
dictionary shapes.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

import pandas as pd

PersonaAttribute: TypeAlias = Literal["gender", "ethnic_background"]
ScopedValueMapKind: TypeAlias = Literal["flat", "group", "record"]


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
    """Dominant entity label inferred from cell values, or ``None`` when unstructured."""
    analysis: dict[str, object]
    """Pattern analysis dict from ``analyze_column_patterns`` (entity, coverage, …)."""
    demo_label: str | None
    """Demographic label inferred from the header (sex, race, …), or ``None``."""


@dataclass
class FieldMeta:
    """Per-field discovery metadata (currently pattern templates)."""

    patterns: list[str] = field(default_factory=list)
    """Value templates or strftime formats attached during discovery."""

    def to_dict(self) -> dict[str, object]:
        return {"patterns": list(self.patterns)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object] | None) -> FieldMeta:
        if not raw:
            return cls()
        patterns = raw.get("patterns") or []
        return cls(patterns=[str(p) for p in cast(Sequence[object], patterns)])


@dataclass
class PersonaMatchBy:
    """Demographic column used to condition persona sampling."""

    persona_attribute: PersonaAttribute
    """Persona attribute name (``gender`` or ``ethnic_background``)."""
    column_name: str
    """Dataframe column whose values constrain which synthetic persona is drawn."""

    def to_dict(self) -> dict[str, str]:
        return {"persona_attribute": self.persona_attribute, "column_name": self.column_name}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> PersonaMatchBy:
        return cls(
            persona_attribute=cast(PersonaAttribute, str(raw["persona_attribute"])),
            column_name=str(raw["column_name"]),
        )


@dataclass
class DetectedPersona:
    """One persona group from structured detection."""

    persona: str
    """Persona identifier (e.g. ``patient``, ``provider_2``)."""
    fields: dict[str, str] = field(default_factory=dict)
    """Map of entity label to column name for persona-backed replacement."""
    field_meta: dict[str, FieldMeta] = field(default_factory=dict)
    """Per-field metadata keyed by entity label (pattern templates, …)."""
    match_persona_by: list[PersonaMatchBy] = field(default_factory=list)
    """Demographic columns that condition which synthetic persona is drawn."""

    def to_dict(self) -> dict[str, object]:
        return {
            "persona": self.persona,
            "fields": dict(self.fields),
            "field_meta": {k: v.to_dict() for k, v in self.field_meta.items()},
            "match_persona_by": [m.to_dict() for m in self.match_persona_by],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DetectedPersona:
        meta_raw = cast(Mapping[str, object], raw.get("field_meta") or {})
        match_raw = cast(Sequence[Mapping[str, object]], raw.get("match_persona_by") or [])
        fields_raw = cast(Mapping[str, object], raw.get("fields") or {})
        return cls(
            persona=str(raw["persona"]),
            fields={str(k): str(v) for k, v in fields_raw.items()},
            field_meta={str(k): FieldMeta.from_dict(cast(Mapping[str, object], v)) for k, v in meta_raw.items()},
            match_persona_by=[PersonaMatchBy.from_dict(m) for m in match_raw],
        )


@dataclass
class DetectedStandalone:
    """One standalone or entity-driven column from structured detection."""

    column: str
    """Dataframe column name."""
    entity: str
    """Entity label driving standalone replacement (e.g. ``ssn``, ``date_of_birth``)."""
    patterns: list[str] = field(default_factory=list)
    """Value templates or strftime formats inferred for the column."""

    def to_dict(self) -> dict[str, object]:
        return {"column": self.column, "entity": self.entity, "patterns": list(self.patterns)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DetectedStandalone:
        patterns = raw.get("patterns") or []
        return cls(
            column=str(raw["column"]),
            entity=str(raw["entity"]),
            patterns=[str(p) for p in cast(Sequence[object], patterns)],
        )


@dataclass
class DiscoveryResult:
    """Authoritative structured-detection output before plan emission.

    Example:
        discovery = DiscoveryResult.from_dict(detection.detect_structured_columns(df, stats, cfg))
    """

    personas: list[DetectedPersona] = field(default_factory=list)
    """Persona groups with fields and demographic matchers."""
    standalone_columns: list[DetectedStandalone] = field(default_factory=list)
    """Columns replaced independently of any persona."""
    identified_not_replaced: list[str] = field(default_factory=list)
    """Column names detected but excluded from replacement (identify-only temporals)."""
    free_text_columns: list[str] = field(default_factory=list)
    """Prose columns selected for free-text PII propagation or LLM detection."""

    def to_dict(self) -> dict[str, object]:
        return {
            "personas": [p.to_dict() for p in self.personas],
            "standalone_columns": [s.to_dict() for s in self.standalone_columns],
            "identified_not_replaced": list(self.identified_not_replaced),
            "free_text_columns": list(self.free_text_columns),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DiscoveryResult:
        personas_raw = cast(Sequence[Mapping[str, object]], raw.get("personas") or [])
        standalone_raw = cast(Sequence[Mapping[str, object]], raw.get("standalone_columns") or [])
        identified = cast(Sequence[object], raw.get("identified_not_replaced") or [])
        free_text = cast(Sequence[object], raw.get("free_text_columns") or [])
        return cls(
            personas=[DetectedPersona.from_dict(p) for p in personas_raw],
            standalone_columns=[DetectedStandalone.from_dict(s) for s in standalone_raw],
            identified_not_replaced=[str(c) for c in identified],
            free_text_columns=[str(c) for c in free_text],
        )


@dataclass
class TextSubstitution:
    """One original-to-synthetic free-text rewrite with its taxonomy label."""

    original: str
    """Original substring found in the free-text cell."""
    synthetic: str
    """Replacement substring written back into the cell."""
    label: str
    """Entity label for the substituted span (e.g. ``last_name``)."""

    def as_pair(self) -> tuple[str, str]:
        return self.original, self.synthetic


@dataclass
class FreeTextDetection:
    """One entity span detected in an original free-text cell.

    Enhancers report evidence only. The replacement layer resolves the entity
    into a scoped synthetic value and applies it programmatically.

    Example:
        FreeTextDetection(row_index=3, column="notes", start=12, end=17, text="Smith", entity="last_name")
    """

    row_index: Hashable
    """Row index in the source dataframe."""
    column: str
    """Column name containing the span."""
    start: int
    """Start character offset of the span (inclusive)."""
    end: int
    """End character offset of the span (exclusive)."""
    text: str
    """Matched substring text."""
    entity: str
    """Entity label for the span (e.g. ``last_name``, ``phone_number``)."""
    confidence: float | None = None
    """Optional model confidence score; ``None`` when not provided."""


@dataclass
class PersonaInstance:
    """One persona-backed replacement unit (group, record, or dataframe signature).

    Carries the original field values and column map up front; after sampling,
    ``synthetic``, ``syn_by_col``, and ``text_pairs`` hold what to write back.

    Example:
        PersonaInstance(
            persona="patient",
            match=("g1",),
            field_cols={"first_name": "fname", "last_name": "lname"},
            patterns_by_label={},
            originals={"first_name": "Jane", "last_name": "Smith"},
        )
    """

    persona: str
    """Persona identifier shared by the grouped columns."""
    match: tuple[object, ...]
    """Scope key tuple (group value, row index, or empty for dataframe scope)."""
    field_cols: dict[str, str]
    """Map of entity label to column name for this persona instance."""
    patterns_by_label: dict[str, list[str]]
    """Per-label value templates or strftime formats for formatted fields."""
    originals: dict[str, str]
    """Original field values before replacement."""
    sex: str | None = None
    """Normalized sex/gender constraint for persona sampling, or ``None``."""
    race_raw: str | None = None
    """Raw ethnic-background value used to condition persona sampling, or ``None``."""
    select_field_values: dict[str, list[str]] | None = None
    """Optional per-field candidate lists for constrained sampling."""
    group_key: str | None = None
    """Group column name when scope is per-group; ``None`` otherwise."""
    row_indices: list[Hashable] = field(default_factory=list)
    """Row indices covered by this instance (record scope)."""
    synthetic_person: dict[str, object] | None = None
    """Sampled synthetic persona record from the backend, or ``None`` before sampling."""
    synthetic_person_source: str | None = None
    """Backend that produced ``synthetic_person`` (``managed``, ``pgm``, ``faker``)."""
    synthetic: dict[str, str] = field(default_factory=dict)
    """Synthetic field values keyed by entity label."""
    syn_by_col: dict[str, str] = field(default_factory=dict)
    """Synthetic values keyed by column name for write-back."""
    text_pairs: list[tuple[str, str]] = field(default_factory=list)
    """Original-to-synthetic pairs propagated into free-text columns."""

    # Mapping-style access so helpers can take either an instance or a plain dict
    # (hand-built fixtures, and call sites not yet migrated to attributes).
    def __getitem__(self, key: str) -> object:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: object) -> None:
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, object]:
        return {
            "persona": self.persona,
            "match": self.match,
            "field_cols": dict(self.field_cols),
            "patterns_by_label": {k: list(v) for k, v in self.patterns_by_label.items()},
            "originals": dict(self.originals),
            "sex": self.sex,
            "race_raw": self.race_raw,
            "select_field_values": (
                None if self.select_field_values is None else {k: list(v) for k, v in self.select_field_values.items()}
            ),
            "group_key": self.group_key,
            "row_indices": list(self.row_indices),
            "synthetic_person": self.synthetic_person,
            "synthetic_person_source": self.synthetic_person_source,
            "synthetic": dict(self.synthetic),
            "syn_by_col": dict(self.syn_by_col),
            "text_pairs": list(self.text_pairs),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> PersonaInstance:
        match_raw = cast(Sequence[object], raw["match"])
        field_cols = cast(Mapping[str, object], raw.get("field_cols") or {})
        patterns_by_label = cast(Mapping[str, object], raw.get("patterns_by_label") or {})
        originals = cast(Mapping[str, object], raw.get("originals") or {})
        sfv_raw = raw.get("select_field_values")
        select_field_values: dict[str, list[str]] | None
        if sfv_raw is None:
            select_field_values = None
        else:
            sfv_map = cast(Mapping[str, object], sfv_raw)
            select_field_values = {
                str(k): [str(v) for v in cast(Sequence[object], vals)] for k, vals in sfv_map.items()
            }
        synthetic_person_raw = raw.get("synthetic_person")
        synthetic_person = (
            None if synthetic_person_raw is None else dict(cast(Mapping[str, object], synthetic_person_raw))
        )
        synthetic = cast(Mapping[str, object], raw.get("synthetic") or {})
        syn_by_col = cast(Mapping[str, object], raw.get("syn_by_col") or {})
        text_pairs_raw = cast(Sequence[Sequence[object]], raw.get("text_pairs") or [])
        return cls(
            persona=str(raw["persona"]),
            match=tuple(match_raw),
            field_cols={str(k): str(v) for k, v in field_cols.items()},
            patterns_by_label={
                str(k): [str(p) for p in cast(Sequence[object], v)] for k, v in patterns_by_label.items()
            },
            originals={str(k): str(v) for k, v in originals.items()},
            sex=None if raw.get("sex") is None else str(raw.get("sex")),
            race_raw=None if raw.get("race_raw") is None else str(raw.get("race_raw")),
            select_field_values=select_field_values,
            group_key=None if raw.get("group_key") is None else str(raw.get("group_key")),
            row_indices=list(cast(Sequence[Hashable], raw.get("row_indices") or [])),
            synthetic_person=synthetic_person,
            synthetic_person_source=(
                None if raw.get("synthetic_person_source") is None else str(raw.get("synthetic_person_source"))
            ),
            synthetic={str(k): str(v) for k, v in synthetic.items()},
            syn_by_col={str(k): str(v) for k, v in syn_by_col.items()},
            text_pairs=[(str(a), str(b)) for a, b in text_pairs_raw],
        )


@dataclass
class ScopedValueMap:
    """A standalone column's original-to-synthetic map, tagged with scope kind.

    Example:
        ScopedValueMap("flat", {"001": "syn1"})
        ScopedValueMap("group", {"g1": {"001": "a"}})
        ScopedValueMap("record", {3: {"001": "b"}})
    """

    kind: ScopedValueMapKind
    """Scope kind: ``flat``, ``group``, or ``record``."""
    data: dict[Hashable, object]
    """Nested map of originals to synthetics; shape depends on ``kind``."""


# Back-compat alias used by call sites that still say StandaloneColMap.
StandaloneColMap = ScopedValueMap


@dataclass
class ReplacementOutcome:
    """Result of applying a resolved plan to a dataframe."""

    replaced_df: pd.DataFrame
    """Dataframe after PII replacement has been applied."""
    instances: list[PersonaInstance] = field(default_factory=list)
    """Persona instances created during replacement."""
    standalone_maps: dict[str, ScopedValueMap] = field(default_factory=dict)
    """Standalone column maps keyed by column name."""
    free_text_applied: list[str] = field(default_factory=list)
    """Free-text column names that received propagated substitutions."""
    details: dict[str, object] = field(default_factory=dict)
    """Optional diagnostic metadata from the replacement pass."""
