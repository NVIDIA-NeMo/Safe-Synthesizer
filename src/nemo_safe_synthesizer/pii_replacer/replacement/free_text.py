# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection, scoped mapping reuse, and one-pass free-text replacement."""

from __future__ import annotations

import re
import time
from collections.abc import Hashable
from dataclasses import dataclass, field
from itertools import chain
from typing import Literal

import pandas as pd

from ...config.replace_pii import EntityType, PiiColumnPlan
from ...errors import GenerationError, InternalError, ParameterError, SafeSynthesizerError
from ..planning.patterns import extract_name_components
from ..transform_result import FreeTextReplacementRecord
from .detection import FreeTextDetector, fresh_detection_entity_types, resolve_overlapping_spans
from .generation import ReplacementGenerationRequest, ReplacementGenerator
from .seeding import derive_seed
from .types import (
    CanonicalValue,
    DetectedSpan,
    DetectionCell,
    DetectionCellId,
    EffectiveDependencyTuple,
    FreeTextMappingKey,
    GroupMappingKey,
    RecordMappingKey,
    detected_text,
    free_text_mapping_key,
)
from .value_checks import generated_value_is_valid

__all__ = ["FreeTextColumnStatistics", "FreeTextExecutionResult", "FreeTextReplacementExecutor"]

_MAX_GENERATION_ATTEMPTS = 10
# One name word: letters, optionally joined by an apostrophe or hyphen, as in "O'Brien" or "Smith-Jones".
_NAME_WORD = re.compile(r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*")
_NAME_PART_TYPES: dict[int, tuple[EntityType, ...]] = {
    2: (EntityType.FIRST_NAME, EntityType.LAST_NAME),
    3: (EntityType.FIRST_NAME, EntityType.MIDDLE_NAME, EntityType.LAST_NAME),
}


@dataclass(slots=True)
class FreeTextColumnStatistics:
    """Mutable aggregate statistics for one planned free-text column."""

    counts: dict[str, int] = field(default_factory=dict)
    """Accepted span count per entity type."""

    values: dict[str, set[str]] = field(default_factory=dict)
    """Distinct accepted span texts per entity type."""

    sources: set[str] = field(default_factory=set)
    """Detector sources that produced accepted spans."""


@dataclass(frozen=True, slots=True)
class FreeTextExecutionResult:
    """Aggregate outputs owned by one free-text execution pass."""

    column_statistics: dict[str, FreeTextColumnStatistics]
    """Detection statistics keyed by free-text column name."""

    generated_replacement_count: int
    """Replacements generated fresh rather than reused from a mapping."""

    generator_time_seconds: float
    """Time spent inside the generator for free-text spans."""

    replacement_records: tuple[FreeTextReplacementRecord, ...]
    """Per-span records when capture was requested, otherwise empty."""


class FreeTextReplacementExecutor:
    """Replace accepted spans while preserving scoped structured mappings.

    Detection always reads complete original cells before structured columns
    mutate. Replacement later uses those original offsets to build each cell
    exactly once. Structured mappings are registered through the same module so
    independently detected exact values and validated name components remain
    consistent within record or group scope.
    """

    def __init__(
        self,
        generator: ReplacementGenerator,
        detector: FreeTextDetector | None,
        *,
        base_seed: int,
        capture_replacement_map: bool = False,
    ) -> None:
        self._generator = generator
        self._detector = detector
        self._base_seed = base_seed
        self._capture_replacement_map = capture_replacement_map
        self._cache: dict[FreeTextMappingKey, str] = {}
        self._statistics: dict[str, FreeTextColumnStatistics] = {}
        self._replacement_records: list[FreeTextReplacementRecord] = []
        self._generation_elapsed = 0.0
        self._generated_count = 0

    def reset(self) -> None:
        """Discard all mappings and aggregates from an earlier dataframe execution."""
        self._cache.clear()
        self._statistics.clear()
        self._replacement_records.clear()
        self._generation_elapsed = 0.0
        self._generated_count = 0

    def detect(
        self,
        dataframe: pd.DataFrame,
        specs: list[PiiColumnPlan],
    ) -> dict[DetectionCellId, tuple[DetectedSpan, ...]]:
        """Detect and validate candidate spans against complete original cells."""
        if not specs:
            return {}
        if self._detector is None:
            raise GenerationError(
                "free-text replacement targets require the detector and span-resolution implementation"
            )

        cells = list(chain.from_iterable(_detection_cells(dataframe, spec) for spec in specs))
        requested = {cell.cell_id: cell for cell in cells}
        try:
            detected = self._detector.detect(cells)
            candidates = _validated_candidates(detected, requested)
        except SafeSynthesizerError:
            raise
        except Exception as exc:
            raise GenerationError("free-text PII detection returned invalid spans") from exc
        return {cell_id: resolve_overlapping_spans(spans) for cell_id, spans in candidates.items()}

    def execute(
        self,
        working: pd.DataFrame,
        specs: list[PiiColumnPlan],
        group_identities: tuple[CanonicalValue, ...] | None,
        spans_by_cell: dict[DetectionCellId, tuple[DetectedSpan, ...]],
    ) -> FreeTextExecutionResult:
        """Apply accepted original-cell spans to every planned free-text target."""
        for spec in specs:
            self._execute_spec(working, spec, group_identities, spans_by_cell)
        return FreeTextExecutionResult(
            column_statistics=self._statistics,
            generated_replacement_count=self._generated_count,
            generator_time_seconds=self._generation_elapsed,
            replacement_records=tuple(self._replacement_records),
        )

    def register_structured_mapping(
        self,
        key: RecordMappingKey | GroupMappingKey,
        spec: PiiColumnPlan,
        original_text: str,
        replacement: str,
        dependencies: EffectiveDependencyTuple,
    ) -> None:
        """Expose exact values and pattern-validated name parts to accepted spans.

        ``original_text`` is the source cell's text form, the same value the
        structured generator received, so a typed original such as an integer
        API key matches the digits a detector finds in prose. ``key`` carries
        its typed identity.
        """
        scope_identity = key.row_position if isinstance(key, RecordMappingKey) else key.original_group_identity
        self._register(scope_identity, spec.entity_type, original_text, replacement)
        self._register_name_components(scope_identity, spec, original_text, replacement)
        self._register_address_component(
            scope_identity,
            spec,
            original_text,
            replacement,
            dependencies,
        )

    def _register_name_components(
        self,
        scope_identity: Hashable,
        spec: PiiColumnPlan,
        original: str,
        replacement: str,
    ) -> None:
        """Register first, middle, and last name parts of a structured full name.

        An explicit name pattern aligns the parts. Without one, only a plain
        two- or three-word name is split, as described in ``_name_parts``.
        """
        if spec.entity_type is not EntityType.FULL_NAME:
            return
        if spec.pattern is None:
            self._register_name_parts(scope_identity, original, replacement)
            return
        original_parts = extract_name_components(spec.entity_type, spec.pattern, original)
        replacement_parts = extract_name_components(spec.entity_type, spec.pattern, replacement)
        if len(original_parts) != len(replacement_parts):
            return
        for (original_type, original_value), (replacement_type, replacement_value) in zip(
            original_parts,
            replacement_parts,
            strict=True,
        ):
            if original_type is replacement_type:
                self._register(scope_identity, original_type, original_value, replacement_value)

    def _register_name_parts(self, scope_identity: Hashable, original: str, replacement: str) -> None:
        """Register each word of a plain full name against the matching word of its replacement."""
        original_parts = _name_parts(original)
        replacement_parts = _name_parts(replacement)
        if original_parts is None or replacement_parts is None or len(original_parts) != len(replacement_parts):
            return
        for (part_type, original_word), (_, replacement_word) in zip(original_parts, replacement_parts, strict=True):
            self._register(scope_identity, part_type, original_word, replacement_word)

    def _register_address_component(
        self,
        scope_identity: Hashable,
        spec: PiiColumnPlan,
        original: str,
        replacement: str,
        dependencies: EffectiveDependencyTuple,
    ) -> None:
        """Register a street component only when typed dependencies prove suffix alignment."""
        if spec.entity_type is not EntityType.STREET_ADDRESS:
            return
        suffix = _address_dependency_suffix(dependencies)
        if not suffix:
            return
        original_street = _street_before_suffix(original, suffix)
        replacement_street = _street_before_suffix(replacement, suffix)
        if original_street is not None and replacement_street is not None:
            self._register(scope_identity, EntityType.STREET_ADDRESS, original_street, replacement_street)

    def _register(
        self,
        scope_identity: Hashable,
        entity_type: EntityType,
        original: str,
        replacement: str,
    ) -> None:
        """Keep the first scoped replacement for an exact entity/value pair."""
        key = FreeTextMappingKey(scope_identity, entity_type, original)
        self._cache.setdefault(key, replacement)

    def _execute_spec(
        self,
        working: pd.DataFrame,
        spec: PiiColumnPlan,
        group_identities: tuple[CanonicalValue, ...] | None,
        spans_by_cell: dict[DetectionCellId, tuple[DetectedSpan, ...]],
    ) -> None:
        """Replace all accepted cells for one free-text plan target."""
        column_position = working.columns.get_loc(spec.column_name)
        if not isinstance(column_position, int):
            raise ParameterError(f"replacement column {spec.column_name!r} is duplicated in the dataframe")
        statistics = self._statistics.setdefault(spec.column_name, FreeTextColumnStatistics())
        for row_position in range(len(working)):
            original = working.iat[row_position, column_position]
            if not isinstance(original, str):
                continue
            cell_id = DetectionCellId(row_position, spec.column_name)
            spans = spans_by_cell.get(cell_id, ())
            if spans:
                scope_identity = row_position if group_identities is None else group_identities[row_position]
                scope = "record" if group_identities is None else "group"
                working.iat[row_position, column_position] = self._replace_cell(
                    DetectionCell(cell_id, original, fresh_detection_entity_types()),
                    spans,
                    scope_identity,
                    scope,
                    statistics,
                )

    def _replace_cell(
        self,
        cell: DetectionCell,
        spans: tuple[DetectedSpan, ...],
        scope_identity: Hashable,
        scope: Literal["record", "group"],
        statistics: FreeTextColumnStatistics,
    ) -> str:
        """Construct one replacement cell from ascending original-text offsets."""
        parts: list[str] = []
        cursor = 0
        for span in spans:
            value = detected_text(cell, span)
            key = free_text_mapping_key(cell, span, scope_identity=scope_identity)
            replacement = _match_letter_case(
                value,
                self._full_name_replacement(key)
                if key.entity_type is EntityType.FULL_NAME
                else self._replacement_for(key),
            )
            parts.extend((cell.text[cursor : span.start], replacement))
            cursor = span.end
            _record_detection(statistics, span, value)
            if self._capture_replacement_map:
                self._replacement_records.append(
                    FreeTextReplacementRecord(
                        row_position=cell.cell_id.row_position,
                        column_name=cell.cell_id.column_name,
                        start=span.start,
                        end=span.end,
                        entity_type=span.entity_type,
                        detection_source=span.source,
                        score=span.score,
                        scope=scope,
                        original_value=value,
                        replacement_value=replacement,
                    )
                )
        parts.append(cell.text[cursor:])
        return "".join(parts)

    def _full_name_replacement(self, key: FreeTextMappingKey) -> str:
        """Replace a detected full name so its words agree with first, middle, and last names in the same scope.

        A full name already mapped, for example by a structured column, keeps
        that mapping and registers its words. Otherwise a plain two- or
        three-word name is built word by word from the first, middle, and last
        name mappings, generating any that are missing, so ``Kenneth Gonzales``
        and a later bare ``Kenneth`` share one first-name replacement in either
        order. Any other full name is generated whole.
        """
        cached = self._cache.get(key)
        if cached is not None:
            self._register_name_parts(key.scope_identity, key.original_value, cached)
            return cached
        parts = _name_parts(key.original_value)
        if parts is None:
            return self._replacement_for(key)
        replacement = " ".join(
            self._replacement_for(FreeTextMappingKey(key.scope_identity, part_type, word)) for part_type, word in parts
        )
        self._cache[key] = replacement
        return replacement

    def _replacement_for(self, key: FreeTextMappingKey) -> str:
        """Return a cached value or generate one valid deterministic replacement."""
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        for attempt in range(_MAX_GENERATION_ATTEMPTS):
            request = ReplacementGenerationRequest(
                entity_type=key.entity_type,
                original_value=key.original_value,
                effective_dependency_tuple=(),
                pattern=None,
                seed=derive_seed(self._base_seed, key, purpose="free_text_replacement", attempt=attempt),
            )
            candidate = self._generate(request)
            if generated_value_is_valid(request, candidate):
                self._cache[key] = candidate
                self._generated_count += 1
                return candidate
        raise GenerationError(
            f"PII replacement could not generate a distinct {key.entity_type.value!r} free-text value "
            f"after {_MAX_GENERATION_ATTEMPTS} attempts"
        )

    def _generate(self, request: ReplacementGenerationRequest) -> str:
        """Call the configured generator while recording aggregate elapsed time."""
        started = time.perf_counter()
        try:
            return self._generator.generate(request)
        except SafeSynthesizerError:
            raise
        except Exception as exc:
            raise GenerationError("PII replacement generation failed for an accepted free-text span") from exc
        finally:
            self._generation_elapsed += time.perf_counter() - started


def _name_parts(value: str) -> tuple[tuple[EntityType, str], ...] | None:
    """Split a plain full name into typed words, or return ``None`` when the split would be a guess.

    Only two or three words separated by single spaces qualify, read as first
    and last, or first, middle, and last. Titles, initials, suffixes, and
    digits do not qualify, so ``Dr. Ada Lovelace`` and ``Ada K. Lovelace``
    return ``None``.

    Example:
        ``"Kenneth Gonzales"`` returns ``((FIRST_NAME, "Kenneth"), (LAST_NAME, "Gonzales"))``.
    """
    words = value.split(" ")
    part_types = _NAME_PART_TYPES.get(len(words))
    if part_types is None or not all(_NAME_WORD.fullmatch(word) for word in words):
        return None
    return tuple(zip(part_types, words, strict=True))


def _match_letter_case(occurrence: str, replacement: str) -> str:
    """Give a shared replacement the letter case of the occurrence it replaces.

    Free-text reuse ignores case, so ``MARGARET`` and ``margaret`` share one generated replacement. An
    all-capitals or all-lowercase occurrence gets the replacement in the same case; any other occurrence keeps
    the replacement as generated.
    """
    if occurrence.isupper():
        return replacement.upper()
    if occurrence.islower():
        return replacement.lower()
    return replacement


def _detection_cells(dataframe: pd.DataFrame, spec: PiiColumnPlan) -> tuple[DetectionCell, ...]:
    """Build nonempty string cells for one target while preserving row positions."""
    column_position = dataframe.columns.get_loc(spec.column_name)
    if not isinstance(column_position, int):
        raise ParameterError(f"replacement column {spec.column_name!r} is duplicated in the dataframe")
    allowed = fresh_detection_entity_types()
    return tuple(
        DetectionCell(DetectionCellId(row_position, spec.column_name), value, allowed)
        for row_position in range(len(dataframe))
        if isinstance((value := dataframe.iat[row_position, column_position]), str) and value
    )


def _validated_candidates(
    detected: tuple[DetectedSpan, ...],
    requested: dict[DetectionCellId, DetectionCell],
) -> dict[DetectionCellId, list[DetectedSpan]]:
    """Validate detector ownership and offsets before grouping candidate spans."""
    candidates: dict[DetectionCellId, list[DetectedSpan]] = {}
    for span in detected:
        cell = requested.get(span.cell_id)
        if cell is None:
            raise InternalError("free-text PII detector returned a span for an unrequested cell")
        detected_text(cell, span)
        candidates.setdefault(span.cell_id, []).append(span)
    return candidates


def _record_detection(statistics: FreeTextColumnStatistics, span: DetectedSpan, value: str) -> None:
    """Add one accepted span to aggregate, PII-safe column statistics."""
    entity = span.entity_type.value
    statistics.counts[entity] = statistics.counts.get(entity, 0) + 1
    statistics.values.setdefault(entity, set()).add(value)
    statistics.sources.add(span.source)


def _address_dependency_suffix(dependencies: EffectiveDependencyTuple) -> str:
    """Render the typed locality suffix shared by structured address generation."""
    values = {entity_type: value.normalized_value for entity_type, value in dependencies if value is not None}
    locality = (
        values.get(EntityType.CITY),
        values.get(EntityType.STATE),
        values.get(EntityType.ZIPCODE),
        values.get(EntityType.COUNTRY),
    )
    return ", ".join(value for value in locality if value)


def _street_before_suffix(value: str, suffix: str) -> str | None:
    """Return a nonempty street prefix only when the complete typed suffix matches."""
    marker = f", {suffix}"
    if not value.endswith(marker):
        return None
    street = value[: -len(marker)]
    return street or None
