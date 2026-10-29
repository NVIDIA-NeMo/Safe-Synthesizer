# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection, scoped mapping reuse, and one-pass free-text replacement."""

from __future__ import annotations

import re
import time
from collections.abc import Hashable
from dataclasses import dataclass, field
from itertools import chain
from random import Random
from string import ascii_uppercase
from typing import Literal

import pandas as pd

from ...config.replace_pii import EntityType, PiiColumnPlan
from ...errors import GenerationError, InternalError, ParameterError, SafeSynthesizerError
from ..planning.patterns import extract_name_components
from ..transform_result import FreeTextReplacementRecord
from .addresses import AddressParts
from .collisions import CandidateChooser
from .common_words import COMMON_WORDS
from .detection import FreeTextDetector, fresh_detection_entity_types, resolve_overlapping_spans
from .generation import ReplacementGenerationRequest, ReplacementGenerator
from .letter_case import match_letter_case
from .person_names import parse_full_name
from .seeding import derive_seed
from .types import (
    CanonicalValue,
    DetectedSpan,
    DetectionCell,
    DetectionCellId,
    FreeTextMappingKey,
    GroupMappingKey,
    ProtectedSpan,
    RecordMappingKey,
    detected_text,
    free_text_mapping_key,
)

__all__ = [
    "FreeTextCellDetections",
    "FreeTextColumnStatistics",
    "FreeTextExecutionResult",
    "FreeTextReplacementExecutor",
]

_MAX_GENERATION_ATTEMPTS = 10
_MIN_KNOWN_VALUE_LENGTH = 3
# One name word: letters, optionally joined by an apostrophe or hyphen, as in "O'Brien" or "Smith-Jones".
_NAME_WORD = re.compile(r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*")
# One initial: a single letter, optionally followed by a period, as in "K" or "K.".
_NAME_INITIAL = re.compile(r"[^\W\d_]\.?")
_NAME_PART_ENTITY_TYPES = frozenset({EntityType.FIRST_NAME, EntityType.MIDDLE_NAME, EntityType.LAST_NAME})
# nameparser roles of the words of a full name that are replaced, and the name part each one is.
_NAME_ROLE_PART_TYPES = {
    "given": EntityType.FIRST_NAME,
    "middle": EntityType.MIDDLE_NAME,
    "family": EntityType.LAST_NAME,
}
# Roles of words kept unchanged, such as Dr. and Jr.
_KEPT_NAME_ROLES = frozenset({"title", "suffix"})

_KnownValueSource = Literal["structured_value", "detected_value"]


@dataclass(frozen=True, slots=True)
class _CellToReplace:
    """One free-text cell with the spans to replace in it."""

    cell: DetectionCell
    """The cell and its original text."""

    spans: tuple[DetectedSpan, ...]
    """Accepted spans in ascending offset order."""

    scope_identity: Hashable
    """Row position, or group identity when a grouping column is set."""


@dataclass(frozen=True, slots=True)
class _NameWord:
    """One word of a full name, its role, and where it is in the name."""

    text: str
    """The word exactly as written, including any trailing period. A last name keeps its particles, as in
    ``van Rossum``."""

    part_type: EntityType | None
    """First, middle, or last name, or ``None`` for a title or suffix kept unchanged."""

    start: int
    """Offset of the word in the full name."""

    end: int
    """Offset just after the word in the full name."""

    is_initial: bool = False
    """Whether the word is a single-letter initial, which is replaced by a letter and never registered."""


@dataclass(slots=True)
class FreeTextColumnStatistics:
    """Mutable aggregate statistics for one planned free-text column."""

    counts: dict[str, int] = field(default_factory=dict)
    """Accepted span count per entity type."""

    values: dict[str, set[str]] = field(default_factory=dict)
    """Distinct accepted span texts per entity type."""

    sources: set[str] = field(default_factory=set)
    """Detector sources that produced accepted spans."""

    kept_name_part_count: int = 0
    """Full-name replacements that kept an original first or last name because no other name was found."""


@dataclass(frozen=True, slots=True)
class FreeTextCellDetections:
    """Validated detector output for the original free-text cells, grouped by cell."""

    spans: dict[DetectionCellId, tuple[DetectedSpan, ...]] = field(default_factory=dict)
    """Accepted, non-overlapping spans to replace."""

    protected: dict[DetectionCellId, tuple[ProtectedSpan, ...]] = field(default_factory=dict)
    """Spans that known-value matches must not overlap."""


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
    consistent within record or group scope. With ``find_and_replace``, every
    cell is also searched for those structured and detected values, so
    occurrences the detector missed get the same replacement.

    Args:
        generator: Produces a replacement for each value that has no mapping yet.
        detector: Finds candidate spans in free-text cells, or ``None`` when the
            plan has no free-text targets; ``detect`` then fails if called with any.
        base_seed: Seed from which every replacement's deterministic seed is derived.
        capture_replacement_map: Whether to keep a per-span record of each
            original and replacement value in the execution result.
        find_and_replace: Whether to also replace occurrences of known
            structured and detected values that the detector missed.
        locale: ``replace_pii.locale``, whose name order full names in text are read in.
    """

    def __init__(
        self,
        generator: ReplacementGenerator,
        detector: FreeTextDetector | None,
        *,
        base_seed: int,
        capture_replacement_map: bool = False,
        find_and_replace: bool = False,
        locale: str = "en_US",
    ) -> None:
        self._generator = generator
        self._detector = detector
        self._locale = locale
        self._base_seed = base_seed
        self._capture_replacement_map = capture_replacement_map
        self._find_and_replace = find_and_replace
        self._cache: dict[FreeTextMappingKey, str] = {}
        # Cached name-part replacements by scope, part type, and the original word's case-folded first letter,
        # so an initial such as "L" can reuse the replacement of the one "Lovelace" seen in its scope.
        self._name_words_by_initial: dict[tuple[Hashable, EntityType, str], dict[str, str]] = {}
        self._statistics: dict[str, FreeTextColumnStatistics] = {}
        self._replacement_records: list[FreeTextReplacementRecord] = []
        self._generation_elapsed = 0.0
        self._generated_count = 0

    def reset(self) -> None:
        """Discard all mappings and aggregates from an earlier dataframe execution."""
        self._cache.clear()
        self._name_words_by_initial.clear()
        self._statistics.clear()
        self._replacement_records.clear()
        self._generation_elapsed = 0.0
        self._generated_count = 0

    def detect(
        self,
        dataframe: pd.DataFrame,
        specs: list[PiiColumnPlan],
    ) -> FreeTextCellDetections:
        """Detect and validate candidate and protected spans against complete original cells."""
        if not specs:
            return FreeTextCellDetections()
        if self._detector is None:
            raise GenerationError(
                "free-text replacement targets require the detector and span-resolution implementation"
            )

        cells = list(chain.from_iterable(_detection_cells(dataframe, spec) for spec in specs))
        requested = {cell.cell_id: cell for cell in cells}
        try:
            detected = self._detector.detect(cells)
            candidates = _validated_candidates(detected.spans, requested)
            protected = _validated_protected(detected.protected, requested)
        except SafeSynthesizerError:
            raise
        except Exception as exc:
            raise GenerationError("free-text PII detection returned invalid spans") from exc
        return FreeTextCellDetections(
            {cell_id: resolve_overlapping_spans(spans) for cell_id, spans in candidates.items()},
            {cell_id: tuple(spans) for cell_id, spans in protected.items()},
        )

    def execute(
        self,
        working: pd.DataFrame,
        specs: list[PiiColumnPlan],
        group_identities: tuple[CanonicalValue, ...] | None,
        detections: FreeTextCellDetections,
    ) -> FreeTextExecutionResult:
        """Apply accepted original-cell spans to every planned free-text target."""
        known_values = (
            self._known_values(working, specs, group_identities, detections.spans) if self._find_and_replace else {}
        )
        cells = {
            spec.column_name: self._cells_to_replace(working, spec, group_identities, detections, known_values)
            for spec in specs
        }
        # Every whole name word in a record or group is mapped before any initial is written, so an
        # initial takes its name's replacement letter wherever the name appears.
        for spec in specs:
            self._map_name_words(cells[spec.column_name], self._statistics_for(spec))
        scope = "record" if group_identities is None else "group"
        for spec in specs:
            self._write_cells(working, spec, cells[spec.column_name], scope)
        return FreeTextExecutionResult(
            column_statistics=self._statistics,
            generated_replacement_count=self._generated_count,
            generator_time_seconds=self._generation_elapsed,
            replacement_records=tuple(self._replacement_records),
        )

    def detect_statistics(
        self,
        dataframe: pd.DataFrame,
        specs: list[PiiColumnPlan],
        group_identities: tuple[CanonicalValue, ...] | None,
        detections: FreeTextCellDetections,
    ) -> dict[str, FreeTextColumnStatistics]:
        """Record the spans ``execute`` would replace in each free-text target, without replacing them.

        Structured mappings must be registered first, as ``execute`` expects, so known-value matches are the same.
        """
        known_values = (
            self._known_values(dataframe, specs, group_identities, detections.spans) if self._find_and_replace else {}
        )
        for spec in specs:
            statistics = self._statistics_for(spec)
            for to_replace in self._cells_to_replace(dataframe, spec, group_identities, detections, known_values):
                for span in to_replace.spans:
                    _record_detection(statistics, span, detected_text(to_replace.cell, span))
        return self._statistics

    def _known_values(
        self,
        working: pd.DataFrame,
        specs: list[PiiColumnPlan],
        group_identities: tuple[CanonicalValue, ...] | None,
        spans_by_cell: dict[DetectionCellId, tuple[DetectedSpan, ...]],
    ) -> dict[Hashable, _KnownValues]:
        """Collect the values to search for in each record or group before any free-text cell changes.

        Structured mappings registered so far come first, so a value that is
        both keeps its structured entity type. Detected values follow in plan
        and row order, and a detected plain full name also adds its words.
        """
        known: dict[Hashable, _KnownValues] = {}
        for key in self._cache:
            known.setdefault(key.scope_identity, _KnownValues()).add(
                key.original_value, key.entity_type, "structured_value"
            )
        for spec in specs:
            column_position = working.columns.get_loc(spec.column_name)
            if not isinstance(column_position, int):
                raise ParameterError(f"replacement column {spec.column_name!r} is duplicated in the dataframe")
            for row_position in range(len(working)):
                spans = spans_by_cell.get(DetectionCellId(row_position, spec.column_name), ())
                text = working.iat[row_position, column_position]
                if not spans or not isinstance(text, str):
                    continue
                scope_identity = row_position if group_identities is None else group_identities[row_position]
                _add_detected_values(known.setdefault(scope_identity, _KnownValues()), text, spans, self._locale)
        return known

    def register_structured_mapping(
        self,
        key: RecordMappingKey | GroupMappingKey,
        spec: PiiColumnPlan,
        original_text: str,
        replacement: str,
        address_parts: AddressParts | None = None,
    ) -> None:
        """Expose exact values, pattern-validated name parts, and street parts to accepted spans.

        ``original_text`` is the source cell's text form, the same value the
        structured generator received, so a typed original such as an integer
        API key matches the digits a detector finds in prose. ``key`` carries
        its typed identity. ``address_parts`` is the detected street of a
        street-address value whose locality was kept.
        """
        scope_identity = key.row_position if isinstance(key, RecordMappingKey) else key.original_group_identity
        self._register(scope_identity, spec.entity_type, original_text, replacement)
        self._register_name_components(scope_identity, spec, original_text, replacement)
        self._register_address_component(scope_identity, spec, replacement, address_parts)

    def _register_name_components(
        self,
        scope_identity: Hashable,
        spec: PiiColumnPlan,
        original: str,
        replacement: str,
    ) -> None:
        """Register first, middle, and last name parts of a structured full name.

        An explicit name pattern aligns the parts, except that a pattern with
        two placeholders next to each other, such as ``{First}{Last}``,
        registers nothing because the split would be a guess. Without a
        pattern, the name is split as described in ``_name_parts``.
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
        """Register each name word of a full name against the word of the same part in its replacement.

        Words pair by part, not position, so ``Lovelace, Ada`` replaced by
        ``Grace Hopper`` registers ``Ada`` as ``Grace`` and ``Lovelace`` as
        ``Hopper``. Middle names pair in order, and a part with a different
        number of words on each side is skipped. Titles and suffixes are
        skipped, so ``John Smith Jr`` replaced by ``Mark Brown`` registers
        ``John`` and ``Smith``. A pair where either side is an initial is
        skipped, because a one-letter mapping would replace every matching
        letter in later text.
        """
        original_parts = _name_words_by_part(original, self._locale)
        replacement_parts = _name_words_by_part(replacement, self._locale)
        for part_type, original_words in original_parts.items():
            replacement_words = replacement_parts.get(part_type, [])
            if len(original_words) != len(replacement_words):
                continue
            for original_word, replacement_word in zip(original_words, replacement_words, strict=True):
                if not (original_word.is_initial or replacement_word.is_initial):
                    self._register(scope_identity, part_type, original_word.text, replacement_word.text)

    def _register_address_component(
        self,
        scope_identity: Hashable,
        spec: PiiColumnPlan,
        replacement: str,
        address_parts: AddressParts | None,
    ) -> None:
        """Register the street of a structured address so the same street in free text gets the same replacement.

        Only a detected street whose original locality was kept is registered: the replacement then has the
        same locality text after a new street, so the two streets line up. A street address replaced whole
        has no such alignment and registers only as a whole value.
        """
        if spec.entity_type is not EntityType.STREET_ADDRESS or address_parts is None:
            return
        replacement_street = address_parts.replacement_street(replacement)
        if replacement_street is not None:
            self._register(scope_identity, EntityType.STREET_ADDRESS, address_parts.street, replacement_street)

    def _register(
        self,
        scope_identity: Hashable,
        entity_type: EntityType,
        original: str,
        replacement: str,
    ) -> None:
        """Keep the first scoped replacement for an exact entity/value pair."""
        self._store(FreeTextMappingKey(scope_identity, entity_type, original), replacement)

    def _store(self, key: FreeTextMappingKey, replacement: str) -> str:
        """Cache ``replacement`` unless ``key`` already has one, and return the cached value.

        Whole name words are also indexed by their first letter for ``_initial_replacement``.
        """
        stored = self._cache.setdefault(key, replacement)
        if key.entity_type in _NAME_PART_ENTITY_TYPES and not _NAME_INITIAL.fullmatch(key.original_value):
            index_key = (key.scope_identity, key.entity_type, key.folded_value[:1])
            self._name_words_by_initial.setdefault(index_key, {}).setdefault(key.folded_value, stored)
        return stored

    def _statistics_for(self, spec: PiiColumnPlan) -> FreeTextColumnStatistics:
        """Return the statistics of one free-text target, creating them on first use."""
        return self._statistics.setdefault(spec.column_name, FreeTextColumnStatistics())

    def _cells_to_replace(
        self,
        working: pd.DataFrame,
        spec: PiiColumnPlan,
        group_identities: tuple[CanonicalValue, ...] | None,
        detections: FreeTextCellDetections,
        known_values: dict[Hashable, _KnownValues],
    ) -> list[_CellToReplace]:
        """Return each cell of one free-text target with its accepted spans, in row order."""
        column_position = working.columns.get_loc(spec.column_name)
        if not isinstance(column_position, int):
            raise ParameterError(f"replacement column {spec.column_name!r} is duplicated in the dataframe")
        cells: list[_CellToReplace] = []
        for row_position in range(len(working)):
            original = working.iat[row_position, column_position]
            if not isinstance(original, str):
                continue
            cell_id = DetectionCellId(row_position, spec.column_name)
            spans = detections.spans.get(cell_id, ())
            scope_identity = row_position if group_identities is None else group_identities[row_position]
            if (values := known_values.get(scope_identity)) is not None:
                protected = detections.protected.get(cell_id, ())
                matches = [
                    match
                    for match in values.find(cell_id, original)
                    if not any(span.overlaps(match) for span in protected)
                ]
                spans = resolve_overlapping_spans((*spans, *matches))
            if spans:
                cell = DetectionCell(cell_id, original, fresh_detection_entity_types())
                cells.append(_CellToReplace(cell, spans, scope_identity))
        return cells

    def _map_name_words(self, cells: list[_CellToReplace], statistics: FreeTextColumnStatistics) -> None:
        """Map every whole first, middle, and last name word in ``cells``, leaving initials for the write pass.

        Example:
            In ``Ada L. called. Ada Lovelace agreed.``, ``Lovelace`` is mapped
            here, so the earlier ``L.`` takes the first letter of its replacement.
        """
        for to_replace in cells:
            for span in to_replace.spans:
                key = free_text_mapping_key(to_replace.cell, span, scope_identity=to_replace.scope_identity)
                if key.entity_type in _NAME_PART_ENTITY_TYPES:
                    self._replacement_for(key, statistics)
                elif key.entity_type is EntityType.FULL_NAME and key not in self._cache:
                    for word in _name_parts(key.original_value, self._locale) or ():
                        if word.part_type is not None and not word.is_initial:
                            self._name_word_replacement(key.scope_identity, word)

    def _write_cells(
        self,
        working: pd.DataFrame,
        spec: PiiColumnPlan,
        cells: list[_CellToReplace],
        scope: Literal["record", "group"],
    ) -> None:
        """Write the replaced text of ``cells`` into one free-text target column."""
        column_position = working.columns.get_loc(spec.column_name)
        if not isinstance(column_position, int):
            raise ParameterError(f"replacement column {spec.column_name!r} is duplicated in the dataframe")
        # A categorical column accepts only its existing categories, so rewritten text needs a plain object column.
        if cells and isinstance(working.iloc[:, column_position].dtype, pd.CategoricalDtype):
            working[spec.column_name] = working[spec.column_name].astype(object)
        statistics = self._statistics_for(spec)
        for to_replace in cells:
            working.iat[to_replace.cell.cell_id.row_position, column_position] = self._replace_cell(
                to_replace.cell, to_replace.spans, to_replace.scope_identity, scope, statistics
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
            replacement = match_letter_case(
                value,
                self._full_name_replacement(key, statistics)
                if key.entity_type is EntityType.FULL_NAME
                else self._replacement_for(key, statistics),
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

    def _full_name_replacement(self, key: FreeTextMappingKey, statistics: FreeTextColumnStatistics) -> str:
        """Replace a detected full name so its words agree with first, middle, and last names in the same scope.

        A full name already mapped, for example by a structured column, keeps
        that mapping and registers its words. Otherwise a name that
        ``_name_parts`` can split is built word by word from the first, middle,
        and last name mappings, generating any that are missing, so ``Kenneth
        Gonzales`` and a later bare ``Kenneth`` share one first-name replacement
        in either order. Titles and suffixes stay unchanged, and initials are
        replaced as described in ``_initial_replacement``. Any other full name
        is generated whole.

        Example:
            After ``Ada Lovelace`` became ``Grace Hopper`` in the same record,
            ``Dr Ada L.`` becomes ``Dr Grace H.``.
        """
        cached = self._cache.get(key)
        if cached is not None:
            self._register_name_parts(key.scope_identity, key.original_value, cached)
            return cached
        words = _name_parts(key.original_value, self._locale)
        if words is None:
            return self._replacement_for(key, statistics)
        # Text between the words, such as the comma in "Lovelace, Ada", is kept.
        pieces: list[str] = []
        cursor = 0
        for word in words:
            pieces += [key.original_value[cursor : word.start], self._name_word_replacement(key.scope_identity, word)]
            cursor = word.end
        pieces.append(key.original_value[cursor:])
        return self._store(key, "".join(pieces))

    def _name_word_replacement(self, scope_identity: Hashable, word: _NameWord) -> str:
        """Return the replacement for one word of a full name: unchanged, a new initial, or a shared name part."""
        if word.part_type is None:
            return word.text
        if word.is_initial:
            return self._initial_replacement(scope_identity, word.part_type, word.text)
        return self._replacement_for(FreeTextMappingKey(scope_identity, word.part_type, word.text))

    def _initial_replacement(self, scope_identity: Hashable, part_type: EntityType, initial: str) -> str:
        """Replace a single-letter initial with a different letter, keeping its letter case and any period.

        The new letter is, in order of preference, the first letter of the
        replacement of the only cached ``part_type`` word in the scope that
        starts with the same letter, the first letter of a replacement
        generated for the initial itself, or a seeded random letter. The new
        letter always differs from the original ignoring case.

        Example:
            With ``Lovelace`` replaced by ``Hopper`` as a last name in the same
            record, the last-name initial ``L.`` becomes ``H.``.
        """
        letter, period = initial[:1], initial[1:]
        new_letter = self._cached_word_letter(scope_identity, part_type, letter)
        if new_letter is None:
            key = FreeTextMappingKey(scope_identity, part_type, letter)
            generated = self._replacement_for(key)[:1]
            if _is_other_letter(generated, letter):
                new_letter = generated
            else:
                rng = Random(derive_seed(self._base_seed, key, purpose="free_text_initial", attempt=0))
                new_letter = rng.choice([candidate for candidate in ascii_uppercase if candidate != letter.upper()])
        return (new_letter.upper() if letter.isupper() else new_letter.lower()) + period

    def _cached_word_letter(self, scope_identity: Hashable, part_type: EntityType, letter: str) -> str | None:
        """Return the first letter of the only cached replacement for a word starting with ``letter``, if usable."""
        cached_words = self._name_words_by_initial.get((scope_identity, part_type, letter.casefold()), {})
        if len(cached_words) != 1:
            return None
        cached_letter = next(iter(cached_words.values()))[:1]
        return cached_letter if _is_other_letter(cached_letter, letter) else None

    def _replacement_for(self, key: FreeTextMappingKey, statistics: FreeTextColumnStatistics | None = None) -> str:
        """Return a cached value or generate one valid deterministic replacement.

        Candidates follow the same collision rules as structured columns: a replacement must differ
        from the detected text ignoring case, and a full name generated whole prefers one that keeps
        neither original name part, counting the fallback in ``statistics``.
        """
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        chooser = CandidateChooser(key.entity_type, key.original_value)
        replacement: str | None = None
        for attempt in range(_MAX_GENERATION_ATTEMPTS):
            request = ReplacementGenerationRequest(
                entity_type=key.entity_type,
                original_value=key.original_value,
                effective_dependency_tuple=(),
                pattern=None,
                seed=derive_seed(self._base_seed, key, purpose="free_text_replacement", attempt=attempt),
            )
            candidate = self._generate(request)
            if chooser.offer(request, candidate):
                replacement = candidate
                break
        else:
            if (fallback := chooser.fallback()) is not None:
                replacement, _ = fallback
                if statistics is not None:
                    statistics.kept_name_part_count += 1
        if replacement is not None:
            self._generated_count += 1
            return self._store(key, replacement)
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


def _name_parts(value: str, locale: str) -> tuple[_NameWord, ...] | None:
    """Split a full name into typed words with nameparser, or return ``None`` when the split would be a guess.

    nameparser reads the name in ``locale``'s order, and a comma marks the last name first. Titles and
    suffixes, such as ``Dr.`` and ``Jr.``, are kept with no part type, and a last name keeps its particles,
    such as ``van`` in ``van Rossum``. A single-letter initial, with or without a period, takes the part
    type nameparser gives it. A name without both a first and a last name, a nickname or maiden name, or a
    word with digits or other punctuation returns ``None``.

    Example:
        ``"Dr Ada K. Lovelace Jr"`` returns ``Dr`` and ``Jr`` with no part type, ``Ada`` as the first name,
        ``K.`` as a middle-name initial, and ``Lovelace`` as the last name.
    """
    words: list[_NameWord] = []
    for token in parse_full_name(value, locale=locale).tokens:
        if token.span is None:
            return None
        role = token.role.value if token.role is not None else None
        if role in _KEPT_NAME_ROLES:
            words.append(_NameWord(token.text, None, token.span.start, token.span.end))
            continue
        part_type = _NAME_ROLE_PART_TYPES.get(role or "")
        if part_type is None or not (_NAME_WORD.fullmatch(token.text) or _NAME_INITIAL.fullmatch(token.text)):
            return None
        previous = words[-1] if words else None
        if part_type is EntityType.LAST_NAME and previous is not None and previous.part_type is EntityType.LAST_NAME:
            words[-1] = _NameWord(value[previous.start : token.span.end], part_type, previous.start, token.span.end)
            continue
        is_initial = _NAME_INITIAL.fullmatch(token.text) is not None
        words.append(_NameWord(token.text, part_type, token.span.start, token.span.end, is_initial=is_initial))
    part_types = {word.part_type for word in words}
    if EntityType.FIRST_NAME not in part_types or EntityType.LAST_NAME not in part_types:
        return None
    return tuple(words)


def _name_words_by_part(value: str, locale: str) -> dict[EntityType, list[_NameWord]]:
    """Return the first, middle, and last name words of ``value`` by part, in order, without titles or suffixes."""
    by_part: dict[EntityType, list[_NameWord]] = {}
    for word in _name_parts(value, locale) or ():
        if word.part_type is not None:
            by_part.setdefault(word.part_type, []).append(word)
    return by_part


def _registrable_name_words(value: str, locale: str) -> tuple[tuple[EntityType, str], ...]:
    """Return the typed whole name words of ``value``, skipping titles, suffixes, and initials."""
    return tuple(
        (word.part_type, word.text)
        for word in _name_parts(value, locale) or ()
        if word.part_type is not None and not word.is_initial
    )


def _is_other_letter(candidate: str, letter: str) -> bool:
    """Return whether ``candidate`` is one letter that differs from ``letter`` ignoring case."""
    return len(candidate) == 1 and candidate.isalpha() and candidate.casefold() != letter.casefold()


def _add_detected_values(values: _KnownValues, text: str, spans: tuple[DetectedSpan, ...], locale: str) -> None:
    """Add each detected value in ``text``, and the words of a detected plain full name, as known values."""
    for span in spans:
        value = text[span.start : span.end]
        values.add(value, span.entity_type, "detected_value")
        if span.entity_type is EntityType.FULL_NAME:
            for part_type, word in _registrable_name_words(value, locale):
                values.add(word, part_type, "detected_value")


class _KnownValues:
    """Values known in one record or group, searched for as whole words ignoring case.

    Example:
        With ``Kenneth`` known as a first name, ``find`` on ``"kenneth called; Kenneths"``
        returns one ``first_name`` span for ``kenneth`` and none inside ``Kenneths``.
    """

    def __init__(self) -> None:
        self._values: dict[str, tuple[str, EntityType, _KnownValueSource]] = {}
        self._pattern: re.Pattern[str] | None = None

    def add(self, value: str, entity_type: EntityType, source: _KnownValueSource) -> None:
        """Keep the first entity type and source added for a value, skipping values unsafe to search for."""
        if not _is_searchable(value) or entity_type not in fresh_detection_entity_types():
            return
        if value.casefold() not in self._values:
            self._values[value.casefold()] = (value, entity_type, source)
            self._pattern = None

    def find(self, cell_id: DetectionCellId, text: str) -> tuple[DetectedSpan, ...]:
        """Return a span for each whole-word occurrence in ``text``, taking the longest value at each position."""
        if not self._values:
            return ()
        if self._pattern is None:
            values = sorted((value for value, _, _ in self._values.values()), key=len, reverse=True)
            alternatives = "|".join(re.escape(value) for value in values)
            self._pattern = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)
        spans: list[DetectedSpan] = []
        for match in self._pattern.finditer(text):
            # Case-insensitive matching and case folding disagree on a few characters, such as "ß"; skip those.
            if (known := self._values.get(match.group().casefold())) is not None:
                _, entity_type, source = known
                spans.append(DetectedSpan(cell_id, match.start(), match.end(), entity_type, source))
        return tuple(spans)


def _is_searchable(value: str) -> bool:
    """Return whether a known value is specific enough to search for in text.

    Values shorter than three characters, values with surrounding whitespace or
    no letter or digit, and one-word values in ``COMMON_WORDS``, such as
    ``Will`` or ``Brown``, are left to the detector.
    """
    if (
        len(value) < _MIN_KNOWN_VALUE_LENGTH
        or value != value.strip()
        or not any(character.isalnum() for character in value)
    ):
        return False
    return any(character.isspace() for character in value) or value.casefold() not in COMMON_WORDS


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


def _validated_protected(
    protected: tuple[ProtectedSpan, ...],
    requested: dict[DetectionCellId, DetectionCell],
) -> dict[DetectionCellId, list[ProtectedSpan]]:
    """Validate protected-span ownership and offsets before grouping them by cell."""
    by_cell: dict[DetectionCellId, list[ProtectedSpan]] = {}
    for span in protected:
        cell = requested.get(span.cell_id)
        if cell is None:
            raise InternalError("free-text PII detector returned a protected span for an unrequested cell")
        if span.end > len(cell.text):
            raise InternalError("protected span end exceeds the original cell length")
        by_cell.setdefault(span.cell_id, []).append(span)
    return by_cell


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
