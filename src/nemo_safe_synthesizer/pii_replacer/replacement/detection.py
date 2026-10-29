# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Free-text PII detection and deterministic span resolution."""

from __future__ import annotations

import ipaddress
import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import NamedTuple, Protocol
from unicodedata import normalize

import regex
from stdnum import luhn
from stdnum.us import ssn as us_ssn
from typing_extensions import TypeIs

from ...config.replace_pii import (
    ENTITIES,
    GLINER_DETECTION_ENTITY_TYPES,
    REGEX_DETECTION_ENTITY_TYPES,
    EntityType,
    FreeTextDetectionConfig,
)
from ...errors import GenerationError, InternalError, ParameterError, SafeSynthesizerError
from ...observability import get_logger, heartbeat
from .birth_dates import birth_date_is_supported
from .types import DetectedSpan, DetectionCell, DetectionCellId, FreeTextDetections, ProtectedSpan

logger = get_logger(__name__)

_DEFAULT_REGEX_TIMEOUT_SECONDS = 0.05
_DEFAULT_MAX_MATCHES_PER_RULE = 1000

__all__ = [
    "CompositeFreeTextDetector",
    "FreeTextDetector",
    "Gliner2Detector",
    "RegexDetector",
    "resolve_overlapping_spans",
]

_FRESH_ENTITY_TYPES = frozenset(GLINER_DETECTION_ENTITY_TYPES + REGEX_DETECTION_ENTITY_TYPES)

# The selected PII checkpoint uses a slightly broader vocabulary than NSS. Only
# labels with a well-defined v3 replacement type are requested and normalized.
# Labels and default thresholds follow the nss-pii-ner-eval v6 study. Request only labels from the
# checkpoint's fine-tuning label list: a label outside it competes with the trained labels and collapses
# their recall. Identifiers are requested only by the kinds that prose names ("customer ID 1234");
# generic labels such as "identifier" or "unique_id" mostly fire on device, vehicle, and biometric codes
# and stay unrequested.
_GLINER_LABELS: dict[str, EntityType] = {
    "full_name": EntityType.FULL_NAME,
    "first_name": EntityType.FIRST_NAME,
    "middle_name": EntityType.MIDDLE_NAME,
    "last_name": EntityType.LAST_NAME,
    "phone_number": EntityType.PHONE_NUMBER,
    "fax_number": EntityType.PHONE_NUMBER,
    "date_of_birth": EntityType.DATE_OF_BIRTH,
    "street_address": EntityType.STREET_ADDRESS,
    "government_id": EntityType.NATIONAL_ID,
    "national_id_number": EntityType.NATIONAL_ID,
    "passport_number": EntityType.NATIONAL_ID,
    "tax_id": EntityType.NATIONAL_ID,
    "tax_number": EntityType.NATIONAL_ID,
    "api_key": EntityType.API_KEY,
    "customer_id": EntityType.UNIQUE_IDENTIFIER,
    "employee_id": EntityType.UNIQUE_IDENTIFIER,
    "medical_record_number": EntityType.UNIQUE_IDENTIFIER,
    "member_id": EntityType.UNIQUE_IDENTIFIER,
    "health_plan_beneficiary_number": EntityType.UNIQUE_IDENTIFIER,
    "certificate_number": EntityType.UNIQUE_IDENTIFIER,
}
# The v6 tuning gave government_id spans that pass the US SSN check a far lower threshold than other
# national IDs, since the check rather than the score tells SSNs apart. Kept until evals retune national_id.
_VALID_SSN_LABEL = "government_id"
_VALID_SSN_THRESHOLD = 0.050
_SSN_SHAPE = regex.compile(r"[0-9]{3}-[0-9]{2}-[0-9]{4}")
_NAME_ENTITY_TYPES = frozenset(
    {EntityType.FIRST_NAME, EntityType.MIDDLE_NAME, EntityType.LAST_NAME, EntityType.FULL_NAME}
)
# Characters allowed between a middle name and its neighbouring first or last name.
_MIDDLE_NAME_NEIGHBOR_GAP = 3
# A word as GLiNER2's default splitter tokenizes it: word characters, combining marks, and zero-width
# joiners, with inner hyphens or underscores. GLiNER2 spans start and end on these words within a chunk.
_GLINER_WORD = regex.compile(r"[\w\p{M}\u200c\u200d]+(?:[-_][\w\p{M}\u200c\u200d]+)*")
_ENTITY_ORDER = {entity.entity_type: position for position, entity in enumerate(ENTITIES)}
_SOURCE_ORDER = {"regex": 0, "gliner": 1, "structured_value": 2, "detected_value": 3}


class FreeTextDetector(Protocol):
    """Detect and resolve PII spans for complete original dataframe cells."""

    def detect(self, cells: Sequence[DetectionCell]) -> FreeTextDetections:
        """Return accepted, non-overlapping spans for ``cells`` and the spans replacement must leave unchanged."""


class _GlinerModel(Protocol):
    def batch_extract_entities(
        self,
        texts: list[str],
        labels: dict[str, dict[str, float]],
        **kwargs: object,
    ) -> list[object]:
        """Return one raw entity result per input text."""


class _RegexMatch(Protocol):
    def group(self, group: int = 0) -> str:
        """Return the matched text."""

    def start(self) -> int:
        """Return the match start offset."""

    def end(self) -> int:
        """Return the match end offset."""


class _CompiledRegex(Protocol):
    def finditer(self, string: str, *, timeout: float) -> Iterable[_RegexMatch]:
        """Yield matches in ``string``, raising ``TimeoutError`` after ``timeout`` seconds."""


@dataclass(frozen=True, slots=True)
class _Chunk:
    """One bounded window of a unique cell text sent to GLiNER2."""

    text: str
    """Window text passed to the model."""

    original_text: str
    """Complete cell text the window was cut from."""

    offset: int
    """Start of the window within ``original_text``."""

    words: tuple[tuple[int, int], ...]
    """Ascending offsets of every word in ``original_text``."""


class _TextSpan(NamedTuple):
    """One accepted GLiNER2 span in a unique cell text, before it is copied to cells."""

    start: int
    """Inclusive start offset in the complete cell text."""

    end: int
    """Exclusive end offset in the complete cell text."""

    entity_type: EntityType
    """Normalized NSS entity type."""

    score: float
    """GLiNER2 confidence score."""


@dataclass(slots=True)
class _TextCandidates:
    """Accepted entity spans and name-exclusion spans for one unique cell text."""

    spans: list[_TextSpan] = field(default_factory=list)
    """Entity spans that passed their thresholds and validators."""

    exclusions: list[tuple[int, int]] = field(default_factory=list)
    """Offsets of exclusion-label spans that passed their thresholds."""


class Gliner2Detector:
    """Lazy local GLiNER2 adapter with positional chunk remapping.

    The model is loaded on the first ``detect`` call and reused afterwards.
    Long texts are split into overlapping chunks, and detected offsets are
    mapped back to the original text.

    Args:
        config: Detection settings: model ID and revision, entity and
            name-exclusion thresholds, chunk geometry, and batch size.
        model_loader: Callable that loads a model from a model ID and an
            optional revision. Defaults to loading the local GLiNER2
            checkpoint; tests pass a fake.

    Raises:
        ParameterError: If a name-exclusion threshold label is also a GLiNER2
            detection label.
    """

    def __init__(
        self,
        config: FreeTextDetectionConfig,
        *,
        model_loader: Callable[[str, str | None], _GlinerModel] | None = None,
    ) -> None:
        colliding = sorted(set(_exclusion_thresholds(config)) & set(_GLINER_LABELS))
        if colliding:
            raise ParameterError(
                f"free_text_detection.name_exclusion_thresholds cannot use GLiNER2 detection labels: {colliding}"
            )
        self._config = config
        self._model_loader = model_loader or _load_gliner2_model
        self._model: _GlinerModel | None = None

    def detect(self, cells: Sequence[DetectionCell]) -> FreeTextDetections:
        """Run GLiNER2 once per unique text and copy valid spans to matching cells.

        Within each unique text, name spans that overlap a name-exclusion span
        are dropped, then the middle-name neighbour rule is applied, both before
        per-cell entity filtering so every cell sees the same model decision.
        Name-exclusion spans are also returned as protected spans.
        """
        cells_by_text = _cells_by_text(cells)
        if not cells_by_text:
            return FreeTextDetections(())

        chunks = list(chain.from_iterable(_chunks(text, self._config) for text in cells_by_text))
        results = self._infer(chunks, _requested_labels(cells, self._config))
        candidates: dict[str, _TextCandidates] = defaultdict(_TextCandidates)
        rejected_birth_dates = 0
        for chunk, result in zip(chunks, results, strict=True):
            rejected_birth_dates += _collect_chunk_candidates(
                chunk, result, candidates[chunk.original_text], self._config
            )
        spans: list[DetectedSpan] = []
        protected: list[ProtectedSpan] = []
        for text, text_cells in cells_by_text.items():
            for span in _filter_name_spans(candidates[text], self._config):
                spans.extend(_copy_span_to_cells(span, text_cells))
            # Overlapping chunks can report the same exclusion span twice.
            for start, end in sorted(set(candidates[text].exclusions)):
                protected.extend(ProtectedSpan(cell.cell_id, start, end) for cell in text_cells)
        if rejected_birth_dates:
            logger.user.warning(
                "GLiNER2 birth-date candidates were ignored because they were not complete parseable dates",
                extra={"rejected_candidate_count": rejected_birth_dates},
            )
        return FreeTextDetections(tuple(spans), tuple(protected))

    def _infer(self, chunks: list[_Chunk], labels: dict[str, dict[str, float]]) -> Sequence[object]:
        """Run one bounded GLiNER2 batch call and validate its outer result shape."""
        try:
            model = self._get_model()
            with heartbeat(
                "GLiNER2 free-text PII inference",
                interval=30.0,
                logger_name=__name__,
                chunk_count=len(chunks),
                batch_size=self._config.batch_size,
                requested_label_count=len(labels),
            ):
                results = model.batch_extract_entities(
                    [chunk.text for chunk in chunks],
                    labels,
                    threshold=min(self._config.entity_thresholds.values()),
                    include_confidence=True,
                    include_spans=True,
                    batch_size=self._config.batch_size,
                )
        except SafeSynthesizerError:
            raise
        except Exception as exc:
            raise GenerationError("GLiNER2 free-text PII inference failed") from exc
        if not isinstance(results, Sequence) or isinstance(results, (str, bytes)) or len(results) != len(chunks):
            raise GenerationError("GLiNER2 free-text PII inference returned an invalid batch result")
        return results

    def _get_model(self) -> _GlinerModel:
        """Load the configured local checkpoint once and reuse it across calls."""
        if self._model is None:
            try:
                self._model = self._model_loader(self._config.model_id, self._config.model_revision)
            except SafeSynthesizerError:
                raise
            except Exception as exc:
                raise GenerationError("GLiNER2 free-text PII model could not be loaded") from exc
        return self._model


@dataclass(frozen=True, slots=True)
class _RegexRule:
    """One built-in regex detector rule."""

    rule_id: str
    """Stable identifier used in error messages."""

    entity_type: EntityType
    """Entity type assigned to accepted matches."""

    expression: _CompiledRegex
    """Candidate pattern, run with a timeout."""

    validator: Callable[[str], bool]
    """Structural check a candidate must pass to become a span."""


class RegexDetector:
    """Bounded deterministic detector for structurally validated PII.

    Each built-in rule runs with a timeout and a cap on matches per cell, so
    an adversarial cell cannot stall or flood detection, and each match must
    pass the rule's structural validator, such as a Luhn check for card
    numbers, to become a span.

    Args:
        timeout_seconds: Time limit for one rule on one cell.
        max_matches_per_rule: Largest number of matches one rule may produce
            for one cell before detection fails.

    Raises:
        InternalError: If ``timeout_seconds`` or ``max_matches_per_rule`` is
            not positive.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = _DEFAULT_REGEX_TIMEOUT_SECONDS,
        max_matches_per_rule: int = _DEFAULT_MAX_MATCHES_PER_RULE,
    ) -> None:
        if timeout_seconds <= 0:
            raise InternalError("regex detector timeout_seconds must be positive")
        if max_matches_per_rule <= 0:
            raise InternalError("regex detector max_matches_per_rule must be positive")
        self._timeout_seconds = timeout_seconds
        self._max_matches_per_rule = max_matches_per_rule

    def detect(self, cells: Sequence[DetectionCell]) -> FreeTextDetections:
        """Return exact spans from applicable built-in rules. Regex rules protect nothing."""
        return FreeTextDetections(tuple(chain.from_iterable(self._detect_cell(cell) for cell in cells)))

    def _detect_cell(self, cell: DetectionCell) -> tuple[DetectedSpan, ...]:
        """Run every applicable bounded regex rule against one complete cell."""
        applicable_rules = (rule for rule in _REGEX_RULES if rule.entity_type in cell.allowed_entity_types)
        return tuple(chain.from_iterable(self._detect_rule(cell, rule) for rule in applicable_rules))

    def _detect_rule(self, cell: DetectionCell, rule: _RegexRule) -> tuple[DetectedSpan, ...]:
        """Apply one regex rule with its timeout, match cap, and structural validator."""
        try:
            matches = rule.expression.finditer(cell.text, timeout=self._timeout_seconds)
            return self._validated_spans(cell, rule, matches)
        except TimeoutError as exc:
            raise GenerationError(f"built-in PII regex rule {rule.rule_id!r} timed out") from exc

    def _validated_spans(
        self, cell: DetectionCell, rule: _RegexRule, matches: Iterable[_RegexMatch]
    ) -> tuple[DetectedSpan, ...]:
        """Return the spans of ``matches`` that pass the rule's validator, enforcing the match cap.

        ``matches`` is consumed lazily, so a ``TimeoutError`` from the regex
        engine propagates to the caller.
        """
        spans: list[DetectedSpan] = []
        for match_count, match in enumerate(matches, start=1):
            if match_count > self._max_matches_per_rule:
                raise GenerationError(
                    f"built-in PII regex rule {rule.rule_id!r} exceeded the maximum of "
                    f"{self._max_matches_per_rule} matches for one cell"
                )
            if span := _validated_regex_span(cell, rule, match):
                spans.append(span)
        return tuple(spans)


class CompositeFreeTextDetector:
    """Combine GLiNER2 and regex detections through one overlap policy."""

    def __init__(
        self,
        config: FreeTextDetectionConfig,
        *,
        gliner: FreeTextDetector | None = None,
        regex_detector: FreeTextDetector | None = None,
    ) -> None:
        self._gliner = gliner or Gliner2Detector(config)
        self._regex = regex_detector or RegexDetector()

    def detect(self, cells: Sequence[DetectionCell]) -> FreeTextDetections:
        """Return globally resolved spans and the protected spans from both mandatory detector sources."""
        gliner = self._gliner.detect(cells)
        regex = self._regex.detect(cells)
        by_cell: dict[DetectionCellId, list[DetectedSpan]] = defaultdict(list)
        for span in (*gliner.spans, *regex.spans):
            by_cell[span.cell_id].append(span)
        accepted: list[DetectedSpan] = []
        for cell in cells:
            accepted.extend(resolve_overlapping_spans(by_cell.get(cell.cell_id, ())))
        return FreeTextDetections(tuple(accepted), (*gliner.protected, *regex.protected))


def _cells_by_text(cells: Sequence[DetectionCell]) -> dict[str, list[DetectionCell]]:
    """Group cells by exact text so GLiNER inference runs once per unique value."""
    grouped: dict[str, list[DetectionCell]] = defaultdict(list)
    for cell in cells:
        if cell.text:
            grouped[cell.text].append(cell)
    return grouped


def _requested_labels(
    cells: Sequence[DetectionCell],
    config: FreeTextDetectionConfig,
) -> dict[str, dict[str, float]]:
    """Return applicable checkpoint labels with their effective thresholds.

    Name-exclusion labels are requested only when some cell allows a name type.
    """
    allowed = frozenset(chain.from_iterable(cell.allowed_entity_types for cell in cells))
    labels = {
        label: {"threshold": _model_label_threshold(label, config)}
        for label, entity_type in sorted(_GLINER_LABELS.items())
        if entity_type in allowed
    }
    if allowed & _NAME_ENTITY_TYPES:
        labels.update(
            {label: {"threshold": threshold} for label, threshold in sorted(_exclusion_thresholds(config).items())}
        )
    return labels


def _exclusion_thresholds(config: FreeTextDetectionConfig) -> dict[str, float]:
    """Return name-exclusion thresholds keyed by normalized checkpoint label."""
    return {_normalize_label(label): threshold for label, threshold in config.name_exclusion_thresholds.items()}


def _collect_chunk_candidates(
    chunk: _Chunk,
    result: object,
    candidates: _TextCandidates,
    config: FreeTextDetectionConfig,
) -> int:
    """Add one chunk's accepted spans to its text's candidates and return the rejected-DOB count."""
    exclusion_thresholds = _exclusion_thresholds(config)
    rejected_birth_dates = 0
    for raw_span in _result_spans(result):
        start, end, label, score = raw_span
        exclusion_threshold = exclusion_thresholds.get(_normalize_label(label))
        if exclusion_threshold is not None:
            _validate_score(score)
            offsets = _original_offsets(chunk, start, end)
            if offsets is not None and score >= exclusion_threshold:
                candidates.exclusions.append(offsets)
            continue
        normalized, rejected_birth_date = _normalize_chunk_span(chunk, raw_span, config)
        rejected_birth_dates += int(rejected_birth_date)
        if normalized is not None:
            candidates.spans.append(normalized)
    return rejected_birth_dates


def _normalize_chunk_span(
    chunk: _Chunk,
    raw_span: tuple[int, int, str, float],
    config: FreeTextDetectionConfig,
) -> tuple[_TextSpan | None, bool]:
    """Validate offsets and normalize one checkpoint label to an NSS entity type."""
    start, end, label, score = raw_span
    _validate_score(score)
    offsets = _original_offsets(chunk, start, end)
    if offsets is None:
        return None, False
    original_start, original_end = offsets
    value = chunk.original_text[original_start:original_end]
    normalized_label = _normalize_label(label)
    entity_type = _GLINER_LABELS.get(normalized_label)
    if entity_type is None:
        return None, False
    threshold = config.entity_thresholds[entity_type]
    if normalized_label == _VALID_SSN_LABEL and _is_valid_ssn(value):
        threshold = min(threshold, _VALID_SSN_THRESHOLD)
    if score < threshold:
        return None, False

    rejected_birth_date = entity_type is EntityType.DATE_OF_BIRTH and not birth_date_is_supported(value)
    if rejected_birth_date:
        return None, rejected_birth_date
    return _TextSpan(original_start, original_end, entity_type, score), False


def _validate_score(score: float) -> None:
    """Reject confidence scores outside the closed unit interval."""
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise GenerationError("GLiNER2 free-text PII inference returned an invalid confidence score")


def _original_offsets(chunk: _Chunk, start: int, end: int) -> tuple[int, int] | None:
    """Map chunk offsets to the complete cell, or ``None`` for an empty repaired span."""
    # GLiNER2's inference collator appends one period to nonempty input that
    # does not already end in '.', '!', or '?'. A prediction may therefore
    # include that synthetic character even though it is absent from the
    # original chunk whose offsets NSS owns.
    if end == len(chunk.text) + 1 and chunk.text and not chunk.text.endswith((".", "!", "?")):
        end = len(chunk.text)
        if start == end:
            return None

    if start < 0 or end <= start or end > len(chunk.text):
        raise GenerationError("GLiNER2 free-text PII inference returned invalid span offsets")
    original_start = chunk.offset + start
    original_end = chunk.offset + end
    if original_end > len(chunk.original_text):
        raise GenerationError("GLiNER2 free-text PII inference returned invalid span offsets")
    # A word too long to fit one chunk is still cut at a chunk edge, and the model can label the
    # fragment; replacing it would rewrite part of a word.
    if _cuts_word(chunk.words, original_start) or _cuts_word(chunk.words, original_end):
        return None
    return original_start, original_end


def _filter_name_spans(candidates: _TextCandidates, config: FreeTextDetectionConfig) -> list[_TextSpan]:
    """Drop names overlapping an exclusion span, then middle names without neighbouring name pieces."""
    kept = [span for span in candidates.spans if not _is_excluded_name(span, candidates.exclusions)]
    if config.require_middle_name_neighbors:
        kept = [
            span for span in kept if span.entity_type is not EntityType.MIDDLE_NAME or _has_name_neighbors(span, kept)
        ]
    return kept


def _is_excluded_name(span: _TextSpan, exclusions: Sequence[tuple[int, int]]) -> bool:
    """Return whether a name span overlaps any accepted exclusion span."""
    return span.entity_type in _NAME_ENTITY_TYPES and any(
        span.start < end and start < span.end for start, end in exclusions
    )


def _has_name_neighbors(middle: _TextSpan, spans: Sequence[_TextSpan]) -> bool:
    """Return whether a middle name sits inside a full name or between a first and last name."""
    inside_full_name = any(
        span.entity_type is EntityType.FULL_NAME
        and span.start <= middle.start
        and middle.end <= span.end
        and (span.start, span.end) != (middle.start, middle.end)
        for span in spans
    )
    first_before = any(
        span.entity_type is EntityType.FIRST_NAME and 0 <= middle.start - span.end <= _MIDDLE_NAME_NEIGHBOR_GAP
        for span in spans
    )
    last_after = any(
        span.entity_type is EntityType.LAST_NAME and 0 <= span.start - middle.end <= _MIDDLE_NAME_NEIGHBOR_GAP
        for span in spans
    )
    return inside_full_name or (first_before and last_after)


def _model_label_threshold(label: str, config: FreeTextDetectionConfig) -> float:
    """Return a safe model-side floor for one checkpoint label."""
    normalized_label = _normalize_label(label)
    threshold = config.entity_thresholds[_GLINER_LABELS[normalized_label]]
    # Valid SSNs are accepted below the national_id threshold, so inference must keep them.
    return min(threshold, _VALID_SSN_THRESHOLD) if normalized_label == _VALID_SSN_LABEL else threshold


def _copy_span_to_cells(span: _TextSpan, cells: list[DetectionCell]) -> tuple[DetectedSpan, ...]:
    """Copy one unique-text result to cells that allow its normalized entity type."""
    return tuple(
        DetectedSpan(cell.cell_id, span.start, span.end, span.entity_type, "gliner", span.score)
        for cell in cells
        if span.entity_type in cell.allowed_entity_types
    )


def _validated_regex_span(
    cell: DetectionCell,
    rule: _RegexRule,
    match: _RegexMatch,
) -> DetectedSpan | None:
    """Convert a structurally valid regex match to a detector span."""
    value = match.group(0)
    if not rule.validator(value):
        return None
    return DetectedSpan(cell.cell_id, match.start(), match.end(), rule.entity_type, "regex")


def resolve_overlapping_spans(spans: Iterable[DetectedSpan]) -> tuple[DetectedSpan, ...]:
    """Resolve overlapping candidates, keeping longer spans first, then more confident ones.

    Spans are ranked by length, then confidence (regex spans count as 1.0 and
    known-value matches as 0.0), then position, then source, GLiNER2 score, and entity order, and each is
    kept unless it overlaps an already kept span. This is the rule
    nss-pii-ner-eval scores with.

    Example:
        Two overlapping five-character spans at offsets 0 and 2 with scores
        0.6 and 0.9 keep the span at offset 2.
    """
    best_duplicates: dict[tuple[DetectionCellId, EntityType, int, int], DetectedSpan] = {}
    for span in spans:
        key = (span.cell_id, span.entity_type, span.start, span.end)
        current = best_duplicates.get(key)
        if current is None or _tie_rank(span) < _tie_rank(current):
            best_duplicates[key] = span

    ranked = sorted(
        best_duplicates.values(),
        key=lambda span: (
            -(span.end - span.start),
            -_confidence(span),
            span.start,
            span.end,
            *_tie_rank(span),
        ),
    )
    accepted: list[DetectedSpan] = []
    for candidate in ranked:
        if not any(_overlaps(candidate, existing) for existing in accepted):
            accepted.append(candidate)
    return tuple(
        sorted(accepted, key=lambda span: (span.cell_id.row_position, span.start, span.end, span.entity_type.value))
    )


def fresh_detection_entity_types() -> frozenset[EntityType]:
    """Return entity types eligible for fresh replacement inside free text."""
    return _FRESH_ENTITY_TYPES


def _confidence(span: DetectedSpan) -> float:
    """Return a span's confidence, treating structurally validated regex matches as certain.

    Known-value matches have no score and count as 0.0, so a detector span of equal length wins.
    """
    if span.source == "regex":
        return 1.0
    return span.score if span.score is not None else 0.0


def _tie_rank(span: DetectedSpan) -> tuple[int, float, int]:
    """Rank exact span ties by source, confidence, and stable entity order."""
    confidence_rank = -(span.score if span.source == "gliner" and span.score is not None else 0.0)
    return (_SOURCE_ORDER[span.source], confidence_rank, _ENTITY_ORDER[span.entity_type])


def _overlaps(left: DetectedSpan, right: DetectedSpan) -> bool:
    """Return whether two half-open spans in the same cell genuinely overlap."""
    return left.cell_id == right.cell_id and left.start < right.end and right.start < left.end


def _chunks(text: str, config: FreeTextDetectionConfig) -> tuple[_Chunk, ...]:
    """Split a complete cell into overlapping chunks with original offsets.

    A chunk edge that would cut a word moves back to that word's start, so the
    model sees whole words. A word too long for that is cut where it falls.

    Example:
        With ``chunk_length=12`` and ``chunk_overlap=5``, ``"alpha beta gamma delta"``
        becomes ``"alpha beta "``, ``"beta gamma "``, and ``"gamma delta"``.
    """
    words = tuple((match.start(), match.end()) for match in _GLINER_WORD.finditer(text))
    chunks: list[_Chunk] = []
    start = 0
    while start + config.chunk_length < len(text):
        end = _word_start_before(words, start + config.chunk_length, floor=start)
        chunks.append(_Chunk(text[start:end], text, start, words))
        start = _word_start_before(words, max(end - config.chunk_overlap, start + 1), floor=start)
    if text:
        chunks.append(_Chunk(text[start:], text, start, words))
    return tuple(chunks)


def _word_start_before(words: Sequence[tuple[int, int]], index: int, *, floor: int) -> int:
    """Move ``index`` back to the start of a word it cuts, unless that start is at or before ``floor``."""
    word = _word_at(words, index)
    if word is None or word[0] <= floor:
        return index
    return word[0]


def _cuts_word(words: Sequence[tuple[int, int]], index: int) -> bool:
    """Return whether ``index`` falls strictly inside a word."""
    return _word_at(words, index) is not None


def _word_at(words: Sequence[tuple[int, int]], index: int) -> tuple[int, int] | None:
    """Return the word ``index`` falls strictly inside, if any."""
    position = bisect_right(words, (index, math.inf)) - 1
    if position >= 0 and words[position][0] < index < words[position][1]:
        return words[position]
    return None


def _result_spans(result: object) -> Iterable[tuple[int, int, str, float]]:
    """Yield validated primitive span metadata from one GLiNER2 result object."""
    if not _is_string_keyed_mapping(result):
        raise GenerationError("GLiNER2 free-text PII inference returned an invalid result")
    entities = result.get("entities")
    if not _is_string_keyed_mapping(entities):
        raise GenerationError("GLiNER2 free-text PII inference returned an invalid result")
    for label, values in entities.items():
        if not isinstance(values, list):
            raise GenerationError("GLiNER2 free-text PII inference returned invalid entities")
        yield from _entity_result_spans(label, values)


def _is_string_keyed_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    """Return whether ``value`` is a mapping whose keys are all strings."""
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def _entity_result_spans(label: str, values: Sequence[object]) -> Iterable[tuple[int, int, str, float]]:
    """Yield primitive span metadata for one GLiNER entity label."""
    for value in values:
        if not _is_string_keyed_mapping(value):
            raise GenerationError("GLiNER2 free-text PII inference omitted required span metadata")
        start = value.get("start")
        end = value.get("end")
        score = value.get("confidence", 1.0)
        if type(start) is not int or type(end) is not int or not isinstance(score, int | float):
            raise GenerationError("GLiNER2 free-text PII inference returned invalid span metadata")
        yield start, end, label, float(score)


def _normalize_label(label: str) -> str:
    """Normalize a checkpoint label's case and separators for lookup."""
    return label.strip().casefold().replace("-", "_").replace(" ", "_")


def _is_valid_ssn(value: str) -> bool:
    """Return whether ``value`` is a US Social Security number written ``ddd-dd-dddd``.

    python-stdnum rejects numbers never assigned, such as area 000, 666, or 900-999 (ITINs and other tax IDs use the
    9xx range), group 00, or serial 0000, and numbers made public in advertising, such as ``078-05-1120``. Such
    values need the ``national_id`` threshold.
    """
    return _SSN_SHAPE.fullmatch(value) is not None and us_ssn.is_valid(value)


def _load_gliner2_model(model_id: str, revision: str | None) -> _GlinerModel:
    """Load the configured checkpoint revision onto the GPU.

    Raises:
        GenerationError: If PyTorch reports no CUDA GPU. CPU inference is too slow for real datasets, and a
            sandbox that hides the GPU would otherwise turn a fast run into one that appears to hang.
    """
    import torch

    if not torch.cuda.is_available():
        raise GenerationError(
            "GLiNER2 free-text PII detection needs a CUDA GPU, but PyTorch reports none. Run NSS where the GPU is "
            "visible; a sandbox or container without GPU access hides it"
        )
    from gliner2 import GLiNER2

    device = "cuda"
    with heartbeat(
        "GLiNER2 model loading",
        interval=30.0,
        logger_name=__name__,
        model_id=model_id,
        revision=revision,
        device=device,
    ):
        return GLiNER2.from_pretrained(model_id, revision=revision, map_location=device)


def _valid_ipv4(value: str) -> bool:
    """Return whether a candidate is a valid IPv4 address."""
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return True


def _valid_ipv6(value: str) -> bool:
    """Return whether a candidate, with any enclosing URL brackets removed, is a valid IPv6 address."""
    try:
        ipaddress.IPv6Address(value.strip("[]"))
    except ValueError:
        return False
    return True


def _valid_email(value: str) -> bool:
    """Return whether an email candidate satisfies local-part and IDNA domain limits."""
    if len(value) > 254 or value.count("@") != 1:
        return False
    local, domain = value.rsplit("@", 1)
    domain = normalize("NFC", domain)
    if not local or len(local.encode("utf-8")) > 64:
        return False
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return False
    try:
        encoded_domain = domain.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    if len(encoded_domain) > 253 or "." not in encoded_domain:
        return False
    labels = encoded_domain.split(".")
    return all(label and len(label) <= 63 and not label.startswith("-") and not label.endswith("-") for label in labels)


def _valid_card(value: str) -> bool:
    """Return whether a candidate contains a 13–19 digit Luhn-valid card number.

    A 15-digit candidate must also start with an American Express prefix (34 or 37): other Luhn-valid
    15-digit numbers are mostly device IMEIs, which use the same check digit.
    """
    digits = "".join(character for character in value if character in "0123456789")
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    if len(digits) == 15 and not digits.startswith(("34", "37")):
        return False
    return luhn.is_valid(digits)


_REGEX_RULES = (
    _RegexRule(
        "builtin.email",
        EntityType.EMAIL,
        regex.compile(
            r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
            r"[\p{L}\p{N}!#$%&'*+/=?^_`{|}~-]+(?:\.[\p{L}\p{N}!#$%&'*+/=?^_`{|}~-]+)*@"
            r"[\p{L}\p{N}](?:[\p{L}\p{N}\p{M}-]{0,61}[\p{L}\p{N}\p{M}])?"
            r"(?:\.[\p{L}\p{N}](?:[\p{L}\p{N}\p{M}-]{0,61}[\p{L}\p{N}\p{M}])?)+"
            r"(?![A-Za-z0-9_-])"
        ),
        _valid_email,
    ),
    _RegexRule(
        "builtin.credit_debit_card",
        EntityType.CREDIT_DEBIT_CARD,
        regex.compile(r"(?<![0-9])(?:[0-9][ -]?){12,18}[0-9](?![0-9])"),
        _valid_card,
    ),
    _RegexRule(
        "builtin.ipv4",
        EntityType.IPV4,
        regex.compile(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9]|\.[0-9])"),
        _valid_ipv4,
    ),
    # Ported from NeMo Anonymizer's nemo-anonymizer.ipv6.v1 rule. The start may follow a lone colon
    # (``ip:2001:db8::1``) but not a hex digit or ``::``, and the alternation also matches IPv4-mapped
    # addresses (``::ffff:192.0.2.1``) whole so the IPv4 rule cannot claim only their tail.
    _RegexRule(
        "builtin.ipv6",
        EntityType.IPV6,
        regex.compile(
            r"(?:(?<![0-9A-Fa-f:])|(?<=^:)|(?<=[^0-9A-Fa-f:]:))"
            r"(?!:(?!:))(?!:::)(?:"
            r"(?:[0-9A-Fa-f]{0,4}:){2,6}(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
            r"|(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}"
            r")(?![0-9A-Fa-f:]|\.[0-9])"
        ),
        _valid_ipv6,
    ),
)
