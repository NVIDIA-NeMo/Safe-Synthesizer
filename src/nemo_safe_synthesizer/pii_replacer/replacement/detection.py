# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Free-text PII detection and deterministic span resolution."""

from __future__ import annotations

import ipaddress
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import NamedTuple, Protocol, TypeGuard
from unicodedata import normalize

import regex

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
from .types import DetectedSpan, DetectionCell, DetectionCellId

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
# Labels and default thresholds follow the nss-pii-ner-eval v5 study. Request only labels from the
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
    "national_id_number": EntityType.NATIONAL_ID,
    "passport_number": EntityType.NATIONAL_ID,
    "tax_id": EntityType.NATIONAL_ID,
    "tax_number": EntityType.NATIONAL_ID,
    "license_number": EntityType.NATIONAL_ID,
    "drivers_license_number": EntityType.NATIONAL_ID,
    "api_key": EntityType.API_KEY,
    "customer_id": EntityType.UNIQUE_IDENTIFIER,
    "employee_id": EntityType.UNIQUE_IDENTIFIER,
    "medical_record_number": EntityType.UNIQUE_IDENTIFIER,
    "member_id": EntityType.UNIQUE_IDENTIFIER,
    "health_plan_beneficiary_number": EntityType.UNIQUE_IDENTIFIER,
}
_GLINER_UNION_LABELS: dict[str, tuple[EntityType, ...]] = {
    "government_id": (EntityType.SSN, EntityType.NATIONAL_ID),
}
_GLINER_LABEL_ENTITY_TYPES = {
    **{label: (entity_type,) for label, entity_type in _GLINER_LABELS.items()},
    **_GLINER_UNION_LABELS,
}
_SSN_SHAPE = regex.compile(r"(\d{3})-(\d{2})-(\d{4})")
_NAME_ENTITY_TYPES = frozenset(
    {EntityType.FIRST_NAME, EntityType.MIDDLE_NAME, EntityType.LAST_NAME, EntityType.FULL_NAME}
)
# Characters allowed between a middle name and its neighbouring first or last name.
_MIDDLE_NAME_NEIGHBOR_GAP = 3
_ENTITY_ORDER = {entity.entity_type: position for position, entity in enumerate(ENTITIES)}
_SOURCE_ORDER = {"regex": 0, "gliner": 1}


class FreeTextDetector(Protocol):
    """Detect and resolve PII spans for complete original dataframe cells."""

    def detect(self, cells: Sequence[DetectionCell]) -> tuple[DetectedSpan, ...]:
        """Return accepted, non-overlapping spans for ``cells``."""


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
    """Lazy local GLiNER2 adapter with positional chunk remapping."""

    def __init__(
        self,
        config: FreeTextDetectionConfig,
        *,
        model_loader: Callable[[str], _GlinerModel] | None = None,
    ) -> None:
        colliding = sorted(set(_exclusion_thresholds(config)) & set(_GLINER_LABEL_ENTITY_TYPES))
        if colliding:
            raise ParameterError(
                f"free_text_detection.name_exclusion_thresholds cannot use GLiNER2 detection labels: {colliding}"
            )
        self._config = config
        self._model_loader = model_loader or _load_gliner2_model
        self._model: _GlinerModel | None = None

    def detect(self, cells: Sequence[DetectionCell]) -> tuple[DetectedSpan, ...]:
        """Run GLiNER2 once per unique text and copy valid spans to matching cells.

        Within each unique text, name spans that overlap a name-exclusion span
        are dropped, then the middle-name neighbour rule is applied, both before
        per-cell entity filtering so every cell sees the same model decision.
        """
        cells_by_text = _cells_by_text(cells)
        if not cells_by_text:
            return ()

        chunks = list(chain.from_iterable(_chunks(text, self._config) for text in cells_by_text))
        results = self._infer(chunks, _requested_labels(cells, self._config))
        candidates: dict[str, _TextCandidates] = defaultdict(_TextCandidates)
        rejected_birth_dates = 0
        for chunk, result in zip(chunks, results, strict=True):
            rejected_birth_dates += _collect_chunk_candidates(
                chunk, result, candidates[chunk.original_text], self._config
            )
        spans: list[DetectedSpan] = []
        for text, text_cells in cells_by_text.items():
            for span in _filter_name_spans(candidates[text], self._config):
                spans.extend(_copy_span_to_cells(span, text_cells))
        if rejected_birth_dates:
            logger.user.warning(
                "GLiNER2 birth-date candidates were ignored because they were not complete parseable dates",
                extra={"rejected_candidate_count": rejected_birth_dates},
            )
        return tuple(spans)

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
                self._model = self._model_loader(self._config.model_id)
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
    """Bounded deterministic detector for structurally validated PII."""

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

    def detect(self, cells: Sequence[DetectionCell]) -> tuple[DetectedSpan, ...]:
        """Return exact spans from applicable built-in rules."""
        return tuple(chain.from_iterable(self._detect_cell(cell) for cell in cells))

    def _detect_cell(self, cell: DetectionCell) -> tuple[DetectedSpan, ...]:
        """Run every applicable bounded regex rule against one complete cell."""
        applicable_rules = (rule for rule in _REGEX_RULES if rule.entity_type in cell.allowed_entity_types)
        return tuple(chain.from_iterable(self._detect_rule(cell, rule) for rule in applicable_rules))

    def _detect_rule(self, cell: DetectionCell, rule: _RegexRule) -> tuple[DetectedSpan, ...]:
        """Apply one regex rule with its timeout, match cap, and structural validator."""
        try:
            matches = rule.expression.finditer(cell.text, timeout=self._timeout_seconds)
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
        except TimeoutError as exc:
            raise GenerationError(f"built-in PII regex rule {rule.rule_id!r} timed out") from exc


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

    def detect(self, cells: Sequence[DetectionCell]) -> tuple[DetectedSpan, ...]:
        """Return globally resolved spans from both mandatory detector sources."""
        candidates = (*self._gliner.detect(cells), *self._regex.detect(cells))
        by_cell: dict[DetectionCellId, list[DetectedSpan]] = defaultdict(list)
        for span in candidates:
            by_cell[span.cell_id].append(span)
        accepted: list[DetectedSpan] = []
        for cell in cells:
            accepted.extend(resolve_overlapping_spans(by_cell.get(cell.cell_id, ())))
        return tuple(accepted)


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
        for label, entity_types in sorted(_GLINER_LABEL_ENTITY_TYPES.items())
        if allowed.intersection(entity_types)
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
    entity_type = _normalize_gliner_label(label, value)
    if entity_type is None:
        return None, False
    if score < config.entity_thresholds[entity_type]:
        return None, False

    # An identifier without a digit is almost always a word the model mistook for one.
    if entity_type is EntityType.UNIQUE_IDENTIFIER and not any(character.isdigit() for character in value):
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
    candidates = _GLINER_LABEL_ENTITY_TYPES[_normalize_label(label)]
    # Union labels are normalized from the returned value, so inference must
    # preserve candidates accepted by any possible NSS entity type.
    return min(config.entity_thresholds[entity_type] for entity_type in candidates)


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

    Spans are ranked by length, then confidence (regex spans count as 1.0),
    then position, then source, GLiNER2 score, and entity order, and each is
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
    """Return a span's confidence, treating structurally validated regex matches as certain."""
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
    """Split a complete cell into overlapping chunks with original offsets."""
    step = config.chunk_length - config.chunk_overlap
    chunks: list[_Chunk] = []
    offset = 0
    while offset < len(text):
        chunks.append(_Chunk(text[offset : offset + config.chunk_length], text, offset))
        if offset + config.chunk_length >= len(text):
            break
        offset += step
    return tuple(chunks)


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


def _is_string_keyed_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
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


def _normalize_gliner_label(label: str, value: str) -> EntityType | None:
    """Normalize one documented checkpoint label, including validated union labels."""
    normalized = _normalize_label(label)
    entity_type = _GLINER_LABELS.get(normalized)
    if normalized == "government_id":
        return EntityType.SSN if _is_valid_ssn(value) else EntityType.NATIONAL_ID
    return entity_type


def _normalize_label(label: str) -> str:
    """Normalize a checkpoint label's case and separators for lookup."""
    return label.strip().casefold().replace("-", "_").replace(" ", "_")


def _is_valid_ssn(value: str) -> bool:
    """Return whether ``value`` is ``ddd-dd-dddd`` with an SSN-assignable area, group, and serial.

    Area 000, 666, and 900-999 (ITINs and other tax IDs use the 9xx range), group 00, and serial 0000
    are never assigned, so such values are typed ``national_id`` instead.
    """
    match = _SSN_SHAPE.fullmatch(value)
    if match is None:
        return False
    area, group, serial = match.groups()
    return area not in {"000", "666"} and not area.startswith("9") and group != "00" and serial != "0000"


def _load_gliner2_model(model_id: str) -> _GlinerModel:
    """Load the configured checkpoint directly onto the best available device."""
    import torch
    from gliner2 import GLiNER2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with heartbeat(
        "GLiNER2 model loading",
        interval=30.0,
        logger_name=__name__,
        model_id=model_id,
        device=device,
    ):
        return GLiNER2.from_pretrained(model_id, map_location=device)


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
    """Return whether a candidate contains a 13–19 digit Luhn-valid card number."""
    digits = "".join(character for character in value if character in "0123456789")
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for position, character in enumerate(digits):
        digit = int(character)
        if position % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


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
