# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value-level entity recognition: anchored regexes, content probes, and column coverage."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from ..entities import ORG_KEYWORDS, Config, is_missing_value
from ..patterns import (
    PATTERN_SAMPLE_SIZE,
    match_date_format,
    match_datetime_format,
    match_duration_format,
    match_time_format,
    pattern_evidence_values,
    value_matches_template,
    value_patterns,
)
from .column_names import name_supports_value_entity

# Labels whose match carries a concrete strftime/template format, which
# ``entity_coverage`` requires before it counts the match.
_TEMPORAL_LABELS: tuple[str, ...] = ("datetime", "date", "time", "duration")

# Every label ``collect_value_entities`` can emit, most specific first. The order
# only breaks ties between candidates with identical coverage.
_VALUE_ENTITY_LABELS: tuple[str, ...] = (
    "email",
    "ipv4",
    "ipv6",
    "unique_identifier",
    "ssn",
    *_TEMPORAL_LABELS,
    "api_key",
    "credit_debit_card",
    "phone_number",
)


# ===========================================================================
# Value-entity detection (regex; whole-column vs mixed)
# ===========================================================================
def _luhn_ok(digits: str) -> bool:
    if not digits.isdigit():
        return False
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Full-match (anchored) regexes used to classify a single cell's value.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\Z")  # ASCII only; IDN not matched
_SSN_RE = re.compile(r"\d{3}-\d{2}-\d{4}\Z")
_IPV4_RE = re.compile(r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\Z")
_IPV6_RE = re.compile(r"(?=.*:)(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}\Z")
UUID_RE = re.compile(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z")
# Allow optional leading ``+`` / ``(`` so ``(415) 555-0100`` and ``+1-415…`` both match.
_PHONE_RE = re.compile(r"\+?\(?\d[\d\-\.\s()]{5,}\d\Z")
_PHONE_EXT_RE = re.compile(r"(?i)[\s,;]*\b(?:x|ext\.?|extension)\s*\d+\s*$")
_CARD_RE = re.compile(r"(?:\d[ -]?){12,18}\d\Z")
_HEX_OPAQUE_RE = re.compile(r"(?i)(?:[0-9a-f]{32}|[0-9a-f]{64})\Z")
_BASE64_OPAQUE_RE = re.compile(r"^[A-Za-z0-9_\-]{22,}={0,2}\Z")
_JWT_OPAQUE_RE = re.compile(r"^[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\Z")
API_PREFIXES = (
    "sk-",
    "pk-",
    "rk-",
    "ak-",
    "ghp_",
    "gho_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
    "AKIA",
    "ASIA",
    "AIza",
    "Bearer ",
)
_API_RE = re.compile(r"[A-Za-z0-9_\-]{20,}\Z")
_COMPACT_YMD_RE = re.compile(r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\Z")
_MULTI_PERSON_DELIM_RE = re.compile(r"(?i)\s+(?:and|&)\s+|\s*/\s*|\s*;\s*")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _strip_phone_extension(s: str) -> str:
    return _PHONE_EXT_RE.sub("", s).strip()


def _looks_like_phone_punctuation(s: str) -> bool:
    """True when the string uses phone-like separators (not a bare digit run / PAN)."""
    return bool(re.search(r"[\s\-\.\(\)+]", s))


# Issuer identification numbers paired with the digit lengths each brand issues.
# Luhn alone passes roughly 1 in 10 arbitrary numbers, so the prefix/length
# pairing is what keeps phone-shaped 13-15 digit values out of the card bucket.
_CARD_BRAND_RULES: tuple[tuple[str, re.Pattern[str], frozenset[int]], ...] = (
    ("visa", re.compile(r"4"), frozenset({13, 16, 19})),
    ("amex", re.compile(r"3[47]"), frozenset({15})),
    # Diners Club issues 14-digit cards and, on the Discover network, 16-19 digit ones.
    ("diners", re.compile(r"36|3[89]|30[0-5]|3095"), frozenset({14, 16, 17, 18, 19})),
    ("mastercard", re.compile(r"5[1-5]|2(?:2[2-9]|[3-6]\d|7[01]|720)"), frozenset({16})),
    ("discover", re.compile(r"6011|65|64[4-9]"), frozenset({16, 19})),
    ("jcb", re.compile(r"35"), frozenset({16, 17, 18, 19})),
    ("unionpay", re.compile(r"62"), frozenset({16, 17, 18, 19})),
)

# Entities whose shape a phone number can never legitimately take. A value that
# is a valid dotted quad, a strict 3-2-4 SSN, or a parseable temporal is not a
# phone even when its digits and separators satisfy ``_PHONE_RE``.
_PHONE_EXCLUSIVE_LABELS = frozenset({"ipv4", "ipv6", "ssn", "date", "datetime", "time", "duration"})


def card_brand(digits: str) -> str | None:
    """Return the card brand for a bare digit string, or ``None`` when none match.

    Example:
        ``"378282246310005"`` -> ``"amex"``
        ``"7111111111111111"`` -> ``None``

    Args:
        digits: Digits-only card number (separators already stripped).

    Returns:
        Brand name when the issuer prefix and digit length agree, else ``None``.
    """
    length = len(digits)
    for brand, prefix_re, lengths in _CARD_BRAND_RULES:
        if length in lengths and prefix_re.match(digits):
            return brand
    return None


def _looks_like_jwt_opaque(s: str) -> bool:
    """True for JWT-like ``a.b.c`` tokens, not digit-only dotted phones.

    ``818.470.1711`` matches the three-segment JWT shape but is a phone number.
    Require at least one alphabetic character and reject all-digit segments.
    """
    if not _JWT_OPAQUE_RE.match(s):
        return False
    parts = s.split(".")
    if len(parts) != 3:
        return False
    if all(p.isdigit() for p in parts):
        return False
    return any(c.isalpha() for c in s)


def _normalize_value_string(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s or None


def collect_value_entities(value: object, *, phone_min_digits: int = 10) -> list[str]:
    """Return every entity label whose shape ``value`` satisfies.

    Each recognizer is evaluated independently, so nothing here decides what a
    column *is*: this reports all shapes a value could be, and
    ``analyze_column_patterns`` scores the entities its header allows. One pass
    per value answers that for every entity at once, which is why the return type
    is a list rather than a per-entity predicate.

    Two shapes are mutually exclusive rather than reported side by side, because
    no real value is both:

    - A card-shaped value that is not a valid PAN and lacks phone-like
      punctuation is treated as non-PII and stops further phone/api matching.
    - ``phone_number`` is suppressed when the value already matched a shape a
      phone cannot take (``_PHONE_EXCLUSIVE_LABELS``), such as a dotted quad,
      a strict 3-2-4 SSN, or a parseable date. Answering "is this a phone?"
      therefore requires evaluating those other recognizers regardless.

    Example:
        ``"3782-822463-10005"`` -> ``["credit_debit_card", "phone_number"]``
        ``"818.470.1711"`` -> ``["phone_number"]``
        ``"255.255.255.0"`` -> ``["ipv4"]``

    Args:
        value: Cell value to classify.
        phone_min_digits: Minimum digit count for phone matches; use ``7`` when the
            column header is already phone-like so short national numbers can match.

    Returns:
        Deduplicated entity labels, most specific first. Empty when nothing
        matches. The order carries no decision weight; see ``_VALUE_ENTITY_LABELS``
        for the order used to break coverage ties.
    """
    s = _normalize_value_string(value)
    if s is None:
        return []

    matches: list[str] = []

    if _EMAIL_RE.match(s):
        matches.append("email")
    if _IPV4_RE.match(s):
        matches.append("ipv4")
    if _IPV6_RE.match(s) and s.count(":") >= 2:
        # Guard against HH:MM:SS time strings (2 colons, all-decimal, no '::'):
        # require real ipv6 structure -- '::' compression, a hex letter, or 3+ colons.
        if "::" in s or s.count(":") >= 3 or re.search(r"[A-Fa-f]", s):
            matches.append("ipv6")

    opaque = False
    if UUID_RE.match(s) or _HEX_OPAQUE_RE.match(s) or _looks_like_jwt_opaque(s):
        opaque = True
    elif (
        _BASE64_OPAQUE_RE.match(s)
        and any(c.isdigit() for c in s)
        and any(c.isalpha() for c in s)
        and not any(s.startswith(p) for p in API_PREFIXES)
    ):
        # Long base64-like blobs without a known API prefix. These also satisfy
        # the api_key shape below; both are reported and the header decides.
        opaque = True
    elif (
        len(s) >= 24
        and re.fullmatch(r"[A-Za-z0-9_\-.:]+", s)
        and any(c.isdigit() for c in s)
        and any(c.isalpha() for c in s)
        and any(c in "-_.:/" for c in s)
        and not any(s.startswith(p) for p in API_PREFIXES)
    ):
        # Long composite job/record ids (prefix + timestamp + uuid + epoch, etc.).
        opaque = True
    if opaque:
        matches.append("unique_identifier")

    if _SSN_RE.match(s):
        matches.append("ssn")
    # Every temporal format table entry carries a separator, so plain integers
    # never match here (compact YYYYMMDD is handled separately).
    matches.extend(label for label in _TEMPORAL_LABELS if temporal_format(s, label))
    if any(s.startswith(p) for p in API_PREFIXES):
        matches.append("api_key")

    phone_s = _strip_phone_extension(s)
    d = _digits(phone_s)
    card_digits = _digits(s)
    if _CARD_RE.match(s) and 13 <= len(card_digits) <= 19:
        if _luhn_ok(card_digits) and card_brand(card_digits) is not None:
            matches.append("credit_debit_card")
        elif not _looks_like_phone_punctuation(phone_s):
            # Bare digit run that is not a valid PAN: not card, not phone/api.
            return matches

    if (
        _PHONE_EXCLUSIVE_LABELS.isdisjoint(matches)
        and _PHONE_RE.match(phone_s)
        and phone_min_digits <= len(d) <= 15
        and _looks_like_phone_punctuation(phone_s)
    ):
        matches.append("phone_number")
    if "api_key" not in matches and _API_RE.match(s) and any(c.isdigit() for c in s) and any(c.isalpha() for c in s):
        matches.append("api_key")

    return matches


def temporal_format(value: object, label: str) -> str | None:
    """Return the concrete strftime/template format for a temporal ``label``."""
    match label:
        case "datetime":
            return match_datetime_format(value)
        case "date":
            return match_date_format(value)
        case "time":
            return match_time_format(value)
        case "duration":
            return match_duration_format(value)
        case _:
            return None


@dataclass(frozen=True)
class EntityCoverage:
    """Per-entity match counts for one column.

    Every entity is counted independently, so a value matching two entities
    counts toward both and the coverages do not sum to 100. That is the point:
    it removes any ordering in which one entity's regex shadows another's.
    """

    total: int
    """Number of sampled non-null values."""
    counts: Mapping[str, int]
    """Entity label to number of sampled values matching it."""
    patterns: Mapping[str, str | None]
    """Entity label to its most common concrete temporal format, when applicable."""

    def coverage(self, label: str) -> float:
        """Percent of sampled values matching ``label``."""
        if not self.total:
            return 0.0
        return round(self.counts.get(label, 0) / self.total * 100, 1)

    def pattern(self, label: str) -> str | None:
        """Dominant concrete format for ``label``, or ``None`` when not temporal."""
        return self.patterns.get(label)


def entity_coverage(
    series: pd.Series,
    sample: int = PATTERN_SAMPLE_SIZE,
    *,
    phone_min_digits: int = 10,
) -> EntityCoverage:
    """Score every entity independently against a column's values.

    Example:
        A column of ``818.470.1711``-style values ->
        ``coverage("phone_number") == 100.0``, regardless of which other
        recognizers those values also satisfy.

    Args:
        series: Column values to analyze.
        sample: Maximum non-null values to sample.
        phone_min_digits: Minimum digit count for phone matches.

    Returns:
        ``EntityCoverage`` for the sampled values.
    """
    non_null = series.dropna()
    if non_null.empty:
        return EntityCoverage(total=0, counts={}, patterns={})
    if len(non_null) > sample:
        non_null = non_null.sample(sample, random_state=0)

    counts: Counter = Counter()
    formats: dict[str, Counter] = {}
    for value in non_null:
        for label in collect_value_entities(value, phone_min_digits=phone_min_digits):
            if label in _TEMPORAL_LABELS:
                # Temporals only count when a concrete format is recoverable,
                # since the plan attaches that format to the column.
                fmt = temporal_format(value, label)
                if fmt is None:
                    continue
                formats.setdefault(label, Counter())[fmt] += 1
            counts[label] += 1

    patterns = {label: max(by_format, key=lambda f: by_format[f]) for label, by_format in formats.items()}
    return EntityCoverage(total=len(non_null), counts=dict(counts), patterns=patterns)


def candidate_entities(name_label: str | None) -> list[str]:
    """Entities a column may be assigned, named entity first so it wins ties.

    Delegates the policy to ``name_supports_value_entity``: the header's own
    entity, plus identify-not-replaced temporals, which are the only entities
    allowed to be inferred from values alone.

    Example:
        ``"phone_number"`` -> ``["phone_number", "datetime", "date", "time", "duration"]``
        ``None`` -> ``["datetime", "date", "time", "duration"]``

    Args:
        name_label: Entity label inferred from the column header, or ``None``.

    Returns:
        Candidate entity labels in tie-break order.
    """
    supported = [label for label in _VALUE_ENTITY_LABELS if name_supports_value_entity(name_label, label)]
    if name_label in supported:
        supported.remove(name_label)
        return [name_label, *supported]
    return supported


def analyze_column_patterns(
    series: pd.Series,
    cfg: Config,
    sample: int = PATTERN_SAMPLE_SIZE,
    *,
    phone_min_digits: int = 10,
    name_label: str | None = None,
) -> dict:
    """Verify a column's content against the entities its header allows.

    Each candidate entity is scored independently (see ``entity_coverage``), then
    the best-covered candidate is returned. Because the structured threshold is
    above 50%, at most one entity can clear it, so scoring candidates separately
    cannot disagree with picking a single winner -- but it does prevent one
    entity's regex from shadowing another's on individual values.

    Candidates come from ``candidate_entities``: the header's entity, plus
    identify-not-replaced temporals. Replaceable entities are never inferred from
    values alone.

    Example:
        A ``phone_number`` column of ``818.470.1711`` values ->
        ``{entity: "phone_number", coverage: 100.0, structured: True}``, even
        though those values also satisfy an opaque-token shape.

    Args:
        series: Column values to analyze.
        cfg: Engine configuration with ``dominant_pattern_min_coverage`` threshold.
        sample: Maximum non-null values to sample for analysis.
        phone_min_digits: Minimum digit count forwarded to value matching.
        name_label: Entity label inferred from the column header, or ``None``.

    Returns:
        Dict with ``entity``, ``pattern``, ``coverage``, and ``structured`` keys.
    """
    table = entity_coverage(series, sample, phone_min_digits=phone_min_digits)

    best_label: str | None = None
    best_coverage = 0.0
    for label in candidate_entities(name_label):
        label_coverage = table.coverage(label)
        if label_coverage > best_coverage:
            best_label, best_coverage = label, label_coverage

    if best_label is None:
        return {"entity": None, "pattern": None, "coverage": 0.0, "structured": False}
    return {
        "entity": best_label,
        "pattern": table.pattern(best_label),
        "coverage": best_coverage,
        "structured": best_coverage >= cfg.dominant_pattern_min_coverage,
    }


def looks_like_person_name(value: object) -> bool:
    """Return whether a value looks like a person name rather than an organization.

    Example:
        ``"Jane Smith"`` -> ``True``
        ``"St. Mary's Hospital"`` -> ``False``

    Args:
        value: Cell value to evaluate.

    Returns:
        ``True`` when the value passes person-name heuristics.
    """
    if is_missing_value(value):
        return False
    s = str(value).strip()
    if _MULTI_PERSON_DELIM_RE.search(s):
        return False
    tokens = [t for t in re.split(r"\s+", s) if t]
    if not (1 <= len(tokens) <= 5):
        return False
    # Word-boundary / token match so "Mary Health" can still pass when the only
    # other token is a personal given name, while "Regional Health Partners" fails.
    lowered_tokens = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in tokens]
    org_hits = [t for t in lowered_tokens if t in ORG_KEYWORDS]
    if org_hits:
        other = [t for t in lowered_tokens if t not in ORG_KEYWORDS]
        if len(other) != 1 or len(tokens) > 2:
            return False
    return True


def sample_looks_like_multi_person(series: pd.Series, sample: int = 40) -> bool:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    if len(non_null) > sample:
        non_null = non_null.sample(sample, random_state=0)
    hits = sum(1 for v in non_null if _MULTI_PERSON_DELIM_RE.search(str(v)))
    return hits / len(non_null) >= 0.5


def sample_looks_like_org_name(series: pd.Series, sample: int = 40) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    if len(non_null) > sample:
        non_null = non_null.sample(sample, random_state=0)
    personish = sum(1 for v in non_null if looks_like_person_name(v))
    return personish / len(non_null) < 0.5


def looks_like_street_address(value: object) -> bool:
    """Return whether a value looks like a full street line with a house number.

    Example:
        ``"123 Main St"`` -> ``True``
        ``"Main Street"`` -> ``False``

    Args:
        value: Cell value to evaluate.

    Returns:
        ``True`` when the value contains both digits and alphabetic street text.
    """
    if is_missing_value(value):
        return False
    s = str(value).strip()
    if len(s) < 5:
        return False
    # Require a house / unit number and alphabetic street text.
    if not re.search(r"\d", s) or not re.search(r"[A-Za-z]{2,}", s):
        return False
    return True


def looks_like_api_key_value(value: object) -> bool:
    """Return whether a cell looks like an API credential rather than a plain number.

    Args:
        value: Cell value to evaluate.

    Returns:
        ``True`` when the value matches known API-key prefixes or credential shapes.
    """
    if is_missing_value(value):
        return False
    s = str(value).strip()
    if not s:
        return False
    # Pure numeric strings are never api keys (avoids token/count columns).
    if re.fullmatch(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", s):
        return False
    if any(s.startswith(p) for p in API_PREFIXES):
        return True
    if _looks_like_jwt_opaque(s):
        return True
    if _API_RE.match(s) and any(c.isdigit() for c in s) and any(c.isalpha() for c in s):
        return True
    return False


def _sample_majority(series: pd.Series, predicate, sample: int = 40, threshold: float = 0.5) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    if len(non_null) > sample:
        non_null = non_null.sample(sample, random_state=0)
    hits = sum(1 for v in non_null if predicate(v))
    return hits / len(non_null) >= threshold


def sample_looks_like_street_address(series: pd.Series, sample: int = 40) -> bool:
    return _sample_majority(series, looks_like_street_address, sample=sample)


def sample_looks_like_api_key(series: pd.Series, sample: int = 40) -> bool:
    return _sample_majority(series, looks_like_api_key_value, sample=sample)


# Weak ``*id`` headers (``valid``, ``userid``, …) must look like opaque codes, not
# category labels. Require both variety and a dominant character template.
_WEAK_UNIQUE_ID_MIN_UNIQUE_RATIO = 0.5


def sample_has_dominant_identifier_template(series: pd.Series, cfg: Config) -> bool:
    """Return whether values share a dominant identifier-like character template.

    Used for weak ``unique_identifier`` name matches. Requires:
    - unique-value ratio at least ``_WEAK_UNIQUE_ID_MIN_UNIQUE_RATIO`` (rejects
      low-cardinality labels such as ``type_0`` / ``type_1``), and
    - a inferred value template covering at least
      ``cfg.dominant_pattern_min_coverage`` percent of the evidence sample.

    Uniqueness and template evidence both use ``pattern_evidence_values`` so
    blanks and configured textual missings do not dilute the ratio when the
    nonempty cells are distinct opaque IDs (e.g. mostly-blank ``userid``).

    Args:
        series: Column values to evaluate.
        cfg: Engine configuration with ``dominant_pattern_min_coverage``.

    Returns:
        ``True`` when the column looks like templated opaque identifiers.
    """
    sample = pattern_evidence_values(series)
    if not sample:
        return False
    if len(set(sample)) / len(sample) < _WEAK_UNIQUE_ID_MIN_UNIQUE_RATIO:
        return False
    patterns = value_patterns(pd.Series(sample), cfg)
    if not patterns:
        return False
    matched = sum(1 for value in sample if value_matches_template(value, patterns[0]))
    return matched / len(sample) * 100 >= cfg.dominant_pattern_min_coverage


def looks_like_sequential_integer_id(series: pd.Series) -> bool:
    """Return whether a numeric column is a contiguous integer sequence.

    Only applies to numeric dtypes so zero-padded string ids (``00000001``) still
    get templates. Dense contiguous integers such as ``1, 2, 3, …`` or
    ``100000, 100001, …`` are skipped from ``unique_identifier`` replacement.

    Args:
        series: Column values to evaluate.

    Returns:
        ``True`` when values form a dense or near-contiguous integer sequence.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if len(non_null) < 3:
        return False
    vals: list[int] = []
    for v in non_null:
        if isinstance(v, bool):
            return False
        try:
            if isinstance(v, float) and not float(v).is_integer():
                return False
            iv = int(v)
        except (TypeError, ValueError):
            return False
        if iv < 0:
            return False
        vals.append(iv)
    uniq = sorted(set(vals))
    if len(uniq) < 3:
        return False
    # Dense contiguous range at any origin (surrogate keys, autoincrement, etc.).
    if uniq[-1] - uniq[0] + 1 == len(uniq):
        return True
    arr = pd.Series(vals, dtype="int64")
    diffs = arr.diff().dropna()
    if len(diffs) and float((diffs == 1).mean()) >= 0.95:
        return True
    return False


def probe_numeric_column(series: pd.Series, name_label: str | None) -> str | None:
    """Return compact YYYYMMDD DOB or long digit-id labels for numeric columns.

    Digit-id hits return the supportive header label (``ssn``, ``national_id``,
    or ``unique_identifier``), not a collapsed ``unique_identifier``.

    Example:
        Header ``date_of_birth``, values ``19850315``, … -> ``"date_of_birth"``
        Header ``ssn``, values ``123456789``, … -> ``"ssn"``

    Args:
        series: Numeric column values to probe.
        name_label: Entity label inferred from the column header, or ``None``.

    Returns:
        Entity label when the numeric probe succeeds, or ``None`` otherwise.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return None
    sample = series.dropna()
    if sample.empty:
        return None
    if len(sample) > 200:
        sample = sample.sample(200, random_state=0)
    digit_strs = []
    for v in sample:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return None
        if iv < 0:
            return None
        digit_strs.append(str(iv))
    if not digit_strs:
        return None
    if (
        name_label == "date_of_birth"
        and sum(1 for s in digit_strs if _COMPACT_YMD_RE.match(s)) / len(digit_strs) >= 0.85
    ):
        return "date_of_birth"
    # Numeric ID probes still need a supportive header (never values alone).
    # Preserve the header's entity (ssn / national_id / unique_identifier) so
    # generators and gates match; do not collapse everything to unique_identifier.
    if name_label in {"unique_identifier", "national_id", "ssn"}:
        if sum(1 for s in digit_strs if len(s) >= 6) / len(digit_strs) >= 0.85:
            return name_label
    return None
