# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value-level entity recognition: anchored regexes, content probes, and column coverage."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from ..entities import ORG_KEYWORDS, Config, is_missing_value
from ..patterns import (
    PATTERN_SAMPLE_SIZE,
    match_date_format,
    match_datetime_format,
    match_duration_format,
    match_time_format,
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


# Ordered strftime formats for temporal value detection. First match wins.
# Datetime/time are checked before date-only formats. All date entries carry a
# separator so plain integers never match (compact YYYYMMDD is handled separately).
def match_value_entity(value: object, *, phone_min_digits: int = 10) -> str | None:
    """Return the best entity label for one cell value via anchored regexes.

    Order matters (specific to general). Used for column classification and
    single-pattern structured columns.

    Example:
        ``"jane@acme.com"`` -> ``"email"``
        ``"415-555-0100"`` -> ``"phone_number"``
        ``"12345"`` -> ``None``

    Args:
        value: Cell value to classify.
        phone_min_digits: Minimum digit count for phone matches; use ``7`` when the
            column header is already phone-like so short national numbers can match.

    Returns:
        Entity label string, or ``None`` when no pattern matches.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    if _EMAIL_RE.match(s):
        return "email"
    if _IPV4_RE.match(s):
        return "ipv4"
    if _IPV6_RE.match(s) and s.count(":") >= 2:
        # Guard against HH:MM:SS time strings (2 colons, all-decimal, no '::'):
        # require real ipv6 structure -- '::' compression, a hex letter, or 3+ colons.
        if "::" in s or s.count(":") >= 3 or re.search(r"[A-Fa-f]", s):
            return "ipv6"
    if UUID_RE.match(s):
        return "unique_identifier"
    if _HEX_OPAQUE_RE.match(s) or _JWT_OPAQUE_RE.match(s):
        return "unique_identifier"
    if _BASE64_OPAQUE_RE.match(s) and any(c.isdigit() for c in s) and any(c.isalpha() for c in s):
        # Prefer opaque unique_identifier over api_key for long base64-like blobs
        # without a known API prefix.
        if not any(s.startswith(p) for p in API_PREFIXES):
            return "unique_identifier"
    # Long composite job/record ids (prefix + timestamp + uuid + epoch, etc.).
    if (
        len(s) >= 24
        and re.fullmatch(r"[A-Za-z0-9_\-.:]+", s)
        and any(c.isdigit() for c in s)
        and any(c.isalpha() for c in s)
        and any(c in "-_.:/" for c in s)
        and not any(s.startswith(p) for p in API_PREFIXES)
    ):
        return "unique_identifier"
    if _SSN_RE.match(s):
        return "ssn"
    if match_datetime_format(s):
        return "datetime"
    if match_date_format(s):
        return "date"
    if match_time_format(s):
        return "time"
    if match_duration_format(s):
        return "duration"
    if any(s.startswith(p) for p in API_PREFIXES):
        return "api_key"
    phone_s = _strip_phone_extension(s)
    d = _digits(phone_s)
    if _CARD_RE.match(s) and 13 <= len(_digits(s)) <= 19:
        if _luhn_ok(_digits(s)):
            return "credit_debit_card"
        # Failed Luhn: only fall through to phone when punctuation looks phone-like.
        if not _looks_like_phone_punctuation(phone_s):
            return None
    if _PHONE_RE.match(phone_s) and phone_min_digits <= len(d) <= 15 and _looks_like_phone_punctuation(phone_s):
        return "phone_number"
    if _API_RE.match(s) and any(c.isdigit() for c in s) and any(c.isalpha() for c in s):
        return "api_key"
    return None


def match_value_pattern(value: object, *, phone_min_digits: int = 10) -> tuple[str | None, str | None]:
    """Return ``(entity, concrete_pattern)`` for one cell value.

    Example:
        ``"03/15/2020"`` -> ``("date", "%m/%d/%Y")``
        ``"jane@acme.com"`` -> ``("email", None)``

    Args:
        value: Cell value to classify.
        phone_min_digits: Minimum digit count forwarded to ``match_value_entity``.

    Returns:
        Tuple of entity label and concrete strftime or template pattern; both
        ``None`` when the value does not match any entity.
    """
    entity = match_value_entity(value, phone_min_digits=phone_min_digits)
    match entity:
        case None:
            return None, None
        case "date":
            fmt = match_date_format(value)
            return ("date", fmt) if fmt else (None, None)
        case "datetime":
            fmt = match_datetime_format(value)
            return ("datetime", fmt) if fmt else (None, None)
        case "time":
            fmt = match_time_format(value)
            return ("time", fmt) if fmt else (None, None)
        case "duration":
            fmt = match_duration_format(value)
            return ("duration", fmt) if fmt else (None, None)
        case _:
            return entity, None


def analyze_column_patterns(
    series: pd.Series,
    cfg: Config,
    sample: int = PATTERN_SAMPLE_SIZE,
    *,
    phone_min_digits: int = 10,
) -> dict:
    """Compute dominant entity, pattern, and coverage for a column.

    Coverage is aggregated by entity so mixed formats of the same entity still
    qualify as structured.

    Example:
        A column that is 90% emails ->
        ``{entity: "email", coverage: 90.0, structured: True}``.

    Args:
        series: Column values to analyze.
        cfg: Engine configuration with ``dominant_pattern_min_coverage`` threshold.
        sample: Maximum non-null values to sample for analysis.
        phone_min_digits: Minimum digit count forwarded to ``match_value_pattern``.

    Returns:
        Dict with ``entity``, ``pattern``, ``coverage``, and ``structured`` keys.
    """
    non_null = series.dropna()
    if non_null.empty:
        return {"entity": None, "pattern": None, "coverage": 0.0, "structured": False}

    if len(non_null) > sample:
        non_null = non_null.sample(sample, random_state=0)
    total = len(non_null)
    counts: Counter = Counter()
    for v in non_null:
        entity, pattern = match_value_pattern(v, phone_min_digits=phone_min_digits)
        if entity is None:
            counts[("__unmatched__", None)] += 1
        else:
            counts[(entity, pattern)] += 1

    typed = {k: c for k, c in counts.items() if k[0] != "__unmatched__"}
    if not typed:
        return {"entity": None, "pattern": None, "coverage": 0.0, "structured": False}

    # Coverage is by entity across all of its patterns (mixed formats of the same
    # entity still count as structured), while ``pattern`` remains the single
    # most common concrete format for template attachment.
    entity_totals: Counter = Counter()
    for (entity, _pattern), count in typed.items():
        entity_totals[entity] += count
    entity, entity_count = max(entity_totals.items(), key=lambda kv: kv[1])
    coverage = round(entity_count / total * 100, 1)
    structured = coverage >= cfg.dominant_pattern_min_coverage
    pattern_counts = {pat: c for (ent, pat), c in typed.items() if ent == entity}
    pattern = max(pattern_counts, key=lambda pat: pattern_counts[pat]) if pattern_counts else None
    return {
        "entity": entity,
        "pattern": pattern,
        "coverage": coverage,
        "structured": structured,
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
    if _JWT_OPAQUE_RE.match(s):
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
