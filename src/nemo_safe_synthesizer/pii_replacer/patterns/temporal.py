# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal formats: match a cell's strftime format, and rank the formats a column writes."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

import pandas as pd

from .evidence import DOMINANT_PATTERN_MIN_COVERAGE, dominant_format, pattern_evidence_values

_DATETIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
]
_DATE_FORMATS = [
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%m/%d/%y",
    "%d-%m-%Y",
    "%m/%Y",
    "%Y/%m",
    "%m-%Y",
    "%Y-%m",
]
_TIME_FORMATS = [
    "%H:%M:%S",
    "%H:%M",
    "%I:%M:%S %p",
    "%I:%M %p",
]
_ISO_DURATION_RE = re.compile(
    r"^P(?=\d|T)(?:\d+Y)?(?:\d+M)?(?:\d+D)?(?:T(?:\d+H)?(?:\d+M)?(?:\d+(?:\.\d+)?S)?)?$",
    re.IGNORECASE,
)
_HUMAN_DURATION_RE = re.compile(
    r"^\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$",
    re.IGNORECASE,
)
_COMPACT_DURATION_RE = re.compile(r"^\d+[hms](?:\d+[hms])?$", re.IGNORECASE)


def try_strftime_formats(value: str, formats: list[str]) -> str | None:
    for fmt in formats:
        try:
            datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
        return fmt
    return None


def match_datetime_format(value: object) -> str | None:
    """Return the strftime format for a datetime cell, or None.

    Args:
        value: Cell value to inspect.

    Returns:
        A strftime format string when the value parses as datetime, else None.

    Example:
        ``"2020-03-15 14:30:00"`` -> ``"%Y-%m-%d %H:%M:%S"``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or not any(ch.isdigit() for ch in s):
        return None
    if "T" not in s and " " not in s:
        return None
    return try_strftime_formats(s, _DATETIME_FORMATS)


def match_date_format(value: object) -> str | None:
    """Return the strftime format a date cell parses as, or None.

    Cheap pre-filters (must contain a digit and a ``/`` or ``-`` separator) keep
    this from attempting ``strptime`` on obvious non-dates such as plain numbers,
    emails, or free text.

    Args:
        value: Cell value to inspect.

    Returns:
        A strftime format string (e.g. ``%m/%d/%Y``) when the value parses as a
        date, else None.

    Example:
        ``"03/15/2020"`` -> ``"%m/%d/%Y"``; ``"12345"`` -> None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or not any(ch.isdigit() for ch in s) or ("/" not in s and "-" not in s):
        return None
    if match_datetime_format(s):
        return None
    return try_strftime_formats(s, _DATE_FORMATS)


def match_time_format(value: object) -> str | None:
    """Return the strftime format for a time-only cell, or None.

    Args:
        value: Cell value to inspect.

    Returns:
        A strftime format string when the value parses as time-only, else None.

    Example:
        ``"2:30 PM"`` -> ``"%I:%M %p"``; ``"192.168.1.1"`` -> None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or ":" not in s or not any(ch.isdigit() for ch in s):
        return None
    if "/" in s or "-" in s:
        return None
    if "::" in s or re.search(r"[A-Fa-f]", s):
        return None
    return try_strftime_formats(s, _TIME_FORMATS)


def match_duration_format(value: object) -> str | None:
    """Return a duration pattern label (``iso8601`` or ``human``), or None.

    Args:
        value: Cell value to inspect.

    Returns:
        ``"iso8601"`` or ``"human"`` when the value matches a known duration
        shape, else None.

    Example:
        ``"P1DT2H"`` -> ``"iso8601"``; ``"45 minutes"`` / ``"2h30m"`` -> ``"human"``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    if _ISO_DURATION_RE.match(s):
        return "iso8601"
    if _HUMAN_DURATION_RE.match(s) or _COMPACT_DURATION_RE.match(s):
        return "human"
    return None


def date_patterns(values: pd.Series, *, min_coverage: float | None = None) -> list[str]:
    """Return zero or one strftime format for plan emission.

    Only formats returned by ``match_datetime_format`` / ``match_date_format``
    are counted. Unparseable values are skipped (they count against coverage
    via the evidence-sample denominator). The plan names the top format only
    when it covers ≥ ``min_coverage`` of the evidence sample.

    Args:
        values: Column date values.
        min_coverage: Dominant coverage threshold (default
            ``DOMINANT_PATTERN_MIN_COVERAGE``).

    Returns:
        ``[strftime]`` when a dominant format exists, else ``[]``.

    Example:
        A column mostly ``03/15/2020`` -> ``["%m/%d/%Y"]``; mixed formats under
        85% coverage -> ``[]``.
    """
    sample = pattern_evidence_values(values)
    counts: Counter = Counter()
    for value in sample:
        fmt = match_datetime_format(value) or match_date_format(value)
        if fmt is not None:
            counts[fmt] += 1
    if not counts:
        return []
    coverage = DOMINANT_PATTERN_MIN_COVERAGE if min_coverage is None else min_coverage
    top = dominant_format(counts, len(sample), min_coverage=coverage)
    return [top] if top else []


def date_pattern(values: pd.Series, *, min_coverage: float | None = None) -> str | None:
    """Dominant strftime format, or ``None`` when coverage is below threshold."""
    patterns = date_patterns(values, min_coverage=min_coverage)
    return patterns[0] if patterns else None
