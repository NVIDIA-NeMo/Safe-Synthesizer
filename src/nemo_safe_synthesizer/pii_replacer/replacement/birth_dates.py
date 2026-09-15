# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parsing and rendering for structured and free-text birth dates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from random import Random

import dateparser

__all__ = ["birth_date_is_supported", "random_birth_date", "shift_birth_date"]

_PRESERVED_PATTERNS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%d-%m-%Y",
)
_NATURAL_DATE_SETTINGS: dict[str, object] = {
    "STRICT_PARSING": True,
    "REQUIRE_PARTS": ["day", "month", "year"],
    "RETURN_AS_TIMEZONE_AWARE": False,
}
_RANDOM_BIRTH_DATE_START = date(1930, 1, 1)
_RANDOM_BIRTH_DATE_DAYS = (date(2010, 12, 31) - _RANDOM_BIRTH_DATE_START).days


@dataclass(frozen=True, slots=True)
class _ParsedBirthDate:
    """A complete birth date and the format that can be safely preserved."""

    value: datetime
    """Parsed calendar date."""

    output_pattern: str | None
    """``strftime`` pattern to render the shifted date in, or ``None`` to emit ISO format."""


def birth_date_is_supported(value: str) -> bool:
    """Return whether ``value`` contains one complete, deterministic date."""
    return _parse_birth_date(value, None) is not None


def shift_birth_date(original: str, pattern: str | None, rng: Random) -> str | None:
    """Shift a parseable birth date by 1 to 365 days in either direction, or return ``None``.

    The plan's pattern only has to cover 99% of a column, so each value is tried
    against the pattern first, then ISO 8601, common numeric formats, and strict
    complete natural-language dates. The replacement uses the pattern when one
    is configured, else the numeric format that parsed, else ISO 8601. ``None``
    means the value could not be parsed; the caller then generates a random
    birth date so the original never survives.

    Example:
        With pattern ``"%m/%d/%Y"``, ``"02/14/1990"`` shifts to a date such as
        ``"11/03/1989"``, ``"1990-02-14"`` also shifts and is written as
        ``%m/%d/%Y``, and ``"08/2010"`` returns ``None``.
    """
    parsed = _parse_birth_date(original, pattern)
    if parsed is None:
        return None
    offset = rng.randint(-365, 365)
    if offset == 0:
        offset = 1
    shifted = parsed.value + timedelta(days=offset)
    if parsed.output_pattern is not None:
        return shifted.strftime(parsed.output_pattern)
    return shifted.date().isoformat()


def random_birth_date(pattern: str | None, rng: Random) -> str:
    """Return a random birth date between 1930 and 2010, written with ``pattern`` or as ISO 8601."""
    value = _RANDOM_BIRTH_DATE_START + timedelta(days=rng.randrange(_RANDOM_BIRTH_DATE_DAYS + 1))
    return value.strftime(pattern) if pattern is not None else value.isoformat()


def _parse_birth_date(original: str, pattern: str | None) -> _ParsedBirthDate | None:
    """Return the parsed date and the format to write its replacement in, or ``None`` if no format fits."""
    if pattern is not None:
        try:
            return _ParsedBirthDate(datetime.strptime(original, pattern), pattern)
        except ValueError:
            pass

    try:
        return _ParsedBirthDate(datetime.fromisoformat(original), pattern)
    except ValueError:
        pass

    for candidate in _PRESERVED_PATTERNS:
        try:
            return _ParsedBirthDate(datetime.strptime(original, candidate), pattern or candidate)
        except ValueError:
            continue

    parsed_natural = dateparser.parse(original, settings=_NATURAL_DATE_SETTINGS)
    if parsed_natural is not None:
        return _ParsedBirthDate(parsed_natural, pattern)
    return None
