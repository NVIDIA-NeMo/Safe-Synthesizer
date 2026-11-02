# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parsing and rendering for structured and free-text birth dates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from random import Random

import dateparser

__all__ = ["birth_date_is_supported", "random_birth_date", "shift_birth_date"]

_YEAR_FIRST_PATTERNS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d")
_MONTH_FIRST_PATTERNS = ("%m/%d/%Y", "%m-%d-%Y")
_DAY_FIRST_PATTERNS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")
# Only en_US writes numeric dates month first, so other locales try day-first formats before
# month-first ones: "05/04/1990" is 5 April in fr_FR and 4 May in en_US.
_MONTH_FIRST_LOCALE = "en_US"
# Only fixed calendar dates are accepted. dateparser's other parsers read values such as "yesterday",
# "in 2 days", or an age like "30 years" relative to the current time, which would make the output
# depend on the run day and turn ages into dates. Timestamps are excluded for the same reason.
_NATURAL_DATE_SETTINGS: dict[str, object] = {
    "PARSERS": ["custom-formats", "absolute-time"],
    "STRICT_PARSING": True,
    "REQUIRE_PARTS": ["day", "month", "year"],
    "RETURN_AS_TIMEZONE_AWARE": False,
}
_RANDOM_BIRTH_DATE_START = date(1930, 1, 1)
_RANDOM_BIRTH_DATE_DAYS = (date(2010, 12, 31) - _RANDOM_BIRTH_DATE_START).days
_SHIFT_STANDARD_DEVIATION_DAYS = 30
_MAX_SHIFT_DAYS = 180


@dataclass(frozen=True, slots=True)
class _ParsedBirthDate:
    """A complete birth date and the format that can be safely preserved."""

    value: datetime
    """Parsed calendar date."""

    output_pattern: str | None
    """``strftime`` pattern to render the shifted date in, or ``None`` to emit ISO format."""


def birth_date_is_supported(value: str) -> bool:
    """Return whether ``value`` contains one complete, deterministic date."""
    return _parse_birth_date(value, None, _MONTH_FIRST_LOCALE) is not None


def shift_birth_date(original: str, pattern: str | None, rng: Random, *, locale: str) -> str | None:
    """Shift a parseable birth date by a small nonzero number of days, or return ``None``.

    The shift is drawn from a normal distribution with a 30-day standard deviation and never
    exceeds 180 days, so the original date does not survive while ages and the column's
    distribution stay close to the original.

    The plan's pattern only has to cover 99% of a column, so each value is tried
    against the pattern first, then ISO 8601, common numeric formats, and strict
    complete natural-language dates. Numeric formats are tried month first for
    ``en_US`` and day first for every other ``locale``. Relative expressions such as ``"yesterday"``
    or an age such as ``"30 years"`` are not fixed dates and are not parsed. The replacement uses the pattern when one
    is configured, else the numeric format that parsed, else ISO 8601. ``None``
    means the value could not be parsed; the caller then generates a random
    birth date so the original never survives.

    Example:
        With pattern ``"%m/%d/%Y"``, ``"02/14/1990"`` shifts to a date such as
        ``"11/03/1989"``, ``"1990-02-14"`` also shifts and is written as
        ``%m/%d/%Y``, and ``"08/2010"`` returns ``None``.
    """
    parsed = _parse_birth_date(original, pattern, locale)
    if parsed is None:
        return None
    shifted = parsed.value + timedelta(days=_shift_days(rng))
    if parsed.output_pattern is not None:
        return shifted.strftime(parsed.output_pattern)
    return shifted.date().isoformat()


def _shift_days(rng: Random) -> int:
    """Draw a nonzero shift in days from a normal distribution, at most ``_MAX_SHIFT_DAYS`` either way.

    A draw that rounds to zero or falls beyond the cap is redrawn rather than moved, so no day
    collects the extra probability and the distribution stays symmetric.
    """
    while True:
        days = round(rng.gauss(0, _SHIFT_STANDARD_DEVIATION_DAYS))
        if days != 0 and abs(days) <= _MAX_SHIFT_DAYS:
            return days


def random_birth_date(pattern: str | None, rng: Random) -> str:
    """Return a random birth date between 1930 and 2010, written with ``pattern`` or as ISO 8601.

    The date is taken at midnight UTC, so a pattern with ``%z`` or ``%Z`` still renders a timezone and the
    value parses with that pattern.

    Example:
        With pattern ``"%Y-%m-%d %z"``, the result looks like ``"1974-06-02 +0000"``.
    """
    value = _RANDOM_BIRTH_DATE_START + timedelta(days=rng.randrange(_RANDOM_BIRTH_DATE_DAYS + 1))
    if pattern is None:
        return value.isoformat()
    return datetime.combine(value, time(), tzinfo=UTC).strftime(pattern)


def _parse_birth_date(original: str, pattern: str | None, locale: str) -> _ParsedBirthDate | None:
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

    for candidate in _numeric_patterns(locale):
        try:
            return _ParsedBirthDate(datetime.strptime(original, candidate), pattern or candidate)
        except ValueError:
            continue

    # Natural-language dates always name a month in words. dateparser costs about 100 ms per value,
    # so skipping it for values without letters keeps numeric columns, such as an age mislabeled
    # as a birth date, from taking minutes.
    if not any(character.isalpha() for character in original):
        return None
    parsed_natural = dateparser.parse(original, settings=_NATURAL_DATE_SETTINGS)
    if parsed_natural is not None:
        return _ParsedBirthDate(parsed_natural, pattern)
    return None


def _numeric_patterns(locale: str) -> tuple[str, ...]:
    """Return the numeric date formats to try, in order, for ``locale``."""
    if locale == _MONTH_FIRST_LOCALE:
        return (*_YEAR_FIRST_PATTERNS, *_MONTH_FIRST_PATTERNS, *_DAY_FIRST_PATTERNS)
    return (*_YEAR_FIRST_PATTERNS, *_DAY_FIRST_PATTERNS, *_MONTH_FIRST_PATTERNS)
