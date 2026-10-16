# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parsing and rendering for birth-date replacements.

This module lives outside ``replacement.generators`` so planning can check a
``date_of_birth`` column with the same parser replacement uses, without loading
generator implementations.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from random import Random

__all__ = ["birth_date_is_supported", "random_birth_date", "shift_birth_date"]

_FALLBACK_BIRTH_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y")
_RANDOM_BIRTH_DATE_START = date(1930, 1, 1)
_RANDOM_BIRTH_DATE_DAYS = (date(2010, 12, 31) - _RANDOM_BIRTH_DATE_START).days


def shift_birth_date(original: str, pattern: str | None, rng: Random) -> str | None:
    """Shift a parseable birth date by 1 to 365 days in either direction, or return ``None``.

    The plan's pattern only has to cover 99% of a column, so each cell is tried
    against the pattern first and then against common date formats. The
    replacement uses the pattern when one is configured, else the format that
    parsed. ``None`` means the cell could not be parsed; the caller then
    generates a random birth date so the original never survives.

    Example:
        With pattern ``"%m/%d/%Y"``, ``"02/14/1990"`` shifts to a date such as
        ``"11/03/1989"``, ``"1990-02-14"`` also shifts and is written as
        ``%m/%d/%Y``, and ``"08/2010"`` returns ``None``.
    """
    parsed = _parse_birth_date(original, pattern)
    if parsed is None:
        return None
    value, output_pattern = parsed
    offset = rng.randint(-365, 365)
    if offset == 0:
        offset = 1
    shifted = value + timedelta(days=offset)
    return shifted.strftime(output_pattern) if output_pattern is not None else shifted.date().isoformat()


def random_birth_date(pattern: str | None, rng: Random) -> str:
    """Return a random birth date between 1930 and 2010, written with ``pattern`` or as ISO 8601."""
    value = _RANDOM_BIRTH_DATE_START + timedelta(days=rng.randrange(_RANDOM_BIRTH_DATE_DAYS + 1))
    return value.strftime(pattern) if pattern is not None else value.isoformat()


def birth_date_is_supported(value: str) -> bool:
    """Return whether ``value`` parses as a complete date without a configured pattern.

    Planning uses this to check that an unpatterned ``date_of_birth`` column really holds dates.

    Example:
        ``"1990-02-14"`` and ``"02/14/1990"`` are supported; an age such as ``"65"`` is not.
    """
    return _parse_birth_date(value, None) is not None


def _parse_birth_date(original: str, pattern: str | None) -> tuple[datetime, str | None] | None:
    """Return the parsed date and the format to write its replacement in, or ``None`` if no format fits."""
    if pattern is not None:
        try:
            return datetime.strptime(original, pattern), pattern
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(original), pattern
    except ValueError:
        pass
    for candidate in _FALLBACK_BIRTH_DATE_FORMATS:
        try:
            return datetime.strptime(original, candidate), pattern or candidate
        except ValueError:
            continue
    return None
