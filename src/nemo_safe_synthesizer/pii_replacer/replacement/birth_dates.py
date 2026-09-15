# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parsing and rendering for structured and free-text birth dates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random

import dateparser

from ...errors import GenerationError

__all__ = ["birth_date_is_supported", "shift_birth_date"]

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


@dataclass(frozen=True, slots=True)
class _ParsedBirthDate:
    """A complete birth date and the format that can be safely preserved."""

    value: datetime
    output_pattern: str | None


def birth_date_is_supported(value: str) -> bool:
    """Return whether ``value`` contains one complete, deterministic date."""
    try:
        _parse_birth_date(value, None)
    except GenerationError:
        return False
    return True


def shift_birth_date(original: str, pattern: str | None, rng: Random) -> str:
    """Shift a complete birth date by a deterministic nonzero offset of at most one year."""
    parsed = _parse_birth_date(original, pattern)
    offset = rng.randint(-365, 365)
    if offset == 0:
        offset = 1
    shifted = parsed.value + timedelta(days=offset)
    if parsed.output_pattern is not None:
        return shifted.strftime(parsed.output_pattern)
    return shifted.date().isoformat()


def _parse_birth_date(original: str, pattern: str | None) -> _ParsedBirthDate:
    """Parse a configured pattern or a strict complete natural-language date."""
    if pattern is not None:
        return _parse_configured_birth_date(original, pattern)

    try:
        parsed_iso = datetime.fromisoformat(original)
    except ValueError:
        parsed_iso = None
    if parsed_iso is not None:
        return _ParsedBirthDate(parsed_iso, None)

    for candidate in _PRESERVED_PATTERNS:
        try:
            return _ParsedBirthDate(datetime.strptime(original, candidate), candidate)
        except ValueError:
            continue

    parsed_natural = dateparser.parse(original, settings=_NATURAL_DATE_SETTINGS)
    if parsed_natural is not None:
        return _ParsedBirthDate(parsed_natural, None)
    raise GenerationError("date_of_birth value is not a complete parseable date")


def _parse_configured_birth_date(original: str, pattern: str) -> _ParsedBirthDate:
    """Parse a structured birth date with its validated strftime pattern."""
    try:
        return _ParsedBirthDate(datetime.strptime(original, pattern), pattern)
    except ValueError as exc:
        raise GenerationError("date_of_birth value does not match its validated pattern") from exc
