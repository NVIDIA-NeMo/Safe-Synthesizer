# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared structured PII replacement generation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from random import Random
from typing import TYPE_CHECKING

from ....config.replace_pii import EntityType
from ....errors import GenerationError
from ...planning.patterns import parse_character_mask, parse_name_pattern, render_name_pattern
from ..types import EffectiveDependencyTuple
from ..value_checks import is_luhn_valid

if TYPE_CHECKING:
    from ..generation import ReplacementGenerationRequest

__all__ = [
    "NAME_ENTITY_TYPES",
    "dependency_values",
    "email_domain",
    "looks_like_uuid",
    "name_values",
    "normalize_organization_domain",
    "random_birth_date",
    "render_address",
    "render_luhn_pattern",
    "render_name",
    "required_pattern_values",
    "resolve_dependency_values",
    "shape_preserving_value",
    "shift_birth_date",
]

NAME_ENTITY_TYPES = frozenset(
    {EntityType.FIRST_NAME, EntityType.MIDDLE_NAME, EntityType.LAST_NAME, EntityType.FULL_NAME}
)
"""Entity types generated from person-name parts."""


def dependency_values(dependencies: EffectiveDependencyTuple) -> dict[EntityType, str]:
    return {entity_type: value.normalized_value for entity_type, value in dependencies if value is not None}


def resolve_dependency_values(
    request: ReplacementGenerationRequest,
    entity_type: EntityType,
    value: str,
) -> tuple[str, ...]:
    """Return the sampler values one dependency value may match."""
    resolved = dict(request.resolved_dependency_values)
    return resolved.get(entity_type, (value.casefold(),))


def name_values(
    dependencies: Mapping[EntityType, str],
    sampled: Mapping[str, str],
) -> dict[str, str]:
    """Resolve name parts from dependencies first, then one sampler result."""
    full_name = dependencies.get(EntityType.FULL_NAME, "")
    full_parts = full_name.split()
    inferred = {
        "first": full_parts[0] if full_parts else "",
        "middle": " ".join(full_parts[1:-1]) if len(full_parts) > 2 else "",
        "last": full_parts[-1] if len(full_parts) > 1 else "",
    }
    return {
        "first": dependencies.get(EntityType.FIRST_NAME) or inferred["first"] or sampled.get("first", ""),
        "middle": dependencies.get(EntityType.MIDDLE_NAME) or inferred["middle"] or sampled.get("middle", ""),
        "last": dependencies.get(EntityType.LAST_NAME) or inferred["last"] or sampled.get("last", ""),
    }


def render_name(
    entity_type: EntityType,
    pattern: str | None,
    rng: Random,
    values: Mapping[str, str],
) -> str:
    if pattern is not None:
        return render_name_pattern(entity_type, pattern, values, rng)
    return {
        EntityType.FIRST_NAME: values["first"],
        EntityType.MIDDLE_NAME: values["middle"],
        EntityType.LAST_NAME: values["last"],
        EntityType.FULL_NAME: f"{values['first']} {values['last']}",
    }[entity_type]


def required_pattern_values(entity_type: EntityType, pattern: str) -> frozenset[str]:
    parts, error = parse_name_pattern(entity_type, pattern)
    if error is not None or parts is None:
        raise GenerationError(error or "invalid name pattern")
    return frozenset(part.placeholder.part for part in parts if part.placeholder is not None)


def email_domain(value: str) -> str | None:
    if "@" not in value:
        return None
    domain = value.rsplit("@", 1)[1]
    return domain if domain and not any(character.isspace() for character in domain) else None


def render_address(street: str, dependencies: Mapping[EntityType, str]) -> str:
    locality = [
        dependencies.get(EntityType.CITY),
        dependencies.get(EntityType.STATE),
        dependencies.get(EntityType.ZIPCODE),
        dependencies.get(EntityType.COUNTRY),
    ]
    suffix = ", ".join(value for value in locality if value)
    return f"{street}, {suffix}" if suffix else street


def normalize_organization_domain(value: str) -> str:
    """Normalize an organization value into one lowercase DNS label."""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]+", "-", ascii_value.casefold()).strip("-")
    return normalized[:63].rstrip("-")


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


_MAX_LUHN_ATTEMPTS = 100


def render_luhn_pattern(pattern: str, rng: Random) -> str:
    parts, error = parse_character_mask(pattern)
    if error is not None or parts is None:
        raise GenerationError(error or "invalid credit/debit-card pattern")
    rendered: list[str] = []
    variable_digits: list[tuple[int, str]] = []
    for part in parts:
        if part.literal is not None:
            rendered.append(part.literal)
            continue
        digit_choices = "".join(dict.fromkeys(character for character in part.choices or "" if character.isdigit()))
        if not digit_choices:
            raise GenerationError("credit/debit-card patterns must generate digits in every variable position")
        rendered.append(rng.choice(digit_choices))
        variable_digits.append((len(rendered) - 1, digit_choices))
    if not variable_digits:
        raise GenerationError("credit/debit-card pattern has no generated digit")
    for _ in range(_MAX_LUHN_ATTEMPTS):
        # Exactly one digit makes the number Luhn-valid at any one position, so repair
        # the last position whose allowed digits include it.
        for position, digit_choices in reversed(variable_digits):
            current = rendered[position]
            for digit in digit_choices:
                rendered[position] = digit
                candidate = "".join(rendered)
                if is_luhn_valid(candidate):
                    return candidate
            rendered[position] = current
        for position, digit_choices in variable_digits:
            rendered[position] = rng.choice(digit_choices)
    raise GenerationError("credit/debit-card pattern cannot produce a Luhn-valid value")


def shape_preserving_value(original: str, rng: Random) -> str:
    rendered: list[str] = []
    for character in original:
        # Every letter and digit is replaced, including non-ASCII ones, so no identifying text survives.
        # Letters without case, such as CJK characters, become uppercase ASCII letters.
        if character.isdigit():
            rendered.append(str(rng.randrange(10)))
        elif character.islower():
            rendered.append(chr(ord("a") + rng.randrange(26)))
        elif character.isalpha():
            rendered.append(chr(ord("A") + rng.randrange(26)))
        else:
            rendered.append(character)
    return "".join(rendered)


def looks_like_uuid(value: str) -> bool:
    return (
        re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            value,
        )
        is not None
    )
