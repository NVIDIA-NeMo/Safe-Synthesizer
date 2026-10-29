# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared structured PII replacement generation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from random import Random
from typing import TYPE_CHECKING

from anyascii import anyascii

from ....config.replace_pii import EntityType
from ....errors import GenerationError
from ...planning.patterns import parse_character_mask, parse_name_pattern, render_name_pattern
from ..types import EffectiveDependencyTuple
from ..value_checks import is_luhn_valid
from .person_names import read_full_name, write_full_name

if TYPE_CHECKING:
    from ..generation import ReplacementGenerationRequest

__all__ = [
    "NAME_ENTITY_TYPES",
    "default_email_pattern",
    "dependency_values",
    "email_domain",
    "name_values",
    "normalize_organization_domain",
    "render_luhn_pattern",
    "render_name",
    "required_pattern_values",
    "resolve_dependency_values",
    "shape_preserving_value",
]

NAME_ENTITY_TYPES = frozenset(
    {EntityType.FIRST_NAME, EntityType.MIDDLE_NAME, EntityType.LAST_NAME, EntityType.FULL_NAME}
)
"""Entity types generated from person-name parts."""


def dependency_values(dependencies: EffectiveDependencyTuple) -> dict[EntityType, str]:
    """Return each present dependency's normalized value by entity type, skipping missing ones."""
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
    *,
    locale: str,
    full_name_pattern: str | None = None,
) -> dict[str, str]:
    """Resolve name parts from dependencies first, then one sampler result.

    A full-name dependency is read by :func:`read_full_name` in the order of ``full_name_pattern``, the pattern its
    column was written with, or else in ``locale``'s order.
    """
    full_name = dependencies.get(EntityType.FULL_NAME, "")
    inferred = read_full_name(full_name, locale=locale, pattern=full_name_pattern) if full_name else {}
    return {
        "first": dependencies.get(EntityType.FIRST_NAME) or inferred.get("first") or sampled.get("first", ""),
        "middle": dependencies.get(EntityType.MIDDLE_NAME) or inferred.get("middle") or sampled.get("middle", ""),
        "last": dependencies.get(EntityType.LAST_NAME) or inferred.get("last") or sampled.get("last", ""),
    }


def default_email_pattern(dependencies: Mapping[EntityType, str], *, locale: str) -> str | None:
    """Return the email pattern for an email that depends on names but has no plan pattern.

    The local part joins whichever of the first and last names the dependencies resolve, then adds two random digits so
    that people sharing a name rarely share an address: replaced first and last names, or a full name, follow
    ``{first}.{last}##@{domain}``, and a first name alone follows ``{first}##@{domain}``. A plan omits the pattern when
    the column mixes formats, such as ``brenda.rivera@`` and ``hughes_b@`` in one column; the replacement still uses
    the replaced names.

    Returns:
        The pattern, or ``None`` when no name dependency resolves and the generator samples a whole email.
    """
    values = name_values(dependencies, {}, locale=locale)
    local_parts = [f"{{{part}}}" for part in ("first", "last") if values[part]]
    return f"{'.'.join(local_parts)}##@{{domain}}" if local_parts else None


def render_name(
    entity_type: EntityType,
    pattern: str | None,
    rng: Random,
    values: Mapping[str, str],
    *,
    locale: str,
) -> str:
    """Render a name from ``first``, ``middle``, and ``last`` parts, following ``pattern`` when one is set.

    Without a pattern, a full name is written in ``locale``'s order (see :func:`write_full_name`), and a name part is
    that part alone.

    Example:
        ``render_name(EntityType.FULL_NAME, None, rng, {"first": "Ada", "middle": "", "last": "Lovelace"},
        locale="en_US")`` is ``"Ada Lovelace"``.
    """
    if pattern is not None:
        return render_name_pattern(entity_type, pattern, values, rng)
    return {
        EntityType.FIRST_NAME: values["first"],
        EntityType.MIDDLE_NAME: values["middle"],
        EntityType.LAST_NAME: values["last"],
        EntityType.FULL_NAME: write_full_name(values["first"], values["last"], locale),
    }[entity_type]


def required_pattern_values(entity_type: EntityType, pattern: str) -> frozenset[str]:
    """Return the name parts a name or email ``pattern`` uses, such as ``{"first", "last"}``.

    Raises:
        GenerationError: If ``pattern`` is not a valid pattern for ``entity_type``.
    """
    parts, error = parse_name_pattern(entity_type, pattern)
    if error is not None or parts is None:
        raise GenerationError(error or "invalid name pattern")
    return frozenset(part.placeholder.part for part in parts if part.placeholder is not None)


def email_domain(value: str) -> str | None:
    """Return the text after the last ``@`` in ``value``, or ``None`` if there is none or it contains spaces."""
    if "@" not in value:
        return None
    domain = value.rsplit("@", 1)[1]
    return domain if domain and not any(character.isspace() for character in domain) else None


def normalize_organization_domain(value: str) -> str:
    """Normalize an organization value into one lowercase DNS label, transliterating it to ASCII with anyascii.

    Example:
        ``"Straße AG"`` becomes ``"strasse-ag"``, ``"Ørsted"`` becomes ``"orsted"``, and ``"北京大学"`` becomes
        ``"beijingdaxue"``.
    """
    ascii_value = anyascii(value)
    normalized = re.sub(r"[^a-z0-9]+", "-", ascii_value.casefold()).strip("-")
    return normalized[:63].rstrip("-")


_MAX_LUHN_ATTEMPTS = 100


def render_luhn_pattern(pattern: str, rng: Random) -> str:
    """Render a card number from a character mask so that it passes the Luhn checksum.

    Literal characters are kept, every generated position gets a random allowed digit, and one
    position is then adjusted so the checksum holds.

    Raises:
        GenerationError: If the mask is invalid, a generated position allows no digit, or no
            Luhn-valid value fits the mask.
    """
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
    """Return ``original`` with every digit and letter replaced at random, keeping its layout.

    Digits stay digits, lowercase letters stay lowercase, other letters become uppercase ASCII,
    and every other character is kept.

    Example:
        ``"D123-456x"`` becomes a value such as ``"Q804-913k"``.
    """
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
