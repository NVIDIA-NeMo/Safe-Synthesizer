# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks the executor applies to generated replacement values.

This module lives outside ``replacement.generators`` so the executor can check
candidates without importing generator implementations.
"""

from __future__ import annotations

import ipaddress
import string
from typing import TYPE_CHECKING

from stdnum import luhn

from ...config.replace_pii import EntityType
from ..planning.patterns import parse_name_pattern

if TYPE_CHECKING:
    from .generation import ReplacementGenerationRequest

__all__ = ["generated_value_is_valid", "is_ipv4", "is_ipv6", "is_luhn_valid", "keeps_original_name_part"]


def generated_value_is_valid(request: ReplacementGenerationRequest, value: str) -> bool:
    """Return whether a generated value meets the request's local constraints.

    A replacement must differ from its original ignoring letter case, because a reused replacement may be
    written in the case of each occurrence it replaces: ``Ada`` accepted for ``ADA`` would be written back
    as ``ADA``.
    """
    if not value or value.casefold() == request.original_value.casefold():
        return False
    if request.entity_type is EntityType.CREDIT_DEBIT_CARD:
        return is_luhn_valid(value)
    if request.entity_type is EntityType.IPV4:
        return is_ipv4(value)
    if request.entity_type is EntityType.IPV6:
        return is_ipv6(value)
    return True


def is_ipv4(value: str) -> bool:
    """Return whether ``value`` is an IPv4 address."""
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return True


def is_ipv6(value: str) -> bool:
    """Return whether ``value``, with any enclosing URL brackets removed, is an IPv6 address."""
    try:
        ipaddress.IPv6Address(value.strip("[]"))
    except ValueError:
        return False
    return True


def is_luhn_valid(value: str) -> bool:
    """Return whether the ASCII digits in ``value``, ignoring separators such as spaces and dashes, pass the Luhn check.

    Example:
        ``is_luhn_valid("4111 1111 1111 1111")`` is ``True``.
    """
    digits = "".join(character for character in value if character in string.digits)
    return len(digits) >= 2 and luhn.is_valid(digits)


def keeps_original_name_part(original: str, candidate: str, pattern: str | None, *, locale: str = "en_US") -> bool:
    """Return whether a full-name replacement keeps the original's first, middle, or last name.

    A whole-name comparison accepts ``John Brown`` for ``John Smith``, which keeps the real first name. Both names are
    read by nameparser in the order of ``pattern`` or ``locale``, so titles such as ``Mr.``, suffixes such as
    ``Jr.``, and a ``Last, First`` layout do not hide a kept part, and each part is compared with the same part.
    Words that are fixed text in ``pattern``, such as ``Dr.`` in ``Dr. {First} {Last}``, are skipped. Trailing
    commas and periods are ignored, and single letters are not compared, so a shared initial does not count.

    Example:
        keeps_original_name_part("Mr. John Smith", "John Brown", None)  # True: same first name
        keeps_original_name_part("Dr. Ana Ruiz", "Dr. Maria Lopez", "Dr. {First} {Last}")  # False
    """
    # Imported here so checking other entity types does not load nameparser or Faker.
    from .person_names import read_full_name

    fixed = _pattern_literal_words(pattern)
    original_parts = read_full_name(original, locale=locale, pattern=pattern)
    candidate_parts = read_full_name(candidate, locale=locale, pattern=pattern)
    for part in ("first", "middle", "last"):
        kept = {word for word in _name_words(original_parts[part]) if len(word) > 1 and word not in fixed}
        if kept & set(_name_words(candidate_parts[part])):
            return True
    return False


def _name_words(value: str) -> list[str]:
    return [word for word in (word.strip(",.").casefold() for word in value.split()) if word]


def _pattern_literal_words(pattern: str | None) -> frozenset[str]:
    if pattern is None:
        return frozenset()
    parts, _ = parse_name_pattern(EntityType.FULL_NAME, pattern)
    return frozenset(word for part in parts or () if part.literal is not None for word in _name_words(part.literal))
