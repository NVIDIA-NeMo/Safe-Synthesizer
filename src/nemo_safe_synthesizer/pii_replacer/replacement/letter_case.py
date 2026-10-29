# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Letter-case rules shared by structured and free-text replacement."""

from __future__ import annotations

from .types import CanonicalValue

__all__ = ["case_insensitive_identity", "match_letter_case"]


def case_insensitive_identity(value: CanonicalValue) -> CanonicalValue:
    """Return the mapping identity of ``value``, which ignores letter case for strings.

    Replacement looks up and reuses mappings by this identity, so ``Ada``, ``ADA``, and ``ada`` in one
    record or group share a replacement. Values of other types are returned unchanged.

    Example:
        ``CanonicalValue("string", "Ada")`` and ``CanonicalValue("string", "ADA")`` both map to
        ``CanonicalValue("string", "ada")``.
    """
    if value.type_tag != "string":
        return value
    return CanonicalValue(value.type_tag, value.normalized_value.casefold())


def match_letter_case(occurrence: str, replacement: str) -> str:
    """Give a shared replacement the letter case of the occurrence it replaces.

    Mappings ignore case, so ``MARGARET`` and ``margaret`` share one generated replacement. An
    all-capitals or all-lowercase occurrence gets the replacement in the same case; any other
    occurrence keeps the replacement as generated.

    Example:
        ``match_letter_case("ADA", "Kathleen")`` is ``"KATHLEEN"``; ``match_letter_case("Ada", "Kathleen")``
        is ``"Kathleen"``.
    """
    if occurrence.isupper():
        return replacement.upper()
    if occurrence.islower():
        return replacement.lower()
    return replacement
