# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision rules for structured replacements.

Every replacement follows the same rules:

1. It differs from its own original, ignoring letter case. This rule is strict.
2. In an identifier-like column whose original values are all distinct, such as a key column,
   replacements stay distinct too, ignoring case. When retries cannot find an unused value, an earlier
   replacement is repeated and counted so the caller can warn. A new replacement there also avoids the
   values of columns that share replacements with it, so a value shared into it later is not a repeat.
   Other columns may repeat replacements, and a replacement may equal another row's original.
3. A full-name replacement avoids keeping the original's first or last name; when retries cannot, it
   keeps one and is counted.

A grouping column is stricter: its replacements must be one-to-one, so two groups never merge, and no
repeat fallback is allowed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from ...config.replace_pii import EntityType
from .value_checks import generated_value_is_valid, keeps_original_name_part

if TYPE_CHECKING:
    from .generation import ReplacementGenerationRequest

__all__ = ["IDENTIFIER_ENTITY_TYPES", "CandidateChooser", "ReplacementOutcome", "all_distinct"]

IDENTIFIER_ENTITY_TYPES = frozenset(
    {
        EntityType.EMAIL,
        EntityType.PHONE_NUMBER,
        EntityType.SSN,
        EntityType.NATIONAL_ID,
        EntityType.CREDIT_DEBIT_CARD,
        EntityType.API_KEY,
        EntityType.IPV4,
        EntityType.IPV6,
        EntityType.UNIQUE_IDENTIFIER,
    }
)
"""Entity types whose replacements stay distinct when the column's original values are all distinct."""

ReplacementOutcome = Literal["new", "repeat", "kept_name_part"]
"""How a replacement was chosen: a new value, or one of the fallbacks in preference order."""


def all_distinct(values: Iterable[str]) -> bool:
    """Return whether there is at least one value and no two are equal, ignoring letter case.

    Example:
        ``["A-1", "a-2"]`` is all distinct; ``["A-1", "a-1"]`` and ``[]`` are not.
    """
    seen: set[str] = set()
    for value in values:
        folded = value.casefold()
        if folded in seen:
            return False
        seen.add(folded)
    return bool(seen)


class CandidateChooser:
    """Choose one replacement from successive generated candidates under the collision rules.

    Offer candidates until one is accepted as new or the caller runs out of attempts, then take the
    best fallback. The chooser does not record the chosen value; the caller does.

    Args:
        entity_type: Entity type of the value being replaced.
        original: Original text the replacement must differ from.
        pattern: Plan pattern of the target, used to skip fixed words when comparing full names.
        used: Case-folded values a column that must stay distinct should not get: its own replacements
            and those of the columns that share replacements with it. ``None`` when replacements may
            repeat. A candidate in this set is kept only as a repeat fallback.
        one_to_one: The same values for a grouping column. When given, a candidate in this set is
            rejected outright and no repeat fallback is allowed, so two groups never share one.
    """

    def __init__(
        self,
        entity_type: EntityType,
        original: str,
        pattern: str | None = None,
        *,
        used: set[str] | None = None,
        one_to_one: set[str] | None = None,
    ) -> None:
        self._entity_type = entity_type
        self._original = original
        self._pattern = pattern
        self._used = used
        self._one_to_one = one_to_one
        self._repeat: str | None = None
        self._kept_name_part: str | None = None

    def offer(self, request: ReplacementGenerationRequest, candidate: str) -> bool:
        """Return whether ``candidate`` can be used as a new replacement; otherwise keep it as a fallback."""
        if not generated_value_is_valid(request, candidate):
            return False
        if self._one_to_one is not None and candidate.casefold() in self._one_to_one:
            return False
        # A full name that differs as a whole can still keep the real first or last name. Prefer one that
        # keeps neither; fall back to it only when no other name was found, so a plan whose dependencies
        # fix a name part cannot fail the run.
        if self._entity_type is EntityType.FULL_NAME and keeps_original_name_part(
            self._original, candidate, self._pattern
        ):
            self._kept_name_part = self._kept_name_part or candidate
            return False
        if self._used is not None and candidate.casefold() in self._used:
            self._repeat = self._repeat or candidate
            return False
        return True

    def fallback(self) -> tuple[str, ReplacementOutcome] | None:
        """Return the best rejected candidate and how it falls short, or ``None`` if none is usable.

        Repeating an earlier replacement is preferred over keeping an original name part. A one-to-one
        target only accepts a kept name part.
        """
        if self._one_to_one is None and self._repeat is not None:
            return self._repeat, "repeat"
        if self._kept_name_part is not None:
            return self._kept_name_part, "kept_name_part"
        return None
