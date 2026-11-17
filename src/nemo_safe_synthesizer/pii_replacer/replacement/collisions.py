# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision rules for structured replacements.

Every replacement follows the same rules:

1. It differs from its own original, ignoring letter case and surrounding whitespace. This rule is strict.
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

from collections.abc import Container, Iterable
from typing import TYPE_CHECKING, Literal

from ...config.replace_pii import EntityType
from .value_checks import generated_value_is_valid, keeps_original_name_part

if TYPE_CHECKING:
    from .generation import ReplacementGenerationRequest

__all__ = [
    "IDENTIFIER_ENTITY_TYPES",
    "CandidateChooser",
    "ReplacementOutcome",
    "ReplacementOwners",
    "all_distinct",
]

IDENTIFIER_ENTITY_TYPES = frozenset(
    {
        EntityType.EMAIL,
        EntityType.PHONE_NUMBER,
        EntityType.NATIONAL_ID,
        EntityType.CREDIT_DEBIT_CARD,
        EntityType.API_KEY,
        EntityType.IPV4,
        EntityType.IPV6,
        EntityType.UNIQUE_IDENTIFIER,
    }
)
"""Entity types whose columns never give two different originals the same replacement."""

_NAME_PART_ENTITY_TYPES = frozenset({EntityType.FIRST_NAME, EntityType.MIDDLE_NAME, EntityType.LAST_NAME})

ReplacementOutcome = Literal["new", "repeat", "kept_name_part", "kept_birth_date"]
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


class ReplacementOwners:
    """The original each replacement went to, for one identifier column or one shared format.

    A replacement may repeat for the same original, as when it appears in several records, but two
    different originals must not share one. Values are compared ignoring letter case.

    Example:
        After ``record("X-1", "a-7")``, ``"x-1"`` is taken for original ``"B-2"`` but not for ``"A-7"``.
    """

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}

    def record(self, replacement: str, original: str) -> None:
        """Note that ``replacement`` went to ``original``; the first original to get a value keeps it."""
        self._owners.setdefault(replacement.casefold(), original.casefold())

    def taken_for(self, original: str) -> _TakenByOtherOriginals:
        """Return the case-folded replacements that ``original`` must not get."""
        return _TakenByOtherOriginals(self._owners, original.casefold())


class _TakenByOtherOriginals(Container[str]):
    """Case-folded replacements already given to an original other than one."""

    def __init__(self, owners: dict[str, str], original: str) -> None:
        self._owners = owners
        self._original = original

    def __contains__(self, value: object) -> bool:
        owner = self._owners.get(value) if isinstance(value, str) else None
        return owner is not None and owner != self._original


class CandidateChooser:
    """Choose one replacement from successive generated candidates under the collision rules.

    Offer candidates until one is accepted as new or the caller runs out of attempts, then take the
    best fallback. The chooser does not record the chosen value; the caller does.

    Args:
        entity_type: Entity type of the value being replaced.
        original: Original text the replacement must differ from.
        pattern: Plan pattern of the target, used to read full names and skip their fixed words.
        used: Case-folded values this original should not get because other originals have them, in its
            column and in the columns that share replacements with it. ``None`` when replacements may
            repeat. A candidate in it is kept only as a repeat fallback.
        one_to_one: The same values for a grouping column. When given, a candidate in this set is
            rejected outright and no repeat fallback is allowed, so two groups never share one.
        written_suffix: Text written after every candidate, such as the original locality kept after
            a generated street. The one-to-one check applies to the candidate followed by this text,
            since that is the value two groups must not share.
        locale: Locale whose name order full names are read in.
    """

    def __init__(
        self,
        entity_type: EntityType,
        original: str,
        pattern: str | None = None,
        *,
        used: Container[str] | None = None,
        one_to_one: Container[str] | None = None,
        written_suffix: str = "",
        locale: str = "en_US",
    ) -> None:
        self._entity_type = entity_type
        self._original = original
        self._pattern = pattern
        self._locale = locale
        self._used = used
        self._one_to_one = one_to_one
        self._written_suffix = written_suffix
        self._repeat: str | None = None
        self._kept_name_part: str | None = None
        self._kept_birth_date: str | None = None

    def offer(self, request: ReplacementGenerationRequest, candidate: str) -> bool:
        """Return whether ``candidate`` can be used as a new replacement; otherwise keep it as a fallback."""
        if not generated_value_is_valid(request, candidate):
            # A name part read from a replaced full name can only equal the original when that full name kept
            # the part; it is then counted as a kept name part instead of failing the run.
            if _inherits_original_name_part(request, candidate):
                self._kept_name_part = self._kept_name_part or candidate
            # A birth date only prints unchanged when its pattern has no day, as with ``%Y``, or when it is a
            # placeholder such as ``0001-01-01`` that the shift leaves alone; keeping it beats failing the run.
            elif request.entity_type is EntityType.DATE_OF_BIRTH and candidate:
                self._kept_birth_date = self._kept_birth_date or candidate
            return False
        if self._one_to_one is not None and (candidate + self._written_suffix).casefold() in self._one_to_one:
            return False
        # A full name that differs as a whole can still keep the real first or last name. Prefer one that
        # keeps neither; fall back to it only when no other name was found, so a plan whose dependencies
        # fix a name part cannot fail the run.
        if self._entity_type is EntityType.FULL_NAME and keeps_original_name_part(
            self._original, candidate, self._pattern, locale=self._locale
        ):
            self._kept_name_part = self._kept_name_part or candidate
            return False
        if self._used is not None and candidate.casefold() in self._used:
            self._repeat = self._repeat or candidate
            return False
        return True

    def fallback(self) -> tuple[str, ReplacementOutcome] | None:
        """Return the best rejected candidate and how it falls short, or ``None`` if none is usable.

        Repeating an earlier replacement is preferred over keeping an original name part, and both over
        keeping an original birth date. A one-to-one target only accepts a kept name part.
        """
        if self._one_to_one is None and self._repeat is not None:
            return self._repeat, "repeat"
        if self._kept_name_part is not None:
            return self._kept_name_part, "kept_name_part"
        if self._one_to_one is None and self._kept_birth_date is not None:
            return self._kept_birth_date, "kept_birth_date"
        return None


def _inherits_original_name_part(request: ReplacementGenerationRequest, candidate: str) -> bool:
    """Return whether ``candidate`` is a name part equal to its original and read from a full-name dependency."""
    return (
        request.entity_type in _NAME_PART_ENTITY_TYPES
        and candidate.strip().casefold() == request.original_value.strip().casefold()
        and any(entity_type is EntityType.FULL_NAME for entity_type, _ in request.effective_dependency_tuple)
    )
