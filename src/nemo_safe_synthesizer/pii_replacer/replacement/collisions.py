# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision rules shared by structured columns and free-text spans.

Every replacement, wherever it is written, follows the same rules:

1. It differs from its own original, ignoring letter case.
2. Identifier-like values avoid every known original and every replacement already used for their
   entity type, ignoring case, across all columns and free text in the run.
3. When retries cannot avoid them, the fallback order is the same everywhere: repeat an earlier
   replacement, then use another original, then (for full names) keep an original name part. Each
   fallback is counted so the caller can warn.

A grouping column is stricter: its replacements must be one-to-one, so two groups never merge, and
no fallback is allowed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from ...config.replace_pii import EntityType
from .value_checks import generated_value_is_valid, keeps_original_name_part

if TYPE_CHECKING:
    from .generation import ReplacementGenerationRequest

__all__ = ["IDENTIFIER_ENTITY_TYPES", "CandidateChooser", "ReplacementOutcome", "ReplacementPool"]

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
"""Entity types whose replacements avoid other originals and earlier replacements of the same type."""

ReplacementOutcome = Literal["new", "repeat", "original", "kept_name_part"]
"""How a replacement was chosen: a new value, or one of the fallbacks in preference order."""


class ReplacementPool:
    """Originals and used replacements of identifier-like types, compared ignoring letter case.

    One pool is shared by every structured column and free-text span of a run, so an SSN replaced in
    an ``ssn`` column and another detected in notes never get the same replacement, and neither
    replacement equals a real SSN found anywhere in the input. Other entity types are not tracked.

    Example:
        After ``reserve_originals(EntityType.EMAIL, ["Ada@x.com"])``, ``is_original(EntityType.EMAIL,
        "ada@X.com")`` is ``True``.
    """

    def __init__(self) -> None:
        self._originals: dict[EntityType, set[str]] = {}
        self._used: dict[EntityType, set[str]] = {}

    def clear(self) -> None:
        """Forget every original and replacement, as before a new dataframe execution."""
        self._originals.clear()
        self._used.clear()

    @staticmethod
    def tracks(entity_type: EntityType) -> bool:
        """Return whether ``entity_type`` is identifier-like and therefore tracked."""
        return entity_type in IDENTIFIER_ENTITY_TYPES

    def reserve_originals(self, entity_type: EntityType, values: Iterable[str]) -> None:
        """Record original values that replacements of ``entity_type`` must avoid."""
        if self.tracks(entity_type):
            self._originals.setdefault(entity_type, set()).update(value.casefold() for value in values)

    def record_used(self, entity_type: EntityType, value: str) -> None:
        """Record a replacement so later replacements of ``entity_type`` avoid it."""
        if self.tracks(entity_type):
            self._used.setdefault(entity_type, set()).add(value.casefold())

    def is_original(self, entity_type: EntityType, value: str) -> bool:
        """Return whether ``value`` equals a known original of ``entity_type``, ignoring case."""
        return value.casefold() in self._originals.get(entity_type, ())

    def is_used(self, entity_type: EntityType, value: str) -> bool:
        """Return whether ``value`` equals an earlier replacement of ``entity_type``, ignoring case."""
        return value.casefold() in self._used.get(entity_type, ())

    def is_taken(self, entity_type: EntityType, value: str) -> bool:
        """Return whether ``value`` is a known original or an earlier replacement of ``entity_type``."""
        return self.is_original(entity_type, value) or self.is_used(entity_type, value)


class CandidateChooser:
    """Choose one replacement from successive generated candidates under the shared collision rules.

    Offer candidates until one is accepted as new or the caller runs out of attempts, then take the
    best fallback. The chooser does not record the chosen value; the caller records it in the pool.

    Args:
        pool: Run-wide originals and used replacements.
        entity_type: Entity type of the value being replaced.
        original: Original text the replacement must differ from.
        pattern: Plan pattern of the target, used to skip fixed words when comparing full names.
        one_to_one: Case-folded originals and replacements of a grouping column. When given, a
            candidate in this set is rejected outright and no repeat or other-original fallback is
            allowed, so two groups never share a replacement.
        written_suffix: Text written after every candidate, such as the original locality kept after
            a generated street. The one-to-one check applies to the candidate followed by this text,
            since that is the value two groups must not share.
    """

    def __init__(
        self,
        pool: ReplacementPool,
        entity_type: EntityType,
        original: str,
        pattern: str | None = None,
        *,
        one_to_one: set[str] | None = None,
        written_suffix: str = "",
    ) -> None:
        self._pool = pool
        self._entity_type = entity_type
        self._original = original
        self._pattern = pattern
        self._one_to_one = one_to_one
        self._written_suffix = written_suffix
        self._repeat: str | None = None
        self._reused_original: str | None = None
        self._kept_name_part: str | None = None

    def offer(self, request: ReplacementGenerationRequest, candidate: str) -> bool:
        """Return whether ``candidate`` can be used as a new replacement; otherwise keep it as a fallback."""
        if not generated_value_is_valid(request, candidate):
            return False
        if self._one_to_one is not None and (candidate + self._written_suffix).casefold() in self._one_to_one:
            return False
        # A full name that differs as a whole can still keep the real first or last name. Prefer one that
        # keeps neither; fall back to it only when no other name was found, so a plan whose dependencies
        # fix a name part cannot fail the run.
        if self._entity_type is EntityType.FULL_NAME and keeps_original_name_part(
            self._original, candidate, self._pattern
        ):
            self._kept_name_part = self._kept_name_part or candidate
            return False
        if self._pool.is_original(self._entity_type, candidate):
            self._reused_original = self._reused_original or candidate
            return False
        if self._pool.is_used(self._entity_type, candidate):
            self._repeat = self._repeat or candidate
            return False
        return True

    def fallback(self) -> tuple[str, ReplacementOutcome] | None:
        """Return the best rejected candidate and how it falls short, or ``None`` if none is usable.

        Repeating an earlier replacement is preferred over using another original, which is preferred
        over keeping an original name part. A one-to-one target only accepts a kept name part.
        """
        if self._one_to_one is None:
            if self._repeat is not None:
                return self._repeat, "repeat"
            if self._reused_original is not None:
                return self._reused_original, "original"
        if self._kept_name_part is not None:
            return self._kept_name_part, "kept_name_part"
        return None
