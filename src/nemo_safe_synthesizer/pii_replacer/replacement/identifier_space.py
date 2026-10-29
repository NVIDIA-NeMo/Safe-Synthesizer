# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite identifier value spaces with deterministic allocation.

A replacement never equals its own original, and in a column whose originals are
all distinct, replacements should stay distinct too. Drawing random values and
retrying fails once a value space is nearly full, as with a dense row counter.
These spaces instead enumerate their values, so allocation starts at a seeded
random position and steps to the next usable value. When a column that must stay
distinct has no unused value left, allocation repeats an earlier replacement.
"""

from __future__ import annotations

import math
import re
from collections.abc import Container, Sequence
from dataclasses import dataclass
from random import Random
from typing import Literal

from ...config.replace_pii import EntityType
from ...errors import ParameterError
from ..planning.patterns import parse_character_mask

__all__ = [
    "MASK_ALLOCATED_ENTITY_TYPES",
    "NUMBER_ALLOCATED_ENTITY_TYPES",
    "AllocationKind",
    "IdentifierAllocator",
    "MaskSpace",
    "NumberSpace",
    "identifier_allocator",
    "number_space_for",
]

MASK_ALLOCATED_ENTITY_TYPES = frozenset({EntityType.UNIQUE_IDENTIFIER, EntityType.API_KEY, EntityType.PHONE_NUMBER})
"""Entity types whose character-mask pattern defines the whole replacement value space."""

NUMBER_ALLOCATED_ENTITY_TYPES = frozenset({EntityType.UNIQUE_IDENTIFIER, EntityType.API_KEY})
"""Entity types whose unpatterned, digit-only values are replaced from a numeric range."""

AllocationKind = Literal["new", "repeat"]
"""How an allocated replacement was chosen: a usable value, or a repeat of an earlier replacement."""

_MIN_FREE_RATIO = 2
"""A numeric range is widened until it has at least this many unused values per distinct original."""

_INTEGER_TEXT = re.compile(r"[0-9]+(?:\.0+)?")


@dataclass(frozen=True, slots=True)
class MaskSpace:
    """Every value a character mask can produce, in a fixed order.

    Example:
        ``MaskSpace.from_pattern("USER-##")`` has 100 values, ``"USER-00"``
        through ``"USER-99"``; ``index_of("USER-42")`` is 42.
    """

    positions: tuple[str | tuple[str, ...], ...]
    """Each mask position: a literal character, or the characters a generated position may produce."""

    @classmethod
    def from_pattern(cls, pattern: str) -> MaskSpace:
        """Parse ``pattern`` as a character mask.

        Raises:
            ParameterError: If ``pattern`` is not a valid character mask.
        """
        parts, error = parse_character_mask(pattern)
        if error is not None or parts is None:
            raise ParameterError(error or "invalid character mask")
        positions: list[str | tuple[str, ...]] = []
        for part in parts:
            if part.literal is not None:
                positions.append(part.literal)
            else:
                positions.append(tuple(dict.fromkeys(part.choices or "")))
        return cls(tuple(positions))

    @property
    def size(self) -> int:
        """Number of distinct values the mask can produce."""
        return math.prod(len(position) for position in self.positions if isinstance(position, tuple))

    @property
    def folded_size(self) -> int:
        """Number of values that still differ when letter case is ignored.

        Example:
            ``MaskSpace.from_pattern("*")`` has 62 values but only 36 ignoring case.
        """
        return math.prod(
            len({character.casefold() for character in position})
            for position in self.positions
            if isinstance(position, tuple)
        )

    def value_at(self, index: int) -> str:
        """Return the value at ``index``, where ``0 <= index < size``."""
        characters: list[str] = []
        for position in reversed(self.positions):
            if isinstance(position, str):
                characters.append(position)
                continue
            index, offset = divmod(index, len(position))
            characters.append(position[offset])
        return "".join(reversed(characters))

    def widened(self) -> MaskSpace | None:
        """Return this mask with its last multi-choice position repeated, or ``None`` if it has none.

        Only a generated position with more than one allowed character, ignoring letter case, adds room,
        since replacements are compared ignoring case. So masks such as ``ID-[0]`` and ``ID-[aA]`` cannot widen.

        Example:
            ``MaskSpace.from_pattern("ID-#").widened()`` produces ``"ID-00"`` through ``"ID-99"``.
        """
        for position_index in range(len(self.positions) - 1, -1, -1):
            position = self.positions[position_index]
            if isinstance(position, tuple) and len({character.casefold() for character in position}) > 1:
                return MaskSpace(
                    (*self.positions[: position_index + 1], position, *self.positions[position_index + 1 :])
                )
        return None

    def index_of(self, value: str) -> int | None:
        """Return the index of ``value`` in this space, or ``None`` if the mask cannot produce it."""
        if len(value) != len(self.positions):
            return None
        index = 0
        for character, position in zip(value, self.positions, strict=True):
            if isinstance(position, str):
                if character != position:
                    return None
                continue
            if character not in position:
                return None
            index = index * len(position) + position.index(character)
        return index


@dataclass(frozen=True, slots=True)
class NumberSpace:
    """Non-negative integers below ``10**digits``, written without padding.

    Example:
        ``NumberSpace(3)`` has the 1,000 values ``"0"`` through ``"999"``;
        ``index_of("042")`` and ``index_of("42.0")`` are both 42.
    """

    digits: int
    """Maximum number of digits in a value."""

    @property
    def size(self) -> int:
        """Number of distinct values in the range."""
        return 10**self.digits

    def value_at(self, index: int) -> str:
        """Return the value at ``index``, where ``0 <= index < size``."""
        return str(index)

    def index_of(self, value: str) -> int | None:
        """Return the integer ``value`` denotes if it is in range, else ``None``."""
        number = _integer_value(value)
        return number if number is not None and number < self.size else None


class IdentifierAllocator:
    """Hand out replacement values from one space; each differs from its own original.

    Attributes:
        space: Value space replacements are drawn from.
    """

    def __init__(self, space: MaskSpace | NumberSpace) -> None:
        self.space = space
        self._allocated: list[int] = []
        self._allocated_set: set[int] = set()

    @property
    def free_count(self) -> int:
        """Number of values not yet allocated to a column that must stay distinct."""
        return self.space.size - len(self._allocated_set)

    def allocate(
        self, seed: int, own_original: str, *, avoid: Container[str] | None
    ) -> tuple[str, AllocationKind] | None:
        """Return a replacement for ``own_original`` and how it was chosen, or ``None`` if none differs from it.

        The replacement is the first value at or after a position drawn uniformly from the whole space
        by ``seed`` that differs from ``own_original`` ignoring letter case: the executor writes each
        replacement in its occurrence's case, which would turn ``"A"`` back into an original ``"a"``. When
        ``avoid`` is given, the column must stay distinct: the replacement must also not have been
        allocated here before or be in ``avoid``, ignoring case. If no such value is left, a seeded choice
        among earlier replacements is repeated.

        The start is drawn over the whole space rather than taken as ``seed % size``: a seed has 64 bits,
        and a long mask such as ``ADM-`` followed by 32 alphanumeric positions has far more, so a modulo
        would only ever vary the last positions and leave the rest at the first allowed character.

        Args:
            seed: Seed that picks the starting position.
            own_original: The original value being replaced.
            avoid: Case-folded values the column must not get, such as values it received from another
                column, or ``None`` when replacements may repeat.

        Example:
            In ``NumberSpace(1)``, seed 9 starts at 7, so ``allocate(9, "7", avoid=None)`` skips ``"7"``
            and returns ``("8", "new")``.
        """
        size = self.space.size
        own_index = self.space.index_of(own_original)
        own_folded = own_original.casefold()
        distinct = avoid is not None
        index = Random(seed).randrange(size)
        for _ in range(size):
            if index != own_index and not (distinct and index in self._allocated_set):
                value = self.space.value_at(index)
                folded = value.casefold()
                if folded != own_folded and (avoid is None or folded not in avoid):
                    if distinct:
                        self._allocated.append(index)
                        self._allocated_set.add(index)
                    return value, "new"
            index = (index + 1) % size
        repeats = [
            value
            for value in (self.space.value_at(allocated) for allocated in self._allocated)
            if value.casefold() != own_folded
        ]
        if repeats:
            return repeats[seed % len(repeats)], "repeat"
        return None


def number_space_for(originals: Sequence[str], *, distinct: bool) -> NumberSpace | None:
    """Return a numeric range for replacing ``originals``, or ``None`` unless they are all integers.

    The range covers every integer up to the longest original's digit count. When the replacements must
    stay distinct, it gains one digit at a time until it holds at least twice as many values as there
    are distinct originals. Originals must be non-negative integers, written with digits only or with a
    ``.0`` fraction.

    Example:
        For a unique key ``"0"`` through ``"23485"``, the result is ``NumberSpace(5)``: 100,000 values
        for 23,486 originals.
    """
    numbers = {_integer_value(value) for value in originals}
    if not numbers or None in numbers:
        return None
    integers = {number for number in numbers if number is not None}
    digits = len(str(max(integers)))
    while distinct and 10**digits < _MIN_FREE_RATIO * len(integers):
        digits += 1
    return NumberSpace(digits)


def identifier_allocator(
    entity_type: EntityType,
    pattern: str | None,
    originals: Sequence[str],
    *,
    distinct: bool,
) -> IdentifierAllocator | None:
    """Return the allocator for an identifier column, or ``None`` if the generator should produce its values.

    Args:
        entity_type: The column's entity type.
        pattern: The column's character mask, if any.
        originals: Text forms of the column's non-missing original values.
        distinct: Whether the column's replacements must stay distinct, which sizes a numeric range.
    """
    if pattern is not None and entity_type in MASK_ALLOCATED_ENTITY_TYPES:
        return IdentifierAllocator(MaskSpace.from_pattern(pattern))
    if pattern is None and entity_type in NUMBER_ALLOCATED_ENTITY_TYPES:
        space = number_space_for(originals, distinct=distinct)
        if space is not None:
            return IdentifierAllocator(space)
    return None


def _integer_value(value: str) -> int | None:
    if _INTEGER_TEXT.fullmatch(value) is None:
        return None
    return int(value.split(".", 1)[0])
