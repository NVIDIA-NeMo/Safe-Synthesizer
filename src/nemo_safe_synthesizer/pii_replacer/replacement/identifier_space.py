# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite identifier value spaces with deterministic allocation.

Identifier replacements should be unique and should not equal any original value
in the column. Drawing random values and retrying fails once a value space is
nearly full, as with a dense row counter. These spaces instead enumerate their
values, so allocation starts at a seeded position and steps to the next unused
value. When no unused value is left, allocation repeats an earlier replacement,
and only when none exists does it use another row's original. A replacement
never equals its own original.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
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

AllocationKind = Literal["new", "repeat", "original"]
"""How an allocated replacement was chosen: an unused value, an earlier replacement, or another original."""

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
        """Return this mask with its last generated position repeated, or ``None`` if it has none.

        Example:
            ``MaskSpace.from_pattern("ID-#").widened()`` produces ``"ID-00"`` through ``"ID-99"``.
        """
        for position_index in range(len(self.positions) - 1, -1, -1):
            position = self.positions[position_index]
            if isinstance(position, tuple):
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
    """Hand out replacement values from one space, preferring unused values, then earlier replacements.

    Attributes:
        space: Value space replacements are drawn from.
    """

    def __init__(self, space: MaskSpace | NumberSpace, originals: Iterable[str]) -> None:
        self.space = space
        self._originals = {index for value in originals if (index := space.index_of(value)) is not None}
        self._allocated: list[int] = []
        self._allocated_set: set[int] = set()

    @property
    def free_count(self) -> int:
        """Number of values that are neither originals nor already allocated."""
        return self.space.size - len(self._originals) - len(self._allocated_set)

    def allocate(
        self,
        seed: int,
        own_original: str,
        *,
        is_taken: Callable[[str], bool] | None = None,
    ) -> tuple[str, AllocationKind] | None:
        """Return a replacement for ``own_original`` and how it was chosen, or ``None`` if none differs from it.

        The replacement is the first unused value at or after the seeded position. A value is unused when
        it is neither an original of this space nor allocated here, and ``is_taken`` (for values used or
        found elsewhere, such as another column of the same type) does not reject it. When no unused value
        is left, a seeded choice among earlier replacements is repeated. When there are none, a seeded
        choice among the other originals is used. The result never equals ``own_original``.
        """
        size = self.space.size
        if self.free_count > 0:
            index = seed % size
            # free_count ignores is_taken, so the scan is bounded by the space size.
            for _ in range(size):
                if index not in self._originals and index not in self._allocated_set:
                    value = self.space.value_at(index)
                    if is_taken is None or not is_taken(value):
                        self._allocated.append(index)
                        self._allocated_set.add(index)
                        return value, "new"
                index = (index + 1) % size
        if self._allocated:
            return self.space.value_at(self._allocated[seed % len(self._allocated)]), "repeat"
        others = sorted(self._originals - {self.space.index_of(own_original)})
        if others:
            return self.space.value_at(others[seed % len(others)]), "original"
        return None


def number_space_for(originals: Sequence[str]) -> NumberSpace | None:
    """Return a numeric range that fits every distinct original and leaves room to replace them.

    The range starts with the longest original's digit count and gains one digit
    at a time until it has at least twice as many unused values as there are
    distinct originals. Returns ``None`` unless every original is a non-negative
    integer, written with digits only or with a ``.0`` fraction.

    Example:
        For a row counter ``"0"`` through ``"23485"``, the result is
        ``NumberSpace(5)``: 76,514 unused values for 23,486 originals.
    """
    numbers = {_integer_value(value) for value in originals}
    if not numbers or None in numbers:
        return None
    integers = {number for number in numbers if number is not None}
    digits = len(str(max(integers)))
    while 10**digits - sum(number < 10**digits for number in integers) < _MIN_FREE_RATIO * len(integers):
        digits += 1
    return NumberSpace(digits)


def identifier_allocator(
    entity_type: EntityType,
    pattern: str | None,
    originals: Sequence[str],
) -> IdentifierAllocator | None:
    """Return the allocator for an identifier column, or ``None`` if the generator should produce its values.

    Args:
        entity_type: The column's entity type.
        pattern: The column's character mask, if any.
        originals: Text forms of the column's non-missing original values.
    """
    if pattern is not None and entity_type in MASK_ALLOCATED_ENTITY_TYPES:
        return IdentifierAllocator(MaskSpace.from_pattern(pattern), originals)
    if pattern is None and entity_type in NUMBER_ALLOCATED_ENTITY_TYPES:
        space = number_space_for(originals)
        if space is not None:
            return IdentifierAllocator(space, originals)
    return None


def _integer_value(value: str) -> int | None:
    if _INTEGER_TEXT.fullmatch(value) is None:
        return None
    return int(value.split(".", 1)[0])
