# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_safe_synthesizer.config.replace_pii import EntityType
from nemo_safe_synthesizer.pii_replacer.replacement.identifier_space import (
    IdentifierAllocator,
    MaskSpace,
    NumberSpace,
    identifier_allocator,
    number_space_for,
)


@pytest.mark.unit
class TestIdentifierSpace:
    def test_mask_space_enumerates_every_value_in_order(self) -> None:
        space = MaskSpace.from_pattern("USER-[AB]#")

        values = [space.value_at(index) for index in range(space.size)]

        assert space.size == 20
        assert values[:3] == ["USER-A0", "USER-A1", "USER-A2"]
        assert values[-1] == "USER-B9"
        assert [space.index_of(value) for value in values] == list(range(20))
        assert space.index_of("USER-C1") is None
        assert space.index_of("ADMIN-A1") is None

    def test_number_space_reads_integers_with_padding_or_a_zero_fraction(self) -> None:
        space = NumberSpace(3)

        assert space.size == 1000
        assert [space.index_of(value) for value in ("042", "42.0", "999", "1000", "4.5", "-1")] == [
            42,
            42,
            999,
            None,
            None,
            None,
        ]

    @pytest.mark.parametrize(
        ("originals", "digits"),
        [
            ([str(row) for row in range(23_486)], 5),
            ([str(row) for row in range(1, 9)], 2),
            (["17", "42", "103"], 3),
        ],
    )
    def test_number_space_widens_until_there_is_room_to_replace_every_value(
        self,
        originals: list[str],
        digits: int,
    ) -> None:
        assert number_space_for(originals) == NumberSpace(digits)

    def test_number_space_rejects_columns_with_non_integer_values(self) -> None:
        assert number_space_for(["17", "A42"]) is None
        assert number_space_for([]) is None

    def test_allocator_uses_every_unused_value_before_repeating_and_never_returns_an_original(self) -> None:
        originals = [str(value) for value in range(90)]
        allocator = IdentifierAllocator(NumberSpace(2), originals)

        unique = [allocator.allocate(seed) for seed in range(10)]
        repeated = [allocator.allocate(seed) for seed in range(5)]

        assert sorted(unique) == [(str(value), False) for value in range(90, 100)]
        assert all(allocation is not None and allocation[1] and int(allocation[0]) >= 90 for allocation in repeated)

    def test_allocator_returns_nothing_when_every_value_is_an_original(self) -> None:
        allocator = IdentifierAllocator(NumberSpace(1), [str(value) for value in range(10)])

        assert allocator.allocate(3) is None

    def test_allocator_is_deterministic_for_the_same_seeds(self) -> None:
        first = IdentifierAllocator(MaskSpace.from_pattern("ID-###"), ["ID-001"])
        second = IdentifierAllocator(MaskSpace.from_pattern("ID-###"), ["ID-001"])

        assert [first.allocate(seed) for seed in (1, 1, 7)] == [second.allocate(seed) for seed in (1, 1, 7)]

    @pytest.mark.parametrize(
        ("entity_type", "pattern", "originals", "expected"),
        [
            (EntityType.UNIQUE_IDENTIFIER, "USER-###", ["USER-001"], MaskSpace),
            (EntityType.PHONE_NUMBER, "###-####", ["555-0101"], MaskSpace),
            (EntityType.UNIQUE_IDENTIFIER, None, ["17", "42"], NumberSpace),
            (EntityType.UNIQUE_IDENTIFIER, None, ["USER-001"], None),
            (EntityType.PHONE_NUMBER, None, ["5550101"], None),
            (EntityType.CREDIT_DEBIT_CARD, "####-####-####-####", ["4111-1111-1111-1111"], None),
        ],
    )
    def test_identifier_allocator_covers_masks_and_digit_only_identifiers(
        self,
        entity_type: EntityType,
        pattern: str | None,
        originals: list[str],
        expected: type | None,
    ) -> None:
        allocator = identifier_allocator(entity_type, pattern, originals)

        if expected is None:
            assert allocator is None
        else:
            assert allocator is not None
            assert isinstance(allocator.space, expected)
