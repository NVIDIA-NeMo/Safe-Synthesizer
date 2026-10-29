# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        ("originals", "distinct", "digits"),
        [
            ([str(row) for row in range(23_486)], True, 5),
            ([str(row) for row in range(1, 9)], True, 2),
            (["17", "42", "103"], True, 3),
            ([str(row) for row in range(1, 9)], False, 1),
        ],
    )
    def test_number_space_widens_only_when_replacements_must_stay_distinct(
        self,
        originals: list[str],
        distinct: bool,
        digits: int,
    ) -> None:
        assert number_space_for(originals, distinct=distinct) == NumberSpace(digits)

    def test_number_space_rejects_columns_with_non_integer_values(self) -> None:
        assert number_space_for(["17", "A42"], distinct=True) is None
        assert number_space_for([], distinct=True) is None

    def test_distinct_allocation_uses_every_other_value_before_repeating(self) -> None:
        allocator = IdentifierAllocator(NumberSpace(1))

        first = [allocator.allocate(seed, "0", avoid=set()) for seed in range(9)]
        repeated = [allocator.allocate(seed, "0", avoid=set()) for seed in range(3)]

        assert sorted(first) == [(str(value), "new") for value in range(1, 10)]
        assert all(allocation is not None and allocation[1] == "repeat" for allocation in repeated)

    def test_allocation_without_distinctness_only_skips_the_own_original(self) -> None:
        allocator = IdentifierAllocator(NumberSpace(1))

        allocations = [allocator.allocate(seed, "7", avoid=None) for seed in range(20)]

        assert all(allocation is not None and allocation[1] == "new" for allocation in allocations)
        assert "7" not in {allocation[0] for allocation in allocations if allocation is not None}
        # Seed 9 starts at 7, the own original, so allocation steps to the next value.
        assert allocator.allocate(9, "7", avoid=None) == ("8", "new")

    def test_long_masks_vary_their_leading_positions(self) -> None:
        allocator = IdentifierAllocator(MaskSpace.from_pattern("ADM-" + "&" * 32))

        values = [allocator.allocate(seed, "ADM-" + "0" * 32, avoid=set()) for seed in range(20)]

        leading = {allocation[0][4:16] for allocation in values if allocation is not None}
        assert len(leading) == 20
        assert all(prefix != "0" * 12 for prefix in leading)

    def test_allocator_returns_nothing_when_only_the_own_original_exists(self) -> None:
        allocator = IdentifierAllocator(MaskSpace.from_pattern("ID-[0]"))

        assert allocator.allocate(5, "ID-0", avoid=set()) is None

    def test_allocation_skips_the_own_original_ignoring_letter_case(self) -> None:
        # The executor writes "A" in the case of "a", which would restore the original.
        assert IdentifierAllocator(MaskSpace.from_pattern("[aA]")).allocate(0, "a", avoid=None) is None
        allocator = IdentifierAllocator(MaskSpace.from_pattern("[aAb]"))

        assert {allocator.allocate(seed, "a", avoid=None) for seed in range(20)} == {("b", "new")}

    def test_distinct_allocation_skips_avoided_values_ignoring_letter_case(self) -> None:
        allocator = IdentifierAllocator(MaskSpace.from_pattern("[aAbBcC]"))
        used = {"b"}

        first = allocator.allocate(3, "a", avoid=used)
        assert first is not None and first[0].casefold() == "c"
        used.add(first[0].casefold())

        repeated = allocator.allocate(3, "a", avoid=used)
        assert repeated == (first[0], "repeat")

    def test_folded_size_counts_each_letter_once(self) -> None:
        assert MaskSpace.from_pattern("ID-*").size == 62
        assert MaskSpace.from_pattern("ID-*").folded_size == 36
        assert MaskSpace.from_pattern("ID-##").folded_size == 100

    def test_allocator_is_deterministic_for_the_same_seeds(self) -> None:
        first = IdentifierAllocator(MaskSpace.from_pattern("ID-###"))
        second = IdentifierAllocator(MaskSpace.from_pattern("ID-###"))

        assert [first.allocate(seed, "ID-001", avoid=set()) for seed in (1, 1, 7)] == [
            second.allocate(seed, "ID-001", avoid=set()) for seed in (1, 1, 7)
        ]

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
        allocator = identifier_allocator(entity_type, pattern, originals, distinct=True)

        if expected is None:
            assert allocator is None
        else:
            assert allocator is not None
            assert isinstance(allocator.space, expected)
