# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from random import Random

import pytest

from nemo_safe_synthesizer.config.replace_pii import EntityType
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.planning.patterns import (
    describe_pattern,
    extract_name_components,
    render_character_mask,
    render_name_pattern,
)


@pytest.mark.unit
class TestPatternRendering:
    def test_invalid_character_mask_raises_parameter_error(self) -> None:
        with pytest.raises(ParameterError):
            render_character_mask("[", Random(0))

    def test_email_name_parts_join_their_words_with_dots(self) -> None:
        values = {"first": "Mary Ann", "last": "De  La Cruz", "domain": "example.com"}

        email = render_name_pattern(EntityType.EMAIL, "{first}_{Last}@{domain}", values, Random(0))
        full_name = render_name_pattern(EntityType.FULL_NAME, "{First} {Last}", values, Random(0))

        assert email == "mary.ann_De.La.Cruz@example.com"
        assert full_name == "Mary Ann De  La Cruz"

    def test_invalid_name_pattern_raises_parameter_error(self) -> None:
        with pytest.raises(ParameterError):
            render_name_pattern(EntityType.EMAIL, "{unknown}@example.com", {}, Random(0))


@pytest.mark.unit
class TestPatternDescription:
    def test_character_mask_lists_tokens_and_choices_but_not_escapes(self) -> None:
        assert describe_pattern(EntityType.UNIQUE_IDENTIFIER, r"ID-&#[01]#\#") == (
            ("&", "digit or uppercase letter"),
            ("#", "digit 0-9"),
            ("[01]", "one of 0, 1"),
        )

    def test_email_lists_placeholders_and_digits(self) -> None:
        symbols = dict(describe_pattern(EntityType.EMAIL, "{f}.{last}##@{domain}"))

        assert list(symbols) == ["{f}", "{last}", "{domain}", "#"]
        assert symbols["{f}"] == "first-name initial"

    def test_date_is_described_by_an_example(self) -> None:
        assert describe_pattern(EntityType.DATE_OF_BIRTH, "%m/%d/%Y") == (("%m/%d/%Y", "a date such as 06/25/1987"),)


@pytest.mark.unit
class TestNameComponentExtraction:
    def test_separated_placeholders_are_extracted(self) -> None:
        components = extract_name_components(EntityType.FULL_NAME, "{Last}, {First}", "Lovelace, Ada")

        assert components == ((EntityType.LAST_NAME, "Lovelace"), (EntityType.FIRST_NAME, "Ada"))

    @pytest.mark.parametrize(
        ("pattern", "value"),
        [("{First}{Last}", "AdaLovelace"), ("{F}{Last}", "ALovelace"), ("{First} {Middle}{Last}", "Ada ByronLovelace")],
    )
    def test_adjacent_placeholders_are_not_split_by_guessing(self, pattern: str, value: str) -> None:
        assert extract_name_components(EntityType.FULL_NAME, pattern, value) == ()
