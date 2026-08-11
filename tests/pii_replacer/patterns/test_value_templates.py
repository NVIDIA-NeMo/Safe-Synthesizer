# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Character templates: the shapes a column of values wears, and which it names."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from tests.pii_replacer.helpers import PHONE_MINORITY, PHONE_TEMPLATE


def test_value_matches_template():
    from nemo_safe_synthesizer.pii_replacer.patterns import value_matches_template

    assert value_matches_template("+1-415-555-0101", PHONE_TEMPLATE)
    assert not value_matches_template(PHONE_MINORITY, PHONE_TEMPLATE)
    # A template position accepts only its own class: '#' is a digit, never a letter.
    assert not value_matches_template("+1-415-555-01O1", PHONE_TEMPLATE)


def test_a_value_is_written_in_the_first_template_that_describes_it():
    from nemo_safe_synthesizer.pii_replacer.patterns import matching_template

    templates = ["PMC-######", "ACC-####", "###-####"]
    assert matching_template("PMC-004821", templates) == "PMC-######"
    assert matching_template("ACC-0042", templates) == "ACC-####"
    # Both of the last two describe '555-0142'; the one listed first wins.
    assert matching_template("555-0142", templates) == "###-####"
    # A shape none of them describes is the value's own.
    assert matching_template("004821-AB", templates) == "######-^^"


def test_a_template_widens_when_too_few_values_back_it():
    """Pinning a character claims the column holds nothing else, which few values cannot show."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import infer_value_pattern

    cfg = Config()
    thin = infer_value_pattern([f"pmc-{600000 + i}" for i in range(3)], cfg)
    plenty = infer_value_pattern([f"pmc-{600000 + i}" for i in range(40)], cfg)

    # Three IDs sharing a prefix are a coincidence, so only the shape survives;
    # forty of them are the column's format, and the prefix is kept.
    assert thin == "@@@-######"
    assert plenty is not None and plenty.startswith("pmc-")


def test_no_template_when_a_separator_shares_a_position_with_digits():
    """Octets of differing width leave dots unaligned, and a template would scatter them."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import infer_value_pattern

    assert infer_value_pattern(["217.197.215.20", "67.248.207.155", "123.183.111.71"], Config()) is None


def test_a_column_names_both_of_its_shapes_most_common_first():
    """A template is read from the values that wear it, so the smaller group keeps its 'ACC'."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import value_patterns

    # Digits spread over their whole range, so only the prefixes are constant.
    values = pd.Series(
        [f"PMC-{(i * 24851) % 1000000:06d}" for i in range(40)] + [f"ACC-{(i * 937) % 10000:04d}" for i in range(10)]
    )

    assert value_patterns(values, Config()) == ["PMC-######", "ACC-####"]


def test_a_column_of_many_shapes_names_only_the_ones_worth_naming():
    """A tail of one-off shapes says nothing about the column, and each keeps its own anyway."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import PATTERN_MAX_PATTERNS, value_patterns

    values = pd.Series([f"{'X' * (shape + 1)}-{i:04d}" for shape in range(8) for i in range(20)])

    assert len(value_patterns(values, Config())) == PATTERN_MAX_PATTERNS


def test_a_shape_too_few_values_wear_is_left_unnamed():
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import value_patterns

    values = pd.Series([f"PMC-{(i * 24851) % 1000000:06d}" for i in range(99)] + ["ACC-0001"])

    assert value_patterns(values, Config()) == ["PMC-######"]


@pytest.mark.parametrize(
    "sentinel", ["N/A", "none", "UNKNOWN", "-", ""], ids=["na", "none", "unknown", "dash", "empty"]
)
def test_a_cell_that_says_it_holds_nothing_is_no_shape_of_its_own(sentinel: str):
    """'N/A' wears no format, and reading one from it would put it among the column's."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import value_patterns

    values = pd.Series([f"PMC-{(i * 24851) % 1000000:06d}" for i in range(40)] + [sentinel] * 10)

    assert value_patterns(values, Config()) == ["PMC-######"]


def test_a_patterned_card_number_still_adds_up():
    """A card number that fails its checksum is spotted by anything that validates one."""
    from nemo_safe_synthesizer.pii_replacer.patterns import luhn_valid, synth_card_value
    from nemo_safe_synthesizer.pii_replacer.replacement import seeded_faker

    rng = seeded_faker(11, "en_US").random
    numbers = [synth_card_value("4###-####-####-####", rng) for _ in range(50)]

    assert all(re.fullmatch(r"4\d{3}-\d{4}-\d{4}-\d{4}", number) for number in numbers)
    assert all(luhn_valid("".join(c for c in number if c.isdigit())) for number in numbers)
