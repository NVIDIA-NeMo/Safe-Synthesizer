# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone columns: identifiers, cards, and the shapes their replacements keep."""

from __future__ import annotations

import re

import pandas as pd

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    PiiColumnPlan,
    PiiEntity,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.replacer import (
    TabularPiiReplacer,
)


def test_a_replacement_is_never_the_value_it_replaces():
    """A tight template must not 'replace' a value with itself and leave PII in place."""
    from nemo_safe_synthesizer.pii_replacer.replacement import seeded_faker, unique_synthetic

    # The template has two possible values, so an unretried draw would reproduce
    # the original about half the time.
    fake = seeded_faker(11, "en_US")
    for _ in range(50):
        new = unique_synthetic("555-0000", "phone_number", ["555-000[01]"], fake.random, fake, set())
        assert new == "555-0001"


def test_a_minority_identifier_keeps_the_literals_of_its_own_group():
    """A listed shape holds what the group holds constant, which one value alone cannot say."""
    from nemo_safe_synthesizer.pii_replacer.patterns import value_matches_template
    from nemo_safe_synthesizer.pii_replacer.replacement import seeded_faker, unique_synthetic

    fake = seeded_faker(3, "en_US")
    patterns = ["PMC-######", "ACC-####"]

    common = unique_synthetic("PMC-004821", "unique_identifier", patterns, fake.random, fake, set())
    minority = unique_synthetic("ACC-0042", "unique_identifier", patterns, fake.random, fake, set())
    unlisted = unique_synthetic("004821-AB", "unique_identifier", patterns, fake.random, fake, set())

    assert common is not None and value_matches_template(common, "PMC-######")
    assert minority is not None and value_matches_template(minority, "ACC-####")
    assert unlisted is not None and value_matches_template(unlisted, "######-^^")


def test_a_minority_identifier_keeps_its_own_prefix_end_to_end():
    """The motivating case: 'ACC-0042' in a mostly 'PMC-######' column, not 'TIX-5807'."""
    ids = [f"PMC-{(i * 24851) % 1000000:06d}" for i in range(40)] + [f"ACC-{(i * 937) % 10000:04d}" for i in range(10)]
    df = pd.DataFrame({"record_id": ids, "full_name": [f"Person {i}" for i in range(50)]})
    cfg = ReplacePiiConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    replacer = TabularPiiReplacer(cfg, data_config=DataParameters())
    replacer.transform_df(df)
    assert replacer.result is not None

    replaced = replacer.result.transformed_df["record_id"]
    assert (replaced != df["record_id"]).all()
    assert all(re.fullmatch(r"PMC-\d{6}", value) for value in replaced[:40])
    assert all(re.fullmatch(r"ACC-\d{4}", value) for value in replaced[40:])


def test_a_cell_that_says_it_holds_nothing_keeps_saying_so():
    """Replacing 'N/A' in its own shape writes 'E/S', a value where the data had none."""
    ids = [f"PMC-{(i * 24851) % 1000000:06d}" for i in range(40)] + ["N/A"] * 10
    df = pd.DataFrame({"record_id": ids, "full_name": [f"Person {i}" for i in range(50)]})
    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="record_id", entity_type=PiiEntity.unique_identifier, patterns=["PMC-######"])
        ]
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(),
    )
    replacer.transform_df(df)
    assert replacer.result is not None

    replaced = replacer.result.transformed_df["record_id"]
    assert all(re.fullmatch(r"PMC-\d{6}", value) for value in replaced[:40])
    assert list(replaced[40:]) == ["N/A"] * 10


def test_card_replacement_keeps_the_columns_grouping_and_its_checksum():
    from nemo_safe_synthesizer.pii_replacer.patterns import luhn_valid

    # Luhn-valid 16-digit numbers written the way the column writes them.
    cards = ["4111-1111-1111-1111", "4012-8888-8888-1881", "4222-2222-2222-2220"] * 8
    df = pd.DataFrame({"full_name": [f"Person {i}" for i in range(24)], "card_number": cards})
    cfg = ReplacePiiConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    replacer = TabularPiiReplacer(cfg, data_config=DataParameters())
    replacer.transform_df(df)
    assert replacer.result is not None

    replaced = replacer.result.transformed_df["card_number"]
    assert (replaced != df["card_number"]).all()
    for value in replaced:
        assert re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{4}", value), value
        assert luhn_valid(value.replace("-", "")), value
