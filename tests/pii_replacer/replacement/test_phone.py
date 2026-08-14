# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phone detection, persona vs standalone routing, and format preservation."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    PersonaColumnSet,
    PiiColumnPlan,
    PiiEntity,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.entities import Config
from nemo_safe_synthesizer.pii_replacer.replacer import (
    TabularPiiReplacer,
)
from tests.pii_replacer.helpers import PHONE_MINORITY, PHONE_TEMPLATE


def _replace_phones(df: pd.DataFrame, spec: PiiColumnPlan, *, standalone: bool) -> pd.Series:
    """Run replacement with ``spec`` for the phone column and return the new values."""
    if standalone:
        plan = PiiReplacementPlan(standalone_columns_to_replace=[spec])
    else:
        plan = PiiReplacementPlan(
            persona_backed_columns=[
                PersonaColumnSet(
                    persona="contact",
                    columns_to_replace=[
                        PiiColumnPlan(column_name="full_name", entity_type=PiiEntity.full_name),
                        spec,
                    ],
                )
            ]
        )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(),
    )
    replacer.transform_df(df)
    assert replacer.result is not None
    return replacer.result.transformed_df["phone"]


@pytest.mark.parametrize(
    ("backend", "persona_backed"),
    [("pgm", True), ("managed", False), ("faker", False)],
)
def test_phone_is_persona_backed_only_under_the_pgm(backend: str, persona_backed: bool):
    """Only the PGM generates a number, so only there can a persona supply one."""
    from nemo_safe_synthesizer.pii_replacer.entities import effective_apply_path
    from nemo_safe_synthesizer.pii_replacer.replacement import build_standalone_maps, extract_instances

    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="contact",
                columns_to_replace=[
                    PiiColumnPlan(column_name="full_name", entity_type=PiiEntity.full_name),
                    PiiColumnPlan(column_name="phone", entity_type=PiiEntity.phone_number),
                ],
            )
        ]
    )
    assert (effective_apply_path("phone_number", backend) == "persona") is persona_backed

    df = pd.DataFrame(
        {
            "full_name": ["Alice Smith", "Bob Jones"],
            "phone": ["+1-415-555-0101", "+1-415-555-0102"],
        }
    )
    cfg = Config(
        locale="en_US",
        random_seed=7,
        persona_backend=backend,
        sdg_pgms_src="/tmp",
        managed_assets_path=None,
    )
    instances = extract_instances(df, plan, cfg)
    maps = build_standalone_maps(df, plan, cfg)
    assert any("phone_number" in inst.field_cols for inst in instances) is persona_backed
    assert ("phone" in maps) is not persona_backed


def test_a_persona_phone_is_printed_in_the_columns_format():
    """The PGM number carries the persona's area code, but its own punctuation."""
    from nemo_safe_synthesizer.pii_replacer.replacement import seeded_faker, synth_value

    value = synth_value(
        "phone_number",
        "+1-415-555-0101",
        {"phone_number": "(206) 555-0181"},
        seeded_faker(7, "en_US"),
        ["+1-###-555-####"],
    )

    assert value == "+1-206-555-0181"


def test_a_persona_phone_keeps_its_own_format_in_a_mixed_column():
    """A number written in none of the column's formats is printed in its own."""
    from nemo_safe_synthesizer.pii_replacer.replacement import seeded_faker, synth_value

    value = synth_value(
        "phone_number",
        PHONE_MINORITY,  # (206) 555-0114, one of the column's minority shapes
        {"phone_number": "+1-312-555-0181"},
        seeded_faker(7, "en_US"),
        ["+1-###-555-####"],
    )

    assert value == "(312) 555-0181"


def test_a_persona_phone_is_left_alone_without_a_pattern():
    from nemo_safe_synthesizer.pii_replacer.replacement import seeded_faker, synth_value

    value = synth_value("phone_number", "+1-415-555-0101", {"phone_number": "(206) 555-0181"}, seeded_faker(7, "en_US"))

    assert value == "(206) 555-0181"


@pytest.mark.parametrize("standalone", [False, True], ids=["persona_backed", "standalone"])
def test_a_phone_is_written_in_the_format_it_was_written_in(fixture_phone_df: pd.DataFrame, standalone: bool):
    """Wherever the plan puts the column, a number the listed template misses keeps its own."""
    from nemo_safe_synthesizer.pii_replacer.patterns import value_matches_template

    replaced = _replace_phones(
        fixture_phone_df,
        PiiColumnPlan(column_name="phone", entity_type=PiiEntity.phone_number, patterns=[PHONE_TEMPLATE]),
        standalone=standalone,
    )
    assert (replaced != fixture_phone_df["phone"]).all()
    assert all(value_matches_template(value, PHONE_TEMPLATE) for value in replaced[:18])
    for value in replaced[18:]:
        assert re.fullmatch(r"\(\d{3}\) \d{3}-\d{4}", value), value


@pytest.mark.parametrize("standalone", [False, True], ids=["persona_backed", "standalone"])
def test_a_column_that_lists_both_its_phone_formats_keeps_both(fixture_phone_df: pd.DataFrame, standalone: bool):
    from nemo_safe_synthesizer.pii_replacer.patterns import value_matches_template

    minority_template = "(###) ###-####"
    replaced = _replace_phones(
        fixture_phone_df,
        PiiColumnPlan(
            column_name="phone",
            entity_type=PiiEntity.phone_number,
            patterns=[PHONE_TEMPLATE, minority_template],
        ),
        standalone=standalone,
    )
    assert (replaced != fixture_phone_df["phone"]).all()
    assert all(value_matches_template(value, PHONE_TEMPLATE) for value in replaced[:18])
    assert all(value_matches_template(value, minority_template) for value in replaced[18:])


def test_phone_replacement_without_a_pattern_uses_faker_phone(fixture_phone_df: pd.DataFrame):
    """With no template, standalone phones get real Faker numbers (not character-class noise)."""
    replaced = _replace_phones(
        fixture_phone_df,
        PiiColumnPlan(column_name="phone", entity_type=PiiEntity.phone_number),
        standalone=False,
    )
    assert (replaced != fixture_phone_df["phone"]).all()
    assert replaced.nunique() == fixture_phone_df["phone"].nunique()
    # Digits remain; shape need not match the original (unlike pattern_preserving_token).
    assert all(re.search(r"\d", v) for v in replaced)


def test_rows_differing_only_by_phone_share_one_persona(fixture_phone_df: pd.DataFrame):
    """A phone is no longer part of a person's identity, so it cannot split one in two."""
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="contact",
                columns_to_replace=[
                    PiiColumnPlan(column_name="full_name", entity_type=PiiEntity.full_name),
                    PiiColumnPlan(column_name="phone", entity_type=PiiEntity.phone_number),
                ],
            )
        ]
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(),
    )
    replacer.transform_df(fixture_phone_df)
    assert replacer.result is not None
    names = replacer.result.transformed_df["full_name"]
    # Rows 0 and 10 are the same contact reached at two different numbers.
    assert fixture_phone_df["full_name"][0] == fixture_phone_df["full_name"][10]
    assert names[0] == names[10]
