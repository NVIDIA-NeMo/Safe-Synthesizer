# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name and address conventions: reading them from a column (singular dominant pattern)."""

from __future__ import annotations

import re
from random import Random

import pandas as pd
import pytest

from tests.pii_replacer.helpers import FIRSTS, LASTS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("SMITH, Jane", "{LAST}, {First}"),
        ("Jane Smith", "{First} {Last}"),
        ("jane smith", "{first} {last}"),
        ("Smith, J.", "{Last}, {F}."),
        ("John Q Smith", "{First} {M} {Last}"),
    ],
    ids=["comma_caps", "plain", "lower", "initial", "middle"],
)
def test_a_name_reads_as_the_convention_it_follows(value: str, expected: str):
    from nemo_safe_synthesizer.pii_replacer.patterns import infer_name_pattern, split_full_name

    assert infer_name_pattern(value, split_full_name(value)) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("j.smith@acme.com", "{f}.{last}@{domain}"),
        ("JSmith@acme.com", "{F}{Last}@{domain}"),
        ("jane_smith@mail.co.uk", "{first}_{last}@{domain}"),
    ],
    ids=["initial_dot", "camel", "underscore"],
)
def test_an_email_reads_as_the_convention_it_follows(value: str, expected: str):
    from nemo_safe_synthesizer.pii_replacer.patterns import infer_email_pattern

    assert infer_email_pattern(value, {"first_name": "Jane", "last_name": "Smith"}) == expected


def test_a_name_column_spells_a_part_the_way_the_column_does():
    """'Pella Y.' is the whole of what this column calls a first name, period and all."""
    from nemo_safe_synthesizer.pii_replacer.patterns import infer_name_pattern

    assert infer_name_pattern("Pella Y.", {"first_name": "Pella Y."}) == "{First}"


@pytest.mark.parametrize(
    ("value", "parts", "expected"),
    [
        pytest.param(
            "jane.smith-jones@acme.com",
            {"first_name": "Jane", "last_name": "Smith-Jones"},
            "{first}.{last}@{domain}",
            id="hyphen_kept",
        ),
        pytest.param(
            "janesmithjones@acme.com",
            {"first_name": "Jane", "last_name": "Smith-Jones"},
            "{first}{last}@{domain}",
            id="hyphen_dropped",
        ),
        pytest.param(
            "pella-y.galenor-quill@dir.invalid",
            {"first_name": "Pella Y.", "last_name": "Galenor-Quill"},
            "{first}.{last}@{domain}",
            id="first_name_holds_an_initial",
        ),
        pytest.param(
            "delacruz.m@acme.com",
            {"first_name": "Maria", "last_name": "de la Cruz"},
            "{last}.{f}@{domain}",
            id="spaces_dropped",
        ),
    ],
)
def test_an_address_reads_a_part_however_it_spells_it(value: str, parts: dict[str, str], expected: str):
    from nemo_safe_synthesizer.pii_replacer.patterns import infer_email_pattern

    assert infer_email_pattern(value, parts) == expected


def test_a_numbered_address_names_the_position_of_its_number_not_the_number():
    from nemo_safe_synthesizer.pii_replacer.patterns import infer_email_pattern, render_email_pattern

    original = "j.smith2019@acme.com"
    pattern = infer_email_pattern(original, {"first_name": "Jane", "last_name": "Smith"})
    assert pattern == "{f}.{last}####@{domain}"

    written = render_email_pattern(pattern, {"first_name": "Robert", "last_name": "Jones"}, original, Random(0))
    assert written is not None
    assert re.fullmatch(r"r\.jones\d{4}@acme\.com", written)
    assert "2019" not in written


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("{LAST}, {First}", "JONES, Robert"),
        ("{Last}, {F}.", "Jones, R."),
        ("{First} {M}. {Last}", "Robert Jones"),
    ],
    ids=["comma_caps", "initial", "absent_part"],
)
def test_a_name_is_written_the_way_the_pattern_reads(pattern: str, expected: str):
    from nemo_safe_synthesizer.pii_replacer.patterns import render_name_pattern

    assert render_name_pattern(pattern, {"first_name": "Robert", "last_name": "Jones"}) == expected


def test_a_numbered_address_column_still_names_one_convention(fixture_numbered_email_df: pd.DataFrame):
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import name_column_pattern

    fields = {"full_name": "contact_name", "email": "contact_email"}
    pattern = name_column_pattern(fixture_numbered_email_df, "email", "contact_email", fields, Config())
    assert pattern == "{first}.{last}.###-####@{domain}"


def test_a_column_of_several_name_conventions_omits_when_none_dominates():
    """Five conventions at equal share → no dominant pattern (≥ 85%)."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import name_column_pattern

    rows = []
    for i in range(60):
        first, last = FIRSTS[i % 10], LASTS[(i // 10) % 10]
        rows.append(
            {
                "patient_name": [
                    f"{last.upper()}, {first}",
                    f"{first} {last}",
                    f"{first.lower()} {last.lower()}",
                    f"{last}, {first[0]}.",
                    f"{first[0]}. {last}",
                ][i % 5]
            }
        )
    df = pd.DataFrame(rows)
    assert name_column_pattern(df, "full_name", "patient_name", {"full_name": "patient_name"}, Config()) is None


def test_a_column_names_its_dominant_convention_only():
    """80% plain / 20% caps — under 85% so omit; 90% plain names the majority."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import name_column_pattern

    rows_mixed = []
    for i in range(50):
        first, last = FIRSTS[i % 10], LASTS[(i // 10) % 10]
        rows_mixed.append({"patient_name": f"{first} {last}" if i % 5 else f"{last.upper()}, {first}"})
    assert (
        name_column_pattern(
            pd.DataFrame(rows_mixed), "full_name", "patient_name", {"full_name": "patient_name"}, Config()
        )
        is None
    )

    rows_dom = []
    for i in range(50):
        first, last = FIRSTS[i % 10], LASTS[(i // 10) % 10]
        # 46/50 = 92% plain
        rows_dom.append({"patient_name": f"{first} {last}" if i >= 4 else f"{last.upper()}, {first}"})
    assert (
        name_column_pattern(
            pd.DataFrame(rows_dom), "full_name", "patient_name", {"full_name": "patient_name"}, Config()
        )
        == "{First} {Last}"
    )


def test_an_address_column_with_no_convention_keeps_its_domains(fixture_contact_df: pd.DataFrame):
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import name_column_pattern

    df = fixture_contact_df.copy()
    df["work_email"] = [f"{'u' * (1 + i % 30)}{i}@acme.com" for i in range(len(df))]
    fields = {"full_name": "patient_name", "email": "work_email"}
    assert name_column_pattern(df, "email", "work_email", fields, Config()) == "@{domain}"


def test_a_column_of_handles_names_the_shape_it_writes_them_in():
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import name_column_pattern

    df = pd.DataFrame(
        {
            "patient_name": [f"{FIRSTS[i % 10]} {LASTS[(i // 10) % 10]}" for i in range(40)],
            "work_email": [f"usr{4700 + i}@acme.com" for i in range(40)],
        }
    )
    fields = {"full_name": "patient_name", "email": "work_email"}
    assert name_column_pattern(df, "email", "work_email", fields, Config()) == "usr47[0123]#@{domain}"


def test_mixed_handle_and_person_email_omits_when_neither_dominates():
    """~1/3 handle + ~2/3 person address — neither reaches 85%."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import name_column_pattern

    rows = []
    for i in range(60):
        first, last = FIRSTS[i % 10], LASTS[(i // 10) % 10]
        addressed = f"{first.lower()}.{last.lower()}@acme.com" if i % 3 else f"usr{4700 + i}@acme.com"
        rows.append({"patient_name": f"{first} {last}", "work_email": addressed})
    df = pd.DataFrame(rows)
    fields = {"full_name": "patient_name", "email": "work_email"}
    assert name_column_pattern(df, "email", "work_email", fields, Config()) is None
