# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name and address conventions: reading them from a column and writing them back."""

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
    """A placeholder is written the way its output is written, so the pattern reads as an example."""
    from nemo_safe_synthesizer.pii_replacer.patterns import infer_persona_pattern, split_full_name

    assert infer_persona_pattern(value, split_full_name(value)) == expected


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
    """'Pella Y.' is the whole of what this column calls a first name, period and all.

    Read run by run, the trailing period falls outside the part and is copied as a
    decoration, writing 'Tuyet.' where the column asked for a first name.
    """
    from nemo_safe_synthesizer.pii_replacer.patterns import infer_persona_pattern
    from nemo_safe_synthesizer.pii_replacer.replacement import persona_written

    assert infer_persona_pattern("Pella Y.", {"first_name": "Pella Y."}) == "{First}"
    written = persona_written("first_name", "Pella Y.", {"first_name": "Tuyet"}, ["{First}"])
    assert written == "Tuyet"


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
    """A part is one part however the address punctuates it.

    Matching the punctuation literally left the part unrecognized, and the scan
    then read its first letter as an initial and copied the rest of the original
    surname into the pattern, as '{l}alenor-quill'.
    """
    from nemo_safe_synthesizer.pii_replacer.patterns import infer_email_pattern

    assert infer_email_pattern(value, parts) == expected


def test_a_numbered_address_names_the_position_of_its_number_not_the_number():
    """Keeping the number would carry a piece of the original address into its replacement."""
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
        # The persona has no middle name, so the part takes its own separator with it.
        ("{First} {M}. {Last}", "Robert Jones"),
    ],
    ids=["comma_caps", "initial", "absent_part"],
)
def test_a_name_is_written_the_way_the_pattern_reads(pattern: str, expected: str):
    from nemo_safe_synthesizer.pii_replacer.patterns import render_persona_pattern

    assert render_persona_pattern(pattern, {"first_name": "Robert", "last_name": "Jones"}) == expected


def test_a_numbered_address_column_still_names_one_convention(fixture_numbered_email_df: pd.DataFrame):
    """Reading each number as part of the convention left every row with one of its own.

    No single one of them then described enough of the column to be worth naming,
    so the column named nothing and every address was replaced by an unrelated one.
    """
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import persona_column_patterns

    fields = {"full_name": "contact_name", "email": "contact_email"}
    patterns = persona_column_patterns(fixture_numbered_email_df, "email", "contact_email", fields, Config())

    assert patterns == ["{first}.{last}.###-####@{domain}"]


def test_a_column_of_several_name_conventions_keeps_each_of_them():
    """The column names every convention it writes, and none is imposed on a name following another."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import persona_column_patterns
    from nemo_safe_synthesizer.pii_replacer.replacement import persona_written

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
    persona = {"first_name": "Danielle", "last_name": "Figueroa"}

    patterns = persona_column_patterns(df, "full_name", "patient_name", {"full_name": "patient_name"}, Config())
    assert sorted(patterns) == sorted(
        ["{LAST}, {First}", "{First} {Last}", "{first} {last}", "{Last}, {F}.", "{F}. {Last}"]
    )

    replaced = [persona_written("full_name", value, persona, patterns) for value in df["patient_name"][:5]]
    assert replaced == ["FIGUEROA, Danielle", "Danielle Figueroa", "danielle figueroa", "Figueroa, D.", "D. Figueroa"]


def test_a_column_names_its_conventions_most_common_first():
    """Order is the whole of the tie-break, so the commonest convention has to come first."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import persona_column_patterns

    rows = []
    for i in range(50):
        first, last = FIRSTS[i % 10], LASTS[(i // 10) % 10]
        rows.append({"patient_name": f"{first} {last}" if i % 5 else f"{last.upper()}, {first}"})
    df = pd.DataFrame(rows)

    patterns = persona_column_patterns(df, "full_name", "patient_name", {"full_name": "patient_name"}, Config())
    assert patterns == ["{First} {Last}", "{LAST}, {First}"]


def test_an_address_column_with_no_convention_keeps_its_domains(fixture_contact_df: pd.DataFrame):
    """A column of handles says nothing about how it writes a person, but its domains are real."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import persona_column_patterns

    df = fixture_contact_df.copy()
    # Every handle a different length from the last, so no shape describes enough
    # of the column to be worth naming and each keeps the one it has.
    df["work_email"] = [f"{'u' * (1 + i % 30)}{i}@acme.com" for i in range(len(df))]
    fields = {"full_name": "patient_name", "email": "work_email"}

    assert persona_column_patterns(df, "email", "work_email", fields, Config()) == ["@{domain}"]


def test_a_column_of_handles_names_the_shape_it_writes_them_in():
    """A handle names nobody, so what the column has to say about it is its shape.

    Read as an identifier column is read, which keeps the run of letters the
    handles share instead of randomizing it away.
    """
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import persona_column_patterns
    from nemo_safe_synthesizer.pii_replacer.replacement import persona_written

    df = pd.DataFrame(
        {
            "patient_name": [f"{FIRSTS[i % 10]} {LASTS[(i // 10) % 10]}" for i in range(40)],
            "work_email": [f"usr{4700 + i}@acme.com" for i in range(40)],
        }
    )
    fields = {"full_name": "patient_name", "email": "work_email"}

    patterns = persona_column_patterns(df, "email", "work_email", fields, Config())
    assert patterns == ["usr47[0123]#@{domain}"]

    written = persona_written(
        "email",
        "usr4700@acme.com",
        {"first_name": "Danielle", "last_name": "Figueroa"},
        patterns,
        {"full_name": str(df.loc[0, "patient_name"])},
        Random(3),
    )
    assert written is not None
    assert re.fullmatch(r"usr47\d\d@acme\.com", written)
    assert written != "usr4700@acme.com"
    assert "figueroa" not in written  # a handle names nobody, and replacing it names nobody either


def test_a_handle_beside_a_person_stays_a_handle():
    """One column can hold both, and each row says which of the two it is."""
    from nemo_safe_synthesizer.pii_replacer.entities import Config
    from nemo_safe_synthesizer.pii_replacer.patterns import persona_column_patterns
    from nemo_safe_synthesizer.pii_replacer.replacement import persona_written

    rows = []
    for i in range(60):
        first, last = FIRSTS[i % 10], LASTS[(i // 10) % 10]
        addressed = f"{first.lower()}.{last.lower()}@acme.com" if i % 3 else f"usr{4700 + i}@acme.com"
        rows.append({"patient_name": f"{first} {last}", "work_email": addressed})
    df = pd.DataFrame(rows)
    fields = {"full_name": "patient_name", "email": "work_email"}
    persona = {"first_name": "Danielle", "last_name": "Figueroa"}

    patterns = persona_column_patterns(df, "email", "work_email", fields, Config())
    assert patterns == ["{first}.{last}@{domain}", "usr47##@{domain}"]

    def written(index: int) -> str | None:
        return persona_written(
            "email",
            str(df.loc[index, "work_email"]),
            persona,
            patterns,
            {"full_name": str(df.loc[index, "patient_name"])},
            Random(5),
        )

    assert written(1) == "danielle.figueroa@acme.com"
    handle = written(3)
    assert handle is not None
    assert re.fullmatch(r"usr47\d\d@acme\.com", handle)
