# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Date formats: what a column of birth dates names, and how each is read back."""

from __future__ import annotations

import re

import pandas as pd

from nemo_safe_synthesizer.config.replace_pii import (
    PersonaColumnSet,
    PiiColumnPlan,
    PiiEntity,
    PiiReplacementPlan,
)
from nemo_safe_synthesizer.pii_replacer.entities import Config


def _dob_replacement(patterns: list[str], dates: list[str]) -> pd.Series:
    from nemo_safe_synthesizer.pii_replacer.replacement import run_replacement

    df = pd.DataFrame({"date_of_birth": dates, "first_name": [f"First{i}" for i in range(len(dates))]})
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="primary_person",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                ],
            )
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="date_of_birth", entity_type=PiiEntity.date_of_birth, patterns=patterns),
        ],
    )
    runtime = Config(
        locale="en_US",
        random_seed=7,
        persona_backend="faker",
        sdg_pgms_src="/tmp",
        managed_assets_path=None,
    )
    out = run_replacement(df, plan, runtime).replaced_df
    return out["date_of_birth"]


def test_a_column_of_one_date_format_keeps_it():
    dates = [f"{(i % 12) + 1:02d}/{(i % 28) + 1:02d}/1980" for i in range(10)]
    result = _dob_replacement(["%m/%d/%Y"], dates)
    for original, new in zip(dates, result, strict=True):
        assert new != original
        assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(new))


def test_a_date_the_listed_formats_do_not_parse_keeps_its_own():
    dates = ["01/15/1980", "02/20/1990", "1975-03-25"]
    result = _dob_replacement(["%m/%d/%Y"], dates)
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(result.iloc[0]))
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(result.iloc[1]))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(result.iloc[2]))


def test_a_column_of_two_date_formats_parses_each_value_in_its_own():
    """Each format the column writes is listed, and each date is read in its own."""
    dates = ["01/15/1980", "1975-03-25", "02/20/1990", "1969-11-02"]
    result = _dob_replacement(["%m/%d/%Y", "%Y-%m-%d"], dates)
    for original, new in zip(dates, result, strict=True):
        assert new != original
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(result.iloc[0]))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(result.iloc[1]))
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(result.iloc[2]))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(result.iloc[3]))


def test_a_column_of_two_date_formats_names_both():
    from nemo_safe_synthesizer.pii_replacer.patterns import date_patterns

    values = pd.Series([f"0{i % 9 + 1}/15/1980" for i in range(30)] + [f"1975-0{i % 9 + 1}-25" for i in range(10)])

    assert date_patterns(values) == ["%m/%d/%Y", "%Y-%m-%d"]


def test_date_patterns_skips_unparseable_values():
    """Dot-separated dates are not in the matcher set; do not invent ``%Y-%m-%d``."""
    from nemo_safe_synthesizer.pii_replacer.patterns import date_patterns

    assert date_patterns(pd.Series(["15.03.2020", "16.03.2020", "17.03.2020"])) == []


def test_date_patterns_ranks_only_recognized_formats():
    """Mixed parseable + unparseable: count only formats that actually match."""
    from nemo_safe_synthesizer.pii_replacer.patterns import date_patterns

    values = pd.Series([f"0{i % 9 + 1}/15/1980" for i in range(20)] + ["15.03.2020", "16.03.2021", "not-a-date"])
    assert date_patterns(values) == ["%m/%d/%Y"]
