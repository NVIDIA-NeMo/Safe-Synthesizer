# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Date formats: what a column of birth dates names (singular dominant)."""

from __future__ import annotations

import pandas as pd


def test_a_column_of_one_date_format_keeps_it():
    from nemo_safe_synthesizer.pii_replacer.patterns import date_pattern

    dates = pd.Series([f"{(i % 12) + 1:02d}/{(i % 28) + 1:02d}/1980" for i in range(40)])
    assert date_pattern(dates) == "%m/%d/%Y"


def test_a_column_of_two_date_formats_omits_when_neither_dominates():
    """Mixed formats under 85% → omit plan pattern."""
    from nemo_safe_synthesizer.pii_replacer.patterns import date_pattern, date_patterns

    values = pd.Series([f"0{i % 9 + 1}/15/1980" for i in range(30)] + [f"1975-0{i % 9 + 1}-25" for i in range(10)])
    assert date_pattern(values) is None
    # date_patterns returns at most one dominant; none here.
    assert date_patterns(values) == []


def test_date_patterns_skips_unparseable_values():
    from nemo_safe_synthesizer.pii_replacer.patterns import date_patterns

    assert date_patterns(pd.Series(["15.03.2020", "16.03.2020", "17.03.2020"])) == []


def test_date_patterns_ranks_only_recognized_formats():
    from nemo_safe_synthesizer.pii_replacer.patterns import date_pattern

    values = pd.Series([f"0{i % 9 + 1}/15/1980" for i in range(40)] + ["15.03.2020", "16.03.2021", "not-a-date"])
    assert date_pattern(values) == "%m/%d/%Y"
