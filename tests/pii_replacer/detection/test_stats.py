# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column statistics: cardinality denominators and structural grain."""

from __future__ import annotations

import pandas as pd


def test_sparse_email_column_unique_ratio_not_null_diluted():
    from nemo_safe_synthesizer.pii_replacer.detection import column_stats

    df = pd.DataFrame({"emailish": [None] * 90 + [f"user{i}@example.com" for i in range(10)]})
    stats = column_stats(df)["emailish"]
    # Denominator is non-null rows (10), not full length (100), so nulls do not dilute.
    assert stats["unique_ratio"] == 1.0
    assert stats["n_unique"] == 10


def test_columns_are_tagged_by_grain_not_replacement_scope(fixture_group_grain_df: pd.DataFrame):
    """Grain says what varies within a group; the plan's scope is a separate idea."""
    from nemo_safe_synthesizer.pii_replacer.detection import scoped_column_stats

    stats = scoped_column_stats(fixture_group_grain_df, "patient_id", 0.95)
    assert stats["patient_id"]["grain"] == "key"
    assert stats["full_name"]["grain"] == "group"
    assert stats["email"]["grain"] == "record"
    assert "scope" not in stats["full_name"]
