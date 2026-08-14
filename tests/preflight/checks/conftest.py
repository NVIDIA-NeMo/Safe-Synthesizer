# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for ``preflight.checks`` tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def fixture_dob_df() -> pd.DataFrame:
    """Birth dates in a dominant %m/%d/%Y format plus one ISO minority row."""
    return pd.DataFrame(
        {
            "patient_id": ["A", "B", "C"],
            "first_name": ["Alice", "Bob", "Cleo"],
            "sex": ["Female", "Male", "Female"],
            "date_of_birth": ["01/15/1980", "02/20/1990", "1975-03-25"],
            "notes": ["visit", "follow-up", "discharge"],
        }
    )
