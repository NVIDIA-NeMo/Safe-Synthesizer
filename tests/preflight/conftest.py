# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters


@pytest.fixture
def default_config():
    """Default SafeSynthesizerParameters with all defaults."""
    return SafeSynthesizerParameters()


@pytest.fixture
def sample_df():
    """Simple 5-column, 500-row DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "id": range(500),
        "name": [f"item_{i}" for i in range(500)],
        "value": rng.random(500),
        "category": rng.choice(["A", "B", "C", "D"], 500),
        "score": rng.integers(0, 100, 500),
    })


@pytest.fixture
def small_df():
    """50-row DataFrame for size-check testing."""
    return pd.DataFrame({
        "col_a": range(50),
        "col_b": ["x"] * 50,
    })


@pytest.fixture
def tiny_df():
    """5-row DataFrame for extreme-small testing."""
    return pd.DataFrame({
        "col_a": range(5),
        "col_b": ["x"] * 5,
    })
