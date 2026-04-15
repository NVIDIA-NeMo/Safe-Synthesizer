# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``data_processing.budget`` token-budget arithmetic."""

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.data_processing.budget import compute_max_new_tokens


@pytest.mark.unit
def test_compute_max_new_tokens_subtracts_schema_and_special_tokens():
    """Budget = context - schema - 2 * NUM_SPECIAL_TOKENS.

    With NUM_SPECIAL_TOKENS=2: 2048 - 100 - 4 = 1944.
    """
    assert compute_max_new_tokens(list(range(100)), 2048) == 1944


@pytest.mark.unit
def test_compute_max_new_tokens_negative_when_schema_exceeds_context():
    """A schema larger than the context window produces a negative budget."""
    assert compute_max_new_tokens(list(range(2050)), 2048) < 0
