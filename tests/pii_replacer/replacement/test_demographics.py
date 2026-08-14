# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demographics vocabulary mapping used to condition persona sampling."""

from __future__ import annotations

from nemo_safe_synthesizer.pii_replacer.replacement.demographics import ethnicity_to_pgm, fuzzy_category, norm_sex


def test_not_hispanic_or_latino_does_not_map_to_hispanic():
    """Census-style negation must leave ethnicity unconstrained, not inverted."""
    assert ethnicity_to_pgm("Not Hispanic or Latino") is None
    assert ethnicity_to_pgm("non hispanic") is None


def test_non_hispanic_white_still_maps_to_white():
    """``Non-Hispanic`` negates ethnicity only; race ``White`` must still match."""
    assert ethnicity_to_pgm("Non-Hispanic White") == ["white"]
    assert ethnicity_to_pgm("non hispanic white") == ["white"]


def test_non_white_and_not_black_do_not_map_to_negated_category():
    assert ethnicity_to_pgm("non-white") is None
    assert ethnicity_to_pgm("not black") is None


def test_positive_ethnicity_values_still_map():
    assert ethnicity_to_pgm("Hispanic or Latino") is not None
    assert ethnicity_to_pgm("White") == ["white"]
    assert ethnicity_to_pgm("Mexican") == ["mexican"]


def test_norm_sex_unaffected_by_ethnicity_negation_guard():
    assert norm_sex("Female") == "Female"
    assert fuzzy_category("Male", {"Male": ["m"], "Female": ["f"]}) == "Male"
