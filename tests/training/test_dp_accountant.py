# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for SafeSynthesizerAccountant, covering the clamping
behaviour that prevents crashes when the HF Trainer runs more
optimizer steps than the PRV accountant was configured for.
"""

import pytest

from nemo_safe_synthesizer.privacy.dp_transformers.privacy_args import SafeSynthesizerAccountant

# Representative DP-training parameters (low num_steps keeps the test fast).
NOISE_MULTIPLIER = 1.0
SAMPLING_PROBABILITY = 0.01
DELTA = 1e-5
NUM_STEPS = 10


@pytest.fixture
def prv_accountant():
    return SafeSynthesizerAccountant(
        use_prv=True,
        noise_multiplier=NOISE_MULTIPLIER,
        sampling_probability=SAMPLING_PROBABILITY,
        delta=DELTA,
        num_steps=NUM_STEPS,
    )


class TestSafeSynthesizerAccountant:
    """Tests for SafeSynthesizerAccountant.compute_epsilon clamping."""

    def test_compute_epsilon_within_range(self, prv_accountant):
        """compute_epsilon works normally for steps within max_compositions."""
        eps = prv_accountant.compute_epsilon(steps=NUM_STEPS)
        assert isinstance(eps, float)
        assert eps > 0

    def test_compute_epsilon_at_max_compositions(self, prv_accountant):
        """compute_epsilon works at the exact max_compositions boundary."""
        max_comp = prv_accountant.max_compositions  # num_steps + 1
        eps = prv_accountant.compute_epsilon(steps=max_comp)
        assert isinstance(eps, float)
        assert eps > 0

    def test_compute_epsilon_beyond_max_compositions_does_not_raise(self, prv_accountant):
        """compute_epsilon clamps instead of raising when steps exceed max_compositions."""
        beyond = prv_accountant.max_compositions + 1
        eps = prv_accountant.compute_epsilon(steps=beyond)
        assert isinstance(eps, float)
        assert eps > 0

    def test_compute_epsilon_clamped_equals_max(self, prv_accountant):
        """Clamped result equals the result at max_compositions."""
        max_comp = prv_accountant.max_compositions
        eps_at_max = prv_accountant.compute_epsilon(steps=max_comp)
        eps_beyond = prv_accountant.compute_epsilon(steps=max_comp + 5)
        assert eps_at_max == eps_beyond

    def test_epsilon_increases_with_steps(self, prv_accountant):
        """Sanity check: more compositions means higher epsilon."""
        eps_low = prv_accountant.compute_epsilon(steps=1)
        eps_high = prv_accountant.compute_epsilon(steps=NUM_STEPS)
        assert eps_high > eps_low
