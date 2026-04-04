# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0 AND MIT

# This file has been adapted from the `dp-transformers` library.
# Original source: https://github.com/microsoft/dp-transformers/blob/main/src/dp_transformers/arguments.py
# See THIRD_PARTY.md for the original MIT license terms.

"""Privacy arguments and noise multiplier computation for DP training.

Provides ``PrivacyArguments`` (target epsilon/delta, noise multiplier, clipping norm),
``SafeSynthesizerAccountant`` for epsilon accounting (PRV or RDP), and
``prv_find_noise_multiplier()`` to solve for noise scale given a target epsilon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier as opacus_get_noise_multiplier
from prv_accountant import Accountant as PRVAccountant
from scipy import optimize

from ...observability import get_logger

logger = get_logger()


@dataclass
class SafeSynthesizerAccountant:
    """Privacy accountant for computing epsilon from training steps.

    Wraps either PRV (privacy random variable) or Opacus RDP accountant.
    Use PRV when possible; RDP is used as fallback if PRV fails to converge.

    Args:
        use_prv: If True, use PRV accountant; otherwise RDP.
        noise_multiplier: Scale of Gaussian noise added to gradients.
        sampling_probability: Probability of a record being in a batch.
        delta: Target delta for (epsilon, delta)-DP.
        num_steps: Maximum number of composition steps (+1 for headroom)
            (i.e. "would one more step exceed the budget?").
    """

    accountant: RDPAccountant | PRVAccountant = field(init=False)

    def __init__(self, use_prv: bool, noise_multiplier, sampling_probability, delta, num_steps):
        # +1 provides headroom for a forward-looking epsilon check
        # (i.e. "would one more step exceed the budget?").
        self.max_compositions = num_steps + 1
        if use_prv:
            self.accountant = PRVAccountant(
                noise_multiplier=noise_multiplier,
                sampling_probability=sampling_probability,
                delta=delta,
                max_compositions=self.max_compositions,
                eps_error=0.01,
            )
        else:
            self.accountant = RDPAccountant()
        self.use_prv = use_prv
        self.delta = delta

    def compute_epsilon(self, steps: int) -> float:
        """Compute epsilon consumed after the given number of steps.

        Args:
            steps: Number of optimizer steps taken.

        Returns:
            The current epsilon value for the configured delta.
        """
        if self.use_prv:
            # Cap to max_compositions so callers never exceed the
            # accountant's pre-computed range.  This can happen when the
            # HF Trainer runs an extra optimizer step for an incomplete
            # gradient-accumulation batch at the end of an epoch.
            steps = min(steps, self.max_compositions)
            acct = cast(PRVAccountant, self.accountant)
            return acct.compute_epsilon(steps)[2]
        else:
            acct = cast(RDPAccountant, self.accountant)
            return acct.get_epsilon(self.delta)


@dataclass
class PrivacyArguments:
    """Store for DP training parameters (epsilon, delta, noise, clipping).

    Exactly one of ``target_epsilon`` or ``noise_multiplier`` must be set.
    If ``target_epsilon`` is set, ``initialize()`` must be called to solve
    for the noise multiplier before training.

    Args:
        target_epsilon: Target epsilon at end of training (mutually exclusive with noise multiplier).
        target_delta: Target delta.
        per_sample_max_grad_norm: Max L2 norm for per-sample gradient clipping.
        noise_multiplier: Gaussian noise scale for gradients. Mutually exclusive with target_epsilon.
        poisson_sampling: Enable Poisson sampling for proper DP accounting.
        use_prv: If True, use PRV accountant; fallback to RDP if PRV fails to converge.
    """

    target_epsilon: float | None = field(
        default=None,
        metadata={"help": "Target epsilon at end of training (mutually exclusive with noise multiplier)"},
    )
    target_delta: float | Literal["auto"] | None = field(default=None, metadata={"help": "Target delta"})
    per_sample_max_grad_norm: float | None = field(
        default=None, metadata={"help": "Max L2 norm for per-sample gradient clipping."}
    )

    noise_multiplier: float | None = field(
        default=None, metadata={"help": "Gaussian noise scale for gradients. Mutually exclusive with target_epsilon."}
    )

    poisson_sampling: bool = field(
        default=False,
        metadata={"help": "Enable Poisson sampling for proper DP accounting"},
    )
    use_prv: bool = field(
        default=True,
        metadata={
            "help": "Flag indicating whether PRV accountant was used. "
            "Due to numerical instability issues, sometimes "
            "PRV accountant fails to converge on a value for noise "
            "multiplier, requiring a switch to RDPAccountant."
        },
    )

    def initialize(self, sampling_probability: float, num_steps: int) -> None:
        """Solve for noise multiplier from target epsilon, or confirm existing multiplier.

        Called before training when ``target_epsilon`` is set. Uses PRV accountant
        first; on convergence failure, falls back to Opacus RDP and sets
        ``use_prv`` to False.

        Args:
            sampling_probability: Probability of a record being in a batch.
            num_steps: Expected number of optimization steps.
        """
        if self.noise_multiplier is None:
            target_eps = self.target_epsilon
            target_del = self.target_delta
            if target_eps is None or target_del is None or target_del == "auto":
                raise ValueError("target_epsilon and target_delta (numeric) are required for DP initialization")
            try:
                self.noise_multiplier = prv_find_noise_multiplier(
                    sampling_probability=sampling_probability,
                    num_steps=num_steps,
                    target_delta=target_del,
                    target_epsilon=target_eps,
                )
            except Exception as known_error:
                if "Discrete mean differs" in str(known_error):
                    logger.warning(
                        f"DP setup failed due to PRV accountant({known_error}), trying Opacus RDP Accountant"
                    )

                try:
                    self.noise_multiplier = opacus_get_noise_multiplier(
                        target_epsilon=target_eps,
                        target_delta=target_del,
                        sample_rate=sampling_probability,
                        steps=num_steps,
                        accountant="rdp",
                    )
                    self.use_prv = False
                except Exception as other_error:
                    raise other_error

        logger.info(
            f"The noise multiplier is set to: {self.noise_multiplier}",
        )

    @property
    def is_initialized(self) -> bool:
        """True if per_sample_max_grad_norm, noise_multiplier, and target_delta are set."""
        return (
            self.per_sample_max_grad_norm is not None
            and self.noise_multiplier is not None
            and self.target_delta is not None
        )

    def __post_init__(self):
        if (self.target_epsilon is None) == (self.noise_multiplier is None):
            raise ValueError("Exactly one of target_epsilon and noise_multiplier must be specified.")
        if self.per_sample_max_grad_norm is None:
            raise ValueError("DP training requires per_sample_max_grad_norm to be specified.")


def prv_find_noise_multiplier(
    sampling_probability: float,
    num_steps: int,
    target_epsilon: float,
    target_delta: float,
    eps_error: float = 0.05,
) -> float:
    """Find a noise multiplier that satisfies a given target epsilon.

    Uses binary search with PRV accountant to solve for the noise scale.
    Adapted from https://github.com/microsoft/prv_accountant/blob/main/prv_accountant/dpsgd.py#L39

    Args:
        sampling_probability: Probability of a record being in a batch.
        num_steps: Number of optimization steps.
        target_epsilon: Desired epsilon at end of training.
        target_delta: Delta for (epsilon, delta)-DP.
        eps_error: Allowed error for the final epsilon value.

    Returns:
        Noise multiplier achieving approximately target_epsilon.

    Raises:
        RuntimeError: If no valid noise multiplier found (e.g. epsilon too low
            or too few records), or if epsilon cannot be computed within eps_error.
    """

    def compute_epsilon(noise_multiplier: float) -> tuple[float, float, float]:
        """Initialize a privacy accountant and compute epsilon bounds for a given noise multiplier after ``num_steps``.

        Uses the PRV (privacy loss random variables) accountant to compose
        privacy guarantees and compute epsilon (https://arxiv.org/abs/2106.02848).

        Args:
            noise_multiplier: Gaussian noise scale (sigma) to evaluate.

        Returns:
            Tuple of three floats: ``(epsilon_lower, epsilon_point, epsilon_upper)``.
            The outer search uses ``[0]`` and ``[2]`` when comparing to ``target_epsilon``.
        """
        # TODO: +1 was added in max_compositions due to a bug in NSS-DP, this requires a fix.
        acc = PRVAccountant(
            noise_multiplier=noise_multiplier,
            sampling_probability=sampling_probability,
            delta=target_delta,
            max_compositions=num_steps + 1,
            eps_error=eps_error / 4,
        )
        return acc.compute_epsilon(num_steps)

    mu_max = 100.0

    mu_R = 1.0
    eps_R = float("inf")
    while eps_R > target_epsilon:
        mu_R *= np.sqrt(2)
        try:
            eps_R = compute_epsilon(mu_R)[2]
        except (OverflowError, RuntimeError):
            pass
        if mu_R > mu_max:
            raise RuntimeError(
                "Unable to automatically determine a noise multiplier for DP optimizer. "
                "Increase epsilon or number of training records/entities."
            )

    mu_L = mu_R
    eps_L = eps_R
    while eps_L < target_epsilon:
        mu_L /= np.sqrt(2)
        eps_L = compute_epsilon(mu_L)[0]

    has_converged = False
    bracket = [mu_L, mu_R]
    while not has_converged:
        mu_err = (bracket[1] - bracket[0]) * 0.01
        mu_guess = optimize.root_scalar(
            lambda mu: compute_epsilon(mu)[2] - target_epsilon,
            bracket=bracket,
            xtol=mu_err,
        ).root
        bracket = [mu_guess - mu_err, mu_guess + mu_err]
        eps_up = compute_epsilon(bracket[0])[2]
        eps_low = compute_epsilon(bracket[1])[0]
        has_converged = (eps_up - eps_low) < 2 * eps_error
    if compute_epsilon(bracket[1])[2] >= target_epsilon + eps_error:
        raise RuntimeError("Error in computing noise multiplier. Unable to compute epsilon within allowed error.")

    return bracket[1]
