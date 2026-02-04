# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0 AND MIT

# This file has been adapted from the `dp-transformers` library.
# Original source: https://github.com/microsoft/dp-transformers/blob/main/src/dp_transformers/arguments.py
# See THIRD_PARTY.md for the original MIT license terms.


from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier as opacus_get_noise_multiplier
from prv_accountant import Accountant as PRVAccountant
from scipy import optimize

from ...observability import get_logger

logger = get_logger()


@dataclass
class SafeSynthesizerAccountant:
    def __init__(self, use_prv: bool, noise_multiplier, sampling_probability, delta, num_steps):
        if use_prv:
            # TODO: +1 was added in max_compositions due to a bug in NavFT, we should try to fix it
            self.accountant = PRVAccountant(
                noise_multiplier=noise_multiplier,
                sampling_probability=sampling_probability,
                delta=delta,
                max_compositions=num_steps + 1,
                eps_error=0.01,
            )
        else:
            self.accountant = RDPAccountant()
        self.use_prv = use_prv
        self.delta = delta

    def compute_epsilon(self, steps):
        if self.use_prv:
            return self.accountant.compute_epsilon(steps)[2]
        else:
            return self.accountant.get_epsilon(self.delta)


@dataclass
class PrivacyArguments:
    target_epsilon: Optional[float] = field(
        default=None,
        metadata={"help": "Target epsilon at end of training (mutually exclusive with noise multiplier)"},
    )
    target_delta: float | Literal["auto"] | None = field(default=None, metadata={"help": "Target delta"})
    per_sample_max_grad_norm: Optional[float] = field(default=None, metadata={"help": "Max per sample clip norm"})
    noise_multiplier: Optional[float] = field(default=None, metadata={"help": "Noise multiplier for DP training"})
    poisson_sampling: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable Poisson sampling for proper DP accounting"},
    )
    use_prv: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Flag indicating whether PRV accountant was used. "
            "Due to numerical instability issues, sometimes "
            "PRV accountant fails to converge on a value for noise "
            "multiplier, requiring a switch to RDPAccountant."
        },
    )

    def initialize(self, sampling_probability: float, num_steps: int) -> None:
        if self.noise_multiplier is None:
            try:
                self.noise_multiplier = prv_find_noise_multiplier(
                    sampling_probability=sampling_probability,
                    num_steps=num_steps,
                    target_delta=self.target_delta,
                    target_epsilon=self.target_epsilon,
                )
            except Exception as known_error:
                if "Discrete mean differs" in str(known_error):
                    logger.warning(
                        f"DP setup failed due to PRV accountant({known_error}), trying Opacus RDP Accountant"
                    )

                try:
                    self.noise_multiplier = opacus_get_noise_multiplier(
                        target_epsilon=self.target_epsilon,
                        target_delta=self.target_delta,
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
    """
    Find a noise multiplier that satisfies a given target epsilon.

    :param float sampling_probability: Probability of a record being in batch
    :param int num_steps: Number of optimization steps
    :param float target_epsilon: Desired target epsilon
    :param float target_delta: Value of DP delta
    :param float eps_error: Error allowed for final epsilon

    This function has been adapted from
    https://github.com/microsoft/prv_accountant/blob/main/prv_accountant/dpsgd.py#L39
    """

    def compute_epsilon(noise_multiplier: float) -> tuple[float, float, float]:
        """
        Initialize a privacy accountant and compute epsilon for a given noise
        multiplier. This privacy accountant uses a fast algorithm to optimally
        compose privacy guarantees and based on the notion of privacy loss
        random variables to quantify the privacy loss of DP algorithms.
        Based on https://arxiv.org/abs/2106.02848
        """
        # TODO: +1 was added in max_compositions due to a bug in NavFT, we should try to fix it
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
