# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass


@dataclass
class Statistics:
    """Container for basic statistical measures."""

    count: int = 0
    mean: float = 0
    sum_sq_diffs: float = 0
    min: int | float = float("inf")
    max: int | float = float("-inf")

    @property
    def stddev(self) -> float:
        return 0 if self.count <= 1 else math.sqrt(self.sum_sq_diffs / (self.count - 1))


@dataclass
class RunningStatistics(Statistics):
    """Class to calculate the running mean and variance using Welford's method.

    This class allows for the calculation of statistics on-the-fly without the need
    for the entire dataset to be loaded into memory.
    """

    def update(self, x: int | float) -> None:
        """Update statistics with new value `x`."""
        self.count += 1
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        new_mean = self.mean + (x - self.mean) * 1.0 / self.count
        new_var = self.sum_sq_diffs + (x - self.mean) * (x - new_mean)
        self.mean, self.sum_sq_diffs = new_mean, new_var

    def reset(self) -> None:
        """Reset the running statistics."""
        self.count = 0
        self.mean = 0
        self.sum_sq_diffs = 0
        self.min = float("inf")
        self.max = float("-inf")
