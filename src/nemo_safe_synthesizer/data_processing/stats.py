# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight running statistics using Welford's online algorithm."""

import math
from dataclasses import dataclass


@dataclass
class Statistics:
    """Container for basic statistical measures.

    Tracks count, mean, min, max, and the sum of squared differences
    (for computing sample standard deviation). Intended as a base for
    ``RunningStatistics``; use that subclass to incrementally accumulate values.
    """

    count: int = 0
    """Number of values observed."""

    mean: float = 0
    """Running mean of observed values."""

    sum_sq_diffs: float = 0
    """Sum of squared differences from the mean (Welford's M2)."""

    min: int | float = float("inf")
    """Minimum observed value."""

    max: int | float = float("-inf")
    """Maximum observed value."""

    @property
    def stddev(self) -> float:
        """Sample standard deviation, or 0 when fewer than two values have been observed."""
        return 0 if self.count <= 1 else math.sqrt(self.sum_sq_diffs / (self.count - 1))


@dataclass
class RunningStatistics(Statistics):
    """Incrementally compute mean and variance using Welford's online algorithm.

    Allows statistics to be calculated on-the-fly without loading the entire
    dataset into memory. Call ``update`` for each new observation and read
    ``mean``, ``stddev``, ``min``, ``max`` at any time.
    """

    def update(self, x: int | float) -> None:
        """Incorporate a new value ``x`` into the running statistics."""
        self.count += 1
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        new_mean = self.mean + (x - self.mean) * 1.0 / self.count
        new_var = self.sum_sq_diffs + (x - self.mean) * (x - new_mean)
        self.mean, self.sum_sq_diffs = new_mean, new_var

    def reset(self) -> None:
        """Reset all statistics to their initial state."""
        self.count = 0
        self.mean = 0
        self.sum_sq_diffs = 0
        self.min = float("inf")
        self.max = float("-inf")
