# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..observability import get_logger

logger = get_logger()


class GenerationStopCondition:
    """Stopping conditions for the generation process.

    Empirically, the fraction of invalid records generated is a good indicator
    of the model's performance. This class implements a condition for stopping
    generation (and potentially training) based on the invalid fraction of
    records generated.

    Args:
        invalid_fraction_threshold: Stop generation if the invalid
            fraction exceeds this threshold for more than the number of
            consecutive batches specified by the `patience` parameter.
        patience: Number of consecutive batches to wait before stopping.
    """

    def __init__(self, invalid_fraction_threshold: float, patience: int):
        self.counter = 0
        self.patience = patience
        self.invalid_fraction_threshold = invalid_fraction_threshold
        self.last_value = None

    def has_been_reached(self, invalid_fraction: float) -> bool:
        """Returns True if the stopping condition has been reached."""
        is_reached = False
        self.last_value = invalid_fraction
        if invalid_fraction >= self.invalid_fraction_threshold:
            self.counter += 1
            if self.counter >= self.patience:
                is_reached = True
                logger.info(
                    f"🛑 Stopping condition reached: {invalid_fraction = :.2} > "
                    f"{self.invalid_fraction_threshold} for {self.counter} "
                    "consecutive batches.",
                )
        else:
            self.counter = 0
        return is_reached
