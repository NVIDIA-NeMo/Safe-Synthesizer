# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants and enums for the generation pipeline."""

from __future__ import annotations

from enum import StrEnum


class GenerationStatus(StrEnum):
    """Status of a generation job.

    Attributes:
        COMPLETE: All target records were generated successfully.
        INCOMPLETE: Generation ended before reaching the target count.
        IN_PROGRESS: Generation is still running.
        STOP_NO_RECORDS: Stopped because a batch produced zero valid records.
        STOP_METRIC_REACHED: Stopped because the invalid-fraction threshold
            was exceeded for too many consecutive batches.
    """

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    IN_PROGRESS = "in_progress"
    STOP_NO_RECORDS = "stop_no_records"
    STOP_METRIC_REACHED = "stop_metric_reached"
