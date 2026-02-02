# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum


class GenerationStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    IN_PROGRESS = "in_progress"
    STOP_NO_RECORDS = "stop_no_records"
    STOP_METRIC_REACHED = "stop_metric_reached"
