# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal result container used between pipeline stages (not public API)."""

from __future__ import annotations

import pandas as pd
from pydantic import Field

from .base import NSSBaseModel
from .external_results import SafeSynthesizerSummary

__all__ = ["SafeSynthesizerResults"]


class SafeSynthesizerResults(NSSBaseModel):
    """Full pipeline output including raw data and evaluation artifacts."""

    synthetic_data: pd.DataFrame | None = Field(
        default=None, description="Generated synthetic DataFrame, or ``None`` if generation was skipped."
    )

    summary: SafeSynthesizerSummary = Field(description="Quality, privacy, and timing metrics.")

    evaluation_report_html: str | None = Field(
        default=None, description="HTML evaluation report string, or ``None`` if evaluation was not run."
    )
