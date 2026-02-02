# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Result objects used internally, not part of any external public API
import pandas as pd

from .base import NSSBaseModel
from .external_results import SafeSynthesizerSummary

__all__ = ["SafeSynthesizerResults"]


class SafeSynthesizerResults(NSSBaseModel):
    """
    Output object for Safe Synthesizer
    """

    synthetic_data: pd.DataFrame | None = None
    summary: SafeSynthesizerSummary
    evaluation_report_html: str | None
