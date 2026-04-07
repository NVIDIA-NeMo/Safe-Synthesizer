# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import Field

from .base import NSSBaseModel
from .parameters import (
    SafeSynthesizerParameters,
)

__all__ = ["SafeSynthesizerJobConfig"]


class SafeSynthesizerJobConfig(NSSBaseModel):
    """Configuration model for Safe Synthesizer jobs.

    Used primarily internally to configure a run submitted to the NeMo Jobs
    Microservice.
    """

    data_source: str = Field(description="The data source for the job.")

    config: SafeSynthesizerParameters = Field(description="The Safe Synthesizer parameters configuration.")

    hf_token_secret: str | None = Field(
        default=None,
        description="Name of platform secret containing the HuggingFace token. "
        "Must exist in the same workspace as the job.",
    )
