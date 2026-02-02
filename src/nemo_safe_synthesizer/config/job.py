# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from .base import NSSBaseModel
from .parameters import (
    SafeSynthesizerParameters,
)

__all__ = ["SafeSynthesizerJobConfig"]


class SafeSynthesizerJobConfig(NSSBaseModel):
    """
    Configuration model for Safe Synthesizer jobs. Used primarily to
    configure ourselves for a run to the NeMo Jobs Microservice.

    Attributes:
        data_source: The data source for the job.
        config: The Safe Synthesizer parameters configuration.
        hf_token_secret: Optional name of a platform secret containing the HuggingFace
            token. The secret should exist in the same workspace as the job. This is
            used to authenticate with HuggingFace Hub for downloading models.
    """

    data_source: str
    config: SafeSynthesizerParameters
    hf_token_secret: str | None = Field(
        default=None,
        description="Name of platform secret containing the HuggingFace token. "
        "Must exist in the same workspace as the job.",
    )
