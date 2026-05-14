# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight configuration knobs."""

from __future__ import annotations

from pydantic import Field

from ..configurator.parameters import Parameters

__all__ = ["PreflightParameters"]


class PreflightParameters(Parameters):
    """User-controllable overrides for the preflight validation layer."""

    disabled_checks: list[str] = Field(
        default_factory=list,
        description=(
            "Names of preflight checks to skip at runtime (e.g. ``gpu.vram``). "
            "Disabled checks are dropped from the report entirely."
        ),
    )
