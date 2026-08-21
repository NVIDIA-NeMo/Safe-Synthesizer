# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placeholder configuration for the next PII replacement implementation."""

from __future__ import annotations

from ..configurator.parameters import Parameters

__all__ = ["ReplacePiiConfig"]


class ReplacePiiConfig(Parameters):
    """Mark PII replacement as requested.

    The replacement engine and its full configuration contract are intentionally
    absent on this branch. Set ``replace_pii`` to ``None`` to run the pipeline.
    """
