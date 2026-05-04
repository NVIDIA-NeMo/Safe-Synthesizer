# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output modes shared across tooling renderers."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["RenderMode"]


class RenderMode(StrEnum):
    """Output target for tooling renderers.

    Adding a value implies a new backend and a matching case in every
    renderer that claims to support it, so keep this enum small.
    """

    RICH = "rich"
