# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a replacement plan is arrived at: discovered from a dataframe.

Validation, YAML IO, and ``resolve_plan`` land in follow-up PRs.
"""

from __future__ import annotations

from .discovery import discover_plan

__all__ = ["discover_plan"]
