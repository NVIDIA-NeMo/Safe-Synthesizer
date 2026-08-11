# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal stubs for the optional internal sdg-pgms USPersonGenerator."""

from __future__ import annotations

from typing import Any

import pandas as pd

class USPersonGenerator:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def generate_samples(self, n: int) -> pd.DataFrame: ...
