# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Skip all tests in this directory if transformers is not available.
pytest.importorskip(
    "transformers", reason="transformers is required for these tests (install with: uv sync --extra cpu)"
)
