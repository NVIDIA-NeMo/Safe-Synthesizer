# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package version information for NeMo Safe Synthesizer.

Version is determined at build time from git tags via uv-dynamic-versioning.
At runtime, the version is read from the installed package metadata.

To create a release version, tag a commit:
    git tag v0.2.0
    uv build --wheel
"""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version("nemo-safe-synthesizer")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for editable installs without metadata
