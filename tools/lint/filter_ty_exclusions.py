#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filter files against ty exclusions from pyproject.toml."""

import fnmatch
import sys
import tomllib

with open("pyproject.toml", "rb") as f:
    config = tomllib.load(f)

excludes = config.get("tool", {}).get("ty", {}).get("src", {}).get("exclude", [])
# Normalize patterns (remove leading ./)
excludes = [p.lstrip("./") for p in excludes]

for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    if not any(
        path == pat
        or path.startswith(pat.rstrip("/") + "/")
        or fnmatch.fnmatch(path, pat)
        or fnmatch.fnmatch(path, "**/" + pat)
        for pat in excludes
    ):
        print(path)
