---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Build documentation site
---
Build the MkDocs Material documentation site.

* Run with: `mise run docs:build`
* Underlying command: `uv run --frozen --group docs mkdocs build`
* Output: `site/` directory
