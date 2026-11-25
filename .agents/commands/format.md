---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Format code
---

Format code: TOML, Python (formatting, lint fixes, import sorting), Markdown, and copyright headers.

- Run with: `mise run format`
- Underlying commands:
  - `dprint fmt`
  - `ruff check --fix` and `ruff format`
  - `rumdl check --fix .`
  - `uv run --script tools/codestyle/copyright_fixer.py .`
- Always run this before committing
