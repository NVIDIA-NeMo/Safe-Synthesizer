---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Format code
---
Format code using ruff (formatting + import sorting + copyright headers).

* Run with: `mise run format`
* Underlying commands:
  * `bash tools/codestyle/format.sh`
  * `uv run --script tools/codestyle/copyright_fixer.py .`
* Always run this before committing
