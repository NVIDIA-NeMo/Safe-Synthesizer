---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Lint and typecheck code
---
Run linting, type checking, and copyright verification.

* Run with: `make lint`
* Runs three steps:
  1. `bash tools/codestyle/lint.sh` -- ruff linting
  2. `bash tools/codestyle/typecheck.sh` -- ty type checking
  3. `python tools/codestyle/copyright_fixer.py --check .` -- copyright headers
* Always run after `make format`
