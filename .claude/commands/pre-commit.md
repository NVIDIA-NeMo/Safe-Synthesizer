---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Run pre-commit hooks
---
Run all pre-commit hooks to lint and format files.

* Run with: `uv run pre-commit run -a`
* A clean run (no changes) means code is ready to commit
* Ty type check errors may need manual fixes
* Note: This is not a Makefile target; it's a utility command
