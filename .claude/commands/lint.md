---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Check code quality
---
Run all read-only CI checks (formatting, lint rules, type checking, copyright).

* Run with: `make check`
* Runs `make format-check` (ruff format + ruff check + copyright) + `make typecheck` (ty)
* Always run after `make format`
