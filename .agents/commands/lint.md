---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Check code quality
---
Run read-only local quality checks (formatting, lint rules, type checking, copyright).

* Run with: `mise run check`
* Runs the `check:*` leaves for formatting, lint rules, type checking, lock state, copyright headers, and Mise task validation
* Always run after `mise run format`
