---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Run all unit tests including slow
---
Run all unit tests including slow tests (excludes e2e and smoke tests).

* Run with: `make test-unit-slow`
* Underlying command: `uv run --frozen pytest -n auto --dist loadscope -vv /root/dev/Safe-Synthesizer/tests -m "unit"`
