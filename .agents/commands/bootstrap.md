---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Bootstrap development environment
---

Set up the development environment from scratch.

1. Install mise and the development tools, sync every dependency group and
   extra (`deps:sync all`), and install the Git hooks:

   ```bash
   make setup
   ```

2. To switch to another dependency profile, sync it. `uv sync` is exact, so a
   profile removes what it does not name:

   ```bash
   mise run deps:sync all    # Every group and extra: CUDA 13.0 on Linux, CPU on macOS
   mise run deps:sync cpu    # CPU-only PyTorch (macOS or Linux without GPU)
   mise run deps:sync cu129  # CUDA 12.9 (Linux with NVIDIA GPU)
   mise run deps:sync cu130  # CUDA 13.0 (Linux with NVIDIA GPU)
   mise run deps:sync engine # Engine dependencies only (no torch)
   mise run deps:sync dev    # The default dependency groups only
   ```

The `cpu`, `cu129`, and `cu130` extras conflict; a profile installs at most one.
