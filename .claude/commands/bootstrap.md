---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Bootstrap development environment
---
Set up the development environment from scratch.

1. Install development tools (uv, ruff, ty, yq, etc.):

   ```bash
   make setup
   ```

2. Install Python dependencies (choose one):

   ```bash
   mise run bootstrap-nss cpu    # CPU-only (macOS or Linux without GPU)
   mise run bootstrap-nss cuda   # CUDA 12.9 (Linux with NVIDIA GPU)
   mise run bootstrap-nss engine # Engine dependencies only (no torch)
   mise run bootstrap-nss dev    # Minimal dev dependencies only
   ```

Note: `cuda` is an alias for `cu129`. Both are equivalent.
