---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Build Python wheel package
---
Build the wheel package. Version comes from git tags via uv-dynamic-versioning.

* Run with: `make build-wheel`
* Underlying commands:
  ```bash
  rm -rf dist/
  uv build --wheel
  ```
* Output: `dist/*.whl`
