---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Run CI tests in a Linux container
---
Build and run CI unit tests in a container. Useful for macOS developers or reproducing CI failures.

* Run with: `make test-ci-container`
* This builds a container from `containers/Dockerfile.test_ci` then runs `make test-ci` inside it
* Container runtime: auto-detects podman or docker
