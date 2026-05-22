---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Run GPU integration and e2e tests
---
Run GPU-dependent tests. Requires CUDA.

* GPU smoke tests: `mise run test:smoke:gpu`
* GPU integration tests: `mise run test:gpu-integration`
* All e2e tests: `mise run test:e2e`
* Default e2e only: `mise run test:e2e:default`
* DP e2e only: `mise run test:e2e:dp`
* Config-dataset combo: `mise run test-nss-tinyllama_nodp-clinc_oos-ci` (12 combos total, see `tests/TESTING.md`)
* Note: e2e tests run with `-n 0` (single process)
