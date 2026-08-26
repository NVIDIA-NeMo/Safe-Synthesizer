<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII Replacement

On this branch, enabling PII replacement runs heuristic auto-discovery (or
loads a user plan) and writes `pii_replacement_plan.yaml` under the run
directory. Column values are left unchanged — replacement execution lands in a
later update.

Disable with `replace_pii: null`, `--no-replace-pii`, or
`.with_replace_pii(enable=False)`.
