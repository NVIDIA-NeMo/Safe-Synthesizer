---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
description: Sync with NMP repository
---
Synchronize code between Safe-Synthesizer (GitHub) and NMP (GitLab).

* Sync from a specific NMP MR:
  ```bash
  make synchronize-from-nmp-mr MR=<number>
  ```

* Full sync (requires `NMP_REPO_PATH` env var):
  ```bash
  make synchronize-from-nmp     # NMP -> Safe-Synthesizer
  make synchronize-to-nmp       # Safe-Synthesizer -> NMP
  ```

* Sync Python files only:
  ```bash
  make synchronize-py-files-from-nmp
  make synchronize-py-files-to-nmp
  ```
