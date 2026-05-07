<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inspect Safe Synthesizer Artifacts

Use for output directories, logs, synthetic data, reports, metrics, and generated datasets.

Read first:

- `docs/user-guide/running.md` for output layout and logging.
- `docs/user-guide/evaluating-data.md` for evaluation reports and metrics.
- `docs/user-guide/environment.md` for artifact path environment variables.

Look for:

- `synthetic_data.csv` for generated data.
- `evaluation_report.html` for the HTML report.
- `evaluation_metrics.json` for machine-readable metrics.
- `logs/` for run logs.
- `train/adapter/` for trained PEFT adapters.

Response shape:

- Artifact root and run path pattern.
- File names relevant to the user's question.
- Docs path that supports the layout.
