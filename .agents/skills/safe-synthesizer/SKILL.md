---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: safe-synthesizer
description: "Use NeMo Safe Synthesizer through task-specific routing: running the CLI or SDK, configuring parameters, troubleshooting runtime failures, inspecting artifacts, and interpreting evaluation outputs. Use when the user asks about safe-synthesizer, NeMo Safe Synthesizer, synthetic data pipeline runs, DP settings, generation failures, artifacts, logs, offline/GPU setup, config overrides, or evaluation metrics."
license: Apache-2.0
---

# Safe Synthesizer

Task router for agents helping a person use NeMo Safe Synthesizer. Read the task file that matches the user request, then read the docs it names before giving user-facing instructions.

## Route

- Run the pipeline, CLI, SDK, logging, or offline setup: read `run.md`.
- Set or override parameters: read `config.md`.
- Diagnose runtime, install, generation, OOM, or validation failures: read `diagnose.md`.
- Find outputs, logs, synthetic data, reports, or metrics: read `artifacts.md`.

## Answer Contract

- Start with the direct command, diagnosis, or file path.
- Cite relevant repo docs paths.
- Include one next action unless the user asks for a full walkthrough.
- Keep usage guidance separate from repo development internals.
- If the user asks to change CLI/config source code, read `docs/developer-guide/configuration_management.md`.
