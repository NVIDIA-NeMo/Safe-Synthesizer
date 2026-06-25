---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: safe-synthesizer
description: "Use NeMo Safe Synthesizer through task-specific routing: running the CLI or SDK, configuring parameters, troubleshooting runtime failures, inspecting artifacts, and interpreting evaluation outputs. Use when the user asks about safe-synthesizer, NeMo Safe Synthesizer, synthetic data pipeline runs, DP settings, generation failures, artifacts, logs, offline/GPU setup, config overrides, or evaluation metrics."
license: Apache-2.0
metadata:
  author: Aaron Gonzales <aagonzales@nvidia.com>
---

# Safe Synthesizer

Task router for agents helping a person use NeMo Safe Synthesizer. Read the task file that matches the user request, then read the docs it names before giving user-facing instructions.

This package is exposed from the repository at `.agents/skills/safe-synthesizer`
as a symlink to `skills/safe-synthesizer`; reference paths below are relative to
this `SKILL.md`.

## Instructions

- Run the pipeline, CLI, SDK, logging, or offline setup: read `references/run.md`.
- Set or override parameters: read `references/config.md`.
- Diagnose runtime, install, generation, OOM, or validation failures: read `references/diagnose.md`.
- Find outputs, logs, synthetic data, reports, or metrics: read `references/artifacts.md`.

## Answer Contract

- Start with the direct command, diagnosis, or file path.
- Cite relevant repo docs paths.
- Include one next action unless the user asks for a full walkthrough.
- Keep usage guidance separate from repo development internals.
- If the user asks to change CLI/config source code, read `docs/developer-guide/configuration_management.md`.

## Examples

- A DP CLI run request routes to `references/run.md`.
- A nested parameter override request routes to `references/config.md`.
- An OOM, install, or validation failure routes to `references/diagnose.md`.
- An output, log, report, or metric lookup routes to `references/artifacts.md`.
