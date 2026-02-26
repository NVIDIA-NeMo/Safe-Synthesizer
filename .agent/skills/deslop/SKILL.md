---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: deslop
description: "Remove AI-generated code slop and clean up code style. Triggers on: review, clean up, deslop, code quality, remove slop, PR review."
related-skills: [python-observability]
---

# Remove AI Code Slop

Check the diff against main and remove AI-generated slop introduced in the branch.

## Focus Areas

- Narrating comments -- this repo's style is "comments explain why, not what"
- Do not use redundant docstrings that just restate the function signature
- Defensive `try`/`except Exception` on trusted internal paths
- using `# type: ignore` / `# ty: ignore` instead of trying to fix type errors
- `cast()` / `Any` to paper over type mismatches -- selective use only where truly needed
- using print statements instead of logging -- see `python-observability` skill
- modern typing syntax (`list`, `X | None`, `set` not `List`, `Optional`, `Set`)
- decorative `**bold**` in markdown and docstrings -- see `agent-markdown-style` rule

## Guardrails

- Keep behavior unchanged unless fixing a clear bug
- Prefer minimal, focused edits over broad rewrites
- Run `make format` and `make lint` after changes -- let tools fix style, not manual edits
- Keep the final summary concise (1-3 sentences)
