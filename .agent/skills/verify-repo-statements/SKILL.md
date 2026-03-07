---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: verify-repo-statements
description: "Verify statements about the repo against source, config, tests, or tooling. Triggers on: verify, claim, statement, true about repo, docs vs code, check against source, verify docs, doc verification."
---

# Verifying Statements About the Repo

Reusable pattern for confirming or refuting any assertion about the repo: from docs, plans, PR descriptions, issues, or chat. Applies to config defaults, CLI behavior, error messages, module layout, dependencies, CI job behavior, "this function does Y," "tests cover Z." Doc verification (user guide vs source) is one use case; the same pattern works for architecture claims, API contracts, and behavioral claims.

## 1. Identify the statement or claim

Source may be: user-facing docs, a plan file, PR body, issue description, or the user's message. Extract concrete claims (e.g. "default learning rate is 0.0005," "DataError is raised when NaNs are present," "format job runs `make format-check`").

## 2. Map to evidence in the repo

Where to look:

| Claim type | Evidence location |
|------------|-------------------|
| Config defaults, fields | `src/nemo_safe_synthesizer/config/`, Pydantic models |
| CLI behavior, flags | `src/nemo_safe_synthesizer/cli/`, Click definitions |
| Error messages, hierarchy | `src/nemo_safe_synthesizer/errors.py` |
| Module layout, imports | `src/nemo_safe_synthesizer/`, `AGENTS.md` module map |
| CI jobs, make targets | `.github/workflows/`, `Makefile`, CONTRIBUTING.md |
| Test coverage, markers | `tests/`, `pytest.ini` |

Use read, grep, or ast-nav to locate the defining source.

## 3. Verify and record

- Run the relevant command or read the relevant file; compare to the claim.
- Record pass (matches), fail (contradiction), or discrepancy (e.g. wording differs, default changed).
- For doc verification at scale: extract a list of claims, then verify each and report in a table or sectioned report.

When verifying many claims (e.g. user guide), batching by domain (config, CLI, errors, output layout) keeps the workflow manageable. See user-guide verification sessions for an example of the full flow.
