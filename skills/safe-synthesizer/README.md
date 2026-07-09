<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Safe Synthesizer Agent Skill

Agent Skill for helping users run, configure, troubleshoot, and inspect outputs
from NeMo Safe Synthesizer.

## Use from this repository

No install step is required when working from this source checkout. Agents that
support project skills can discover this skill from:

```text
skills/safe-synthesizer/
```

The `.agents/skills/safe-synthesizer` path is a repository symlink for
agents that discover project skills from `.agents/skills/`.

Ask a Safe Synthesizer usage question, or explicitly invoke the skill in an
agent that supports slash-style skill calls:

```text
/safe-synthesizer How do I run on a CSV with DP enabled?
```

## Copy into another project

From the target project root:

```bash
SAFE_SYNTHESIZER_REPO=/path/to/Safe-Synthesizer
mkdir -p .agents/skills
cp -R "$SAFE_SYNTHESIZER_REPO/skills/safe-synthesizer" .agents/skills/
```

For a user-level install:

```bash
SAFE_SYNTHESIZER_REPO=/path/to/Safe-Synthesizer
mkdir -p "$HOME/.agents/skills"
cp -R "$SAFE_SYNTHESIZER_REPO/skills/safe-synthesizer" "$HOME/.agents/skills/"
```

## Install from GitHub

Set `DEST_DIR` to the directory that should contain the `safe-synthesizer`
skill folder:

```bash
DEST_DIR="$HOME/.agents/skills"
TMP_DIR="$(mktemp -d)"
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/NVIDIA-NeMo/Safe-Synthesizer.git "$TMP_DIR"
git -C "$TMP_DIR" sparse-checkout set skills/safe-synthesizer
mkdir -p "$DEST_DIR"
cp -R "$TMP_DIR/skills/safe-synthesizer" "$DEST_DIR/"
rm -rf "$TMP_DIR"
```

For a project-local install, set `DEST_DIR=.agents/skills` from the target
project root.

The task references include links to the published Safe Synthesizer user guide,
so copied installs do not require the repository's local `docs/` directory.

## Publishable Package Notes

To publish this as an installable skill, package the
`skills/safe-synthesizer/` directory with:

- version metadata
- a README with install and usage examples
- a verification command that checks `SKILL.md`, `references/run.md`,
  `references/config.md`, `references/diagnose.md`, and
  `references/artifacts.md`
- stable documentation links for users outside this repository

An npm or `npx` installer would copy this directory into `.agents/skills/` for
a project install or `$HOME/.agents/skills/` for a user-level install.
