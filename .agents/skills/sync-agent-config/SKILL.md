# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
name: sync-agent-config
description: "Keep agent config in sync with source-of-truth files. Triggers on: new Makefile target, new module, docs structure change, pytest markers change, new skill, agent config stale."
related-skills: [diagnose-failures]
license: Apache-2.0
---

# Sync Agent Config

Advisory skill for keeping agent configuration files in sync when source-of-truth files change.

## When to Use

- "I added a new Makefile target"
- "I created a new source module"
- "I changed the docs/ structure or mkdocs.yml"
- "I modified pytest.ini markers"
- "I added a new skill"

## Change-to-Update Mapping

| When this changes... | Update these files... |
|---------------------|----------------------|
| Makefile (new/renamed/removed targets) | `.claude/commands/` (add/rename/remove command file), `.cursor/rules/claude-commands.mdc` (update index table), `AGENTS.md` (workflow section if lifecycle changes). If the target becomes a CI job, also update `.agents/skills/diagnose-failures/SKILL.md` (CI job table) |
| Source modules (`src/nemo_safe_synthesizer/*/`) | `AGENTS.md` (module map section) |
| `docs/` structure (`mkdocs.yml` nav changes) | `.cursor/rules/writing-docs.mdc` (directory structure section) |
| `pytest.ini` markers (new markers added/removed) | `.agents/skills/diagnose-failures/SKILL.md` (markers table), `.claude/commands/unit-test.md` and related test commands |
| New skill added | `AGENTS.md` (skill index table), create in `.agents/skills/<name>/` |
| `errors.py` (new error classes) | `AGENTS.md` (error hierarchy table), `.agents/skills/diagnose-failures/SKILL.md` (error hierarchy section) |
| Hook scripts changed (`.cursor/hooks/`) | `.cursor/hooks.json` (Cursor config), `.claude/settings.json` (Claude Code config), `AGENTS.md` (hooks table) |
| `.cursor/worktrees.json` or `setup-worktree.sh` changed | `.agents/skills/git-worktrees/SKILL.md` (venv setup section) |

## New Skill Checklist

When creating a new skill:

1. Create `.agents/skills/<name>/SKILL.md` with YAML frontmatter (`name`, `description` with trigger keywords, `related-skills`)
2. Add to `AGENTS.md` skill index table -- skills not listed there are invisible to agents in fresh sessions
3. Force-add the directory (`.agents/skills/` is gitignored): `git add -f .agents/skills/<name>/`

## Verification

After updating, grep for stale references:

```bash
# Check for Makefile targets referenced in commands but no longer in Makefile
for f in .claude/commands/*.md; do
  target=$(grep -oP 'make \K[\w-]+' "$f" | head -1)
  [ -n "$target" ] && ! grep -q "^$target:" Makefile && echo "Stale: $f references make $target"
done

# Check for removed module directories still in AGENTS.md
for dir in $(grep '| `.*/' AGENTS.md | grep -oP '`\K[^`]+(?=/`)'); do
  [ ! -d "src/nemo_safe_synthesizer/$dir" ] && echo "Stale module in AGENTS.md: $dir"
done
```
