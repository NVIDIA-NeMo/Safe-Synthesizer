---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: skill-audit
description: "Audit agent chat transcripts and update repo skills based on patterns, friction, and lessons learned. Triggers on: update skills, audit skills, review chats, improve skills, skill gaps, transcript analysis, what did we learn."
related-skills: [sync-agent-config]
---

# Skill Audit

Mine agent chat transcripts for patterns and update the repo's skills accordingly.

## Locate Transcripts

Transcripts live under the Cursor projects directory. Two formats exist:

- Flat: `<uuid>.txt` -- plain text, one file per chat
- Directory-based: `<uuid>/<uuid>.jsonl` with optional `subagents/` subdirectory

```bash
# Find all transcript directories
ls -d /root/.cursor/projects/*/agent-transcripts/ 2>/dev/null

# List recent transcripts by date
ls -lt "$TDIR" | head -20
```

Multiple workspace directories may exist -- check all of them.

## Search Transcripts

Use `grep -rl` to find transcripts matching a topic, then count hits to prioritize:

```bash
# Find transcripts mentioning a topic
grep -rl '<keyword>' "$TDIR" | head -20

# Rank by relevance
for f in $(grep -rl '<keyword>' "$TDIR"); do
  echo "$(grep -c '<keyword>' "$f") $f"
done | sort -rn
```

Search terms by skill area:

| Skill area | Search terms |
|------------|-------------|
| github-cli | `gh pr`, `gh run`, `gh issue`, `gh api`, `gh auth` |
| testing | `pytest`, `make test`, `xdist`, `fixture`, `conftest` |
| typing | `ty check`, `ty:`, `type:`, `mypy` |
| config | `pydantic_options`, `Parameter`, `AutoParam`, `configurator` |
| deps | `uv lock`, `uv add`, `uv sync`, `lockfile` |
| sync | `synchronize`, `NMP`, `rsync`, `sync-from` |

## Analyze Transcripts

Use parallel subagents (3-4 at a time) to read batches of transcripts. For each, extract:

- UUID, date, brief title (6 words max)
- What the user was trying to accomplish
- Tools/skills activated or attempted
- Friction encountered (errors, workarounds, wasted turns)
- Patterns discovered (commands, flags, workflows)
- Anti-patterns observed (things that didn't work)
- Lessons that should be captured in skills

Prioritize transcripts with the most keyword hits and largest file sizes.

## Cross-Reference with Skills

Read all current skills in parallel:

```bash
ls .agent/skills/*/SKILL.md
ls .agent/skills/*/references/ 2>/dev/null
```

For each finding, ask:
- Which skill should capture this? (or does it need a new skill?)
- Is this already documented? If so, is the documentation sufficient?
- Is this a one-off or a recurring pattern? (recurring = higher priority)

## Propose Updates

Structure the output as a plan with:
- One section per skill file being updated
- Citation of the source chat(s) using `[short title](uuid)` format
- The specific text/section to add or modify
- Anti-patterns and common mistakes to call out

## Execute Updates

Follow the [`sync-agent-config`](../sync-agent-config/SKILL.md) checklist when creating new skills. For existing skills, edit in place.

After any skill changes: `make format && make lint`.

## Conventions

- Cite source chats for every proposed change -- no unsupported claims
- Recurring patterns (3+ chats) take priority over one-offs
- Keep skill edits minimal and focused -- don't rewrite skills that work fine
- If creating a new skill, follow the new skill checklist in `sync-agent-config`
