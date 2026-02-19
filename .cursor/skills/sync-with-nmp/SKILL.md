---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: sync-with-nmp
description: "Synchronize code between the Safe-Synthesizer GitHub repo and the internal NMP GitLab repo. Use when the user wants to sync changes from an NMP merge request, push changes to NMP, pull the latest NMP code, or manage the bidirectional sync workflow. Trigger keywords - sync, synchronize, NMP, GitLab, merge request, MR, rsync, upstream, downstream, pull from NMP, push to NMP."
---

# Sync with NMP

## Detailed References

- **[Full Reference](./references/full-reference.md)** - Comprehensive bidirectional sync reference
- **[Workflows](./references/workflows.md)** - Single MR sync, batch sync, push to NMP, verify consistency, pre-sync investigation

**Note:** The commands below are quick references. For comprehensive detail, see the references above.

## Architecture

```
Safe-Synthesizer (GitHub)           NMP (GitLab)
├── src/                    ←→      packages/nemo_safe_synthesizer/src/
├── tests/                  ←→      packages/nemo_safe_synthesizer/tests/
├── design.md               ←→      packages/nemo_safe_synthesizer/design.md
├── script/                 ←→      packages/nemo_safe_synthesizer/script/
└── (metafiles)             ←→      packages/nemo_safe_synthesizer/(metafiles)
```

## Shell Permissions

Always use `required_permissions: ["all"]` for sync commands. `glab` API calls need unrestricted network access, and `rsync` crosses repo boundaries.

## Prerequisites

- `glab` CLI configured for `gitlab-master.nvidia.com`
- NMP repo cloned locally (set `NMP_REPO_PATH`)
- `rsync` and `jq` installed

## Quick Reference

```bash
# Sync a specific NMP MR (creates branch, applies changes)
make synchronize-from-nmp-mr MR=<number>

# Full sync NMP → Safe-Synthesizer (requires NMP_REPO_PATH)
make synchronize-from-nmp

# Full sync Safe-Synthesizer → NMP
make synchronize-to-nmp

# Sync only Python files
make synchronize-py-files-from-nmp
make synchronize-py-files-to-nmp

# Sync only metafiles (design.md, scripts, etc.)
make synchronize-metafiles-from-nmp
make synchronize-metafiles-to-nmp
```
