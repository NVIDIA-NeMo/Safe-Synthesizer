<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# .codex/

Codex-specific hook registration for the shared agent tooling.

## hooks.json

Registers the shared signing hook for `PreToolUse` shell calls. Codex requires
the project and hook definition to be trusted before the hook runs; use
`/hooks` to review its status.

Codex currently ignores project hooks in linked Git worktrees due to
[openai/codex#27133](https://github.com/openai/codex/issues/27133). The commit
requirements in `AGENTS.md` remain authoritative when the hook is unavailable.

Use staged commits with standalone signing flags, such as
`git commit -s -S -m "message"`. The shared guard rejects non-message quoted
arguments and explicit signing negations rather than attempting to interpret
them.
