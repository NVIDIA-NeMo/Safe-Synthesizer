#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Guards recognized direct `git commit` commands against missing sign-off or
# signing flags. Exit code 2 denies the action in Cursor, Claude Code, and Codex.
# This is intentionally not a parser for nested shells or process wrappers.

set -eu

json_input=$(cat)

# Cursor puts the command at .command (beforeShellExecution);
# Claude Code puts it at .tool_input.command (PreToolUse for Bash).
command=$(printf '%s\n' "$json_input" | jq -r '.command // .tool_input.command // empty')
# Quoted text cannot contain shell structure that this direct-command guard
# should inspect. Preserve each quoted word as a neutral placeholder so message
# options still retain their argument during the flag scan.
command_text=$(printf '%s\n' "$command" | sed -E \
    -e 's/"([^"\\]|\\.)*"/QUOTED/g' \
    -e "s/'[^']*'/QUOTED/g")
shell_word="([^[:space:]\"']+|\"[^\"]*\"|'[^']*')+"
assignment="[A-Za-z_][A-Za-z0-9_]*=(${shell_word})?"
command_prefix="(${assignment}[[:space:]]+)*(env([[:space:]]+${shell_word})*[[:space:]]+)?(${assignment}[[:space:]]+)*"
git_global_option="[[:space:]]+-[^[:space:]]+([[:space:]]+${shell_word})?"
commit_pattern="^[[:space:]]*\\(*[[:space:]]*${command_prefix}git(${git_global_option})*[[:space:]]+commit([[:space:]]|$)"

while IFS= read -r segment; do
    if ! printf '%s\n' "$segment" | grep -qE "$commit_pattern"; then
        continue
    fi

    flag_text=$(printf '%s\n' "$segment" | sed -E \
        -e 's/[()]//g' \
        -e 's/(^|[[:space:]])(-m[^[:space:]]+|--message=[^[:space:]]+)([[:space:]]|$)/ /g' \
        -e 's/(^|[[:space:]])(-m|--message)(=[^[:space:]]+|[[:space:]]+[^[:space:]]+)//g')

    if printf '%s\n' "$flag_text" | grep -q 'QUOTED'; then
        echo "Quoted non-message commit arguments are unsupported. Use standalone -s and -S flags." >&2
        exit 2
    fi
    if printf '%s\n' "$flag_text" | grep -qE '(^|[[:space:]])--no-signoff([[:space:]]|$)' \
        || ! printf '%s\n' "$flag_text" | grep -qE '(^|[[:space:]])(--signoff|-s)([[:space:]]|$)'; then
        echo "All commits require DCO sign-off. Re-run with --signoff (or -s)." >&2
        exit 2
    fi
    if printf '%s\n' "$flag_text" | grep -qE '(^|[[:space:]])--no-gpg-sign([[:space:]]|$)' \
        || ! printf '%s\n' "$flag_text" | grep -qE '(^|[[:space:]])(--gpg-sign(=[^[:space:]]+)?|-S[^[:space:]]*)([[:space:]]|$)'; then
        echo "All commits must be GPG-signed. Re-run with --gpg-sign (or -S)." >&2
        exit 2
    fi
done < <(printf '%s\n' "$command_text" | sed -E 's/[;&|]+/\n/g')

exit 0
