#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# sync-from-mr.sh - Sync changes from a merged NMP GitLab MR to Safe-Synthesizer
#
# Only copies files that were actually changed in the MR, rather than mirroring
# the full src/ and tests/ directories. This avoids deleting Safe-Synthesizer-
# specific files that don't yet exist in NMP.
#
# Usage: ./tools/sync-from-mr.sh <MR_NUMBER>
#

set -euo pipefail

# Configuration
readonly GITLAB_HOST="gitlab-master.nvidia.com"
readonly PROJECT_ID="150981"
readonly NMP_REPO_PATH="${NMP_REPO_PATH:-$HOME/dev/aire/microservices/nmp}"
readonly NMP_PKG_PREFIX="packages/nemo_safe_synthesizer/"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SAFE_SYNTH_ROOT="$(git rev-parse --show-toplevel)"

# Check dependencies
for cmd in glab jq; do
    command -v "$cmd" &>/dev/null || { echo "Error: $cmd is required"; exit 1; }
done

# Validate arguments
MR_NUMBER="${1:-}"
if [[ ! "$MR_NUMBER" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <MR_NUMBER>"
    exit 1
fi

# Check NMP repo exists
[[ -d "$NMP_REPO_PATH/.git" ]] || { echo "Error: NMP repo not found at $NMP_REPO_PATH"; exit 1; }

echo "Syncing from NMP MR !$MR_NUMBER..."

# Get merge commit SHA from GitLab API
MR_JSON=$(glab api --hostname "$GITLAB_HOST" "projects/$PROJECT_ID/merge_requests/$MR_NUMBER")
SQUASH_SHA=$(echo "$MR_JSON" | jq -r '.squash_commit_sha')

if [[ "$SQUASH_SHA" == "null" || -z "$SQUASH_SHA" ]]; then
    echo "Error: MR !$MR_NUMBER is not merged or has no squash commit"
    echo "$MR_JSON" | jq '{iid, title, state, merged_at}'
    exit 1
fi

echo "Squash commit: $SQUASH_SHA"

# Checkout the squash commit in NMP repo
cd "$NMP_REPO_PATH"
git fetch origin main
git checkout "$SQUASH_SHA" --quiet
cd "$SAFE_SYNTH_ROOT"

# Get all file changes in the MR scoped to our package
MR_CHANGES=$(glab api --hostname "$GITLAB_HOST" "projects/$PROJECT_ID/merge_requests/$MR_NUMBER/changes")

# Extract files in src/ and tests/ that were changed in the MR
# Each line: <action>\t<relative_path>
# action is one of: copy, delete
SYNC_PLAN=$(echo "$MR_CHANGES" | jq -r '
    .changes[] |
    # Consider both old_path and new_path for renames
    . as $change |
    [
        # If renamed/moved, delete the old path (if it was in our package)
        (if $change.renamed_file and ($change.old_path | startswith("'"$NMP_PKG_PREFIX"'"))
         then {action: "delete", path: ($change.old_path | ltrimstr("'"$NMP_PKG_PREFIX"'"))}
         else empty end),
        # Handle the new_path
        (if ($change.new_path | startswith("'"$NMP_PKG_PREFIX"'"))
         then
            ($change.new_path | ltrimstr("'"$NMP_PKG_PREFIX"'")) as $rel |
            if ($rel | startswith("src/")) or ($rel | startswith("tests/"))
            then
                if $change.deleted_file
                then {action: "delete", path: $rel}
                else {action: "copy", path: $rel}
                end
            else empty
            end
         else empty end)
    ] | .[] | "\(.action)\t\(.path)"
')

if [[ -z "$SYNC_PLAN" ]]; then
    echo "No src/ or tests/ files changed in MR !$MR_NUMBER for nemo_safe_synthesizer."
    echo ""
    echo "This MR may have only changed files outside src/ and tests/."
else
    # Process the sync plan
    COPIED=0
    DELETED=0

    while IFS=$'\t' read -r action rel_path; do
        case "$action" in
            copy)
                src="$NMP_REPO_PATH/$NMP_PKG_PREFIX$rel_path"
                dst="$SAFE_SYNTH_ROOT/$rel_path"
                if [[ -f "$src" ]]; then
                    mkdir -p "$(dirname "$dst")"
                    cp "$src" "$dst"
                    echo "  copied: $rel_path"
                    COPIED=$((COPIED + 1))
                else
                    echo "  WARNING: source file not found: $src"
                fi
                ;;
            delete)
                dst="$SAFE_SYNTH_ROOT/$rel_path"
                if [[ -f "$dst" ]]; then
                    rm "$dst"
                    echo "  deleted: $rel_path"
                    DELETED=$((DELETED + 1))
                else
                    echo "  (skip delete, not present): $rel_path"
                fi
                ;;
        esac
    done <<< "$SYNC_PLAN"

    echo ""
    echo "Synced $COPIED file(s), deleted $DELETED file(s)."
fi

# Report files outside src/tests that changed in the MR (need manual review)
echo ""
echo "Files changed in MR but NOT synced (outside src/ and tests/, may need manual review):"
NON_SYNCED=$(echo "$MR_CHANGES" | jq -r '
    [.changes[] | (.old_path, .new_path)
     | select(startswith("'"$NMP_PKG_PREFIX"'"))
     | ltrimstr("'"$NMP_PKG_PREFIX"'")
     | select(startswith("src/") | not)
     | select(startswith("tests/") | not)]
    | unique | .[]')
if [[ -n "$NON_SYNCED" ]]; then
    echo "$NON_SYNCED"
else
    echo "  (none)"
fi

echo ""
echo "Sync complete! Next steps:"
echo "  git status"
echo "  git diff"
echo "  git add -A && git commit -m 'chore: sync from NMP MR !$MR_NUMBER'"
