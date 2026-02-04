#!/usr/bin/env bash
#
# sync-from-mr.sh - Sync changes from a merged NMP GitLab MR to Safe-Synthesizer
#
# Usage: ./script/sync-from-mr.sh <MR_NUMBER>
#

set -euo pipefail

# Configuration
readonly GITLAB_HOST="gitlab-master.nvidia.com"
readonly PROJECT_ID="150981"
readonly NMP_REPO_PATH="${NMP_REPO_PATH:-$HOME/dev/aire/microservices/nmp}"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SAFE_SYNTH_ROOT="$(git rev-parse --show-toplevel)"

# Check dependencies
for cmd in glab jq rsync; do
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
    echo "Error: MR !$MR_NUMBER is not merged"
    echo "$MR_JSON" | jq .
    exit 1
fi

echo "Squash commit: $SQUASH_SHA"

# Checkout the squash commit in NMP repo
cd "$NMP_REPO_PATH"
git fetch origin main
git checkout "$SQUASH_SHA" --quiet

# Rsync excludes
RSYNC_EXCLUDES=(
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='.pytest_cache'
)

# Sync src/ and tests/ directories
echo "Syncing src/..."
rsync -av --delete "${RSYNC_EXCLUDES[@]}" \
    "$NMP_REPO_PATH/packages/nemo_safe_synthesizer/src/" \
    "$SAFE_SYNTH_ROOT/src/"

echo "Syncing tests/..."
rsync -av --delete "${RSYNC_EXCLUDES[@]}" \
    "$NMP_REPO_PATH/packages/nemo_safe_synthesizer/tests/" \
    "$SAFE_SYNTH_ROOT/tests/"

# Report files outside src/tests that changed in the MR
echo ""
echo "Files changed in MR but NOT synced (may need manual review):"
MR_CHANGES=$(glab api --hostname "$GITLAB_HOST" "projects/$PROJECT_ID/merge_requests/$MR_NUMBER/changes")
NON_SYNCED=$(echo "$MR_CHANGES" | jq -r '
    [.changes[] | (.old_path, .new_path)
     | select(startswith("packages/nemo_safe_synthesizer/"))
     | select(startswith("packages/nemo_safe_synthesizer/src/") | not)
     | select(startswith("packages/nemo_safe_synthesizer/tests/") | not)
     | ltrimstr("packages/nemo_safe_synthesizer/")]
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
echo "  git add -A && git commit -m 'Sync from NMP MR !$MR_NUMBER'"
