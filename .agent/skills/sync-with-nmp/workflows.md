# Sync Workflows

Detailed multi-step recipes for synchronizing between Safe-Synthesizer and NMP. Referenced from [SKILL.md](SKILL.md).

## Workflow: Sync a Single NMP MR

The most common flow. An MR merged in NMP needs to be reflected in Safe-Synthesizer.

### Step 1: Verify the MR is Merged

Use the GitLab MCP server or browse `gitlab-master.nvidia.com` to confirm the MR is merged and note its squash commit SHA.

### Step 2: Check What Files Changed

Review the MR's changed files in GitLab. Files under `src/` and `tests/` sync automatically. Others need manual attention.

### Step 3: Ensure Clean Working Tree

```bash
cd /Users/aagonzales/dev/Safe-Synthesizer
git status
git stash  # if needed
git checkout main && git pull
```

### Step 4: Run the Sync

```bash
make synchronize-from-nmp-mr MR=$MR_IID
```

This auto-creates a branch `$USER/sync-$MR_IID-from-nmp` if you're on main.

### Step 5: Handle Non-Synced Files

The script reports files outside `src/` and `tests/`. If any were listed:

```bash
export NMP_REPO_PATH="$HOME/dev/aire/microservices/nmp"

# Option A: Copy specific files manually
cp "$NMP_REPO_PATH/packages/nemo_safe_synthesizer/design.md" ./design.md

# Option B: Sync all metafiles
make synchronize-metafiles-from-nmp
```

### Step 6: Format, Lint, Test

```bash
make format
make lint
make test
```

Fix any issues before proceeding.

### Step 7: Commit and Push

```bash
git add -A
git commit -s -m "chore: sync from NMP MR !$MR_IID"
git push -u origin HEAD
```

### Step 8: Create PR

```bash
gh pr create \
  --title "chore: sync from NMP MR !$MR_IID" \
  --body "$(cat <<'EOF'
## Summary
Syncs changes from NMP MR !<MR_IID> into Safe-Synthesizer.

## Type of Change
- [x] Chore (sync)

## Testing
- [x] Tests pass locally (`make test`)
- [x] Format and lint pass (`make format && make lint`)
EOF
)"
```

---

## Workflow: Sync Multiple NMP MRs

When several MRs have landed in NMP since the last sync.

### Step 1: Identify MRs to Sync

Use the GitLab MCP server or browse the NMP project's merged MRs to find those touching `packages/nemo_safe_synthesizer/`. Note their IIDs.

### Step 2: Full Sync Instead of Per-MR

For multiple MRs, a full sync is simpler than syncing each MR individually:

```bash
cd /Users/aagonzales/dev/Safe-Synthesizer
git checkout main && git pull
git checkout -b $USER/sync-batch-from-nmp

export NMP_REPO_PATH="$HOME/dev/aire/microservices/nmp"

# Ensure NMP is on latest main
cd "$NMP_REPO_PATH"
git checkout main && git pull
cd -

# Full sync
make synchronize-from-nmp
```

### Step 3: Review, Format, Lint, Test

```bash
git diff --stat
make format
make lint
make test
```

### Step 4: Commit with MR References

```bash
git add -A
git commit -s -m "$(cat <<'EOF'
chore: batch sync from NMP

Syncs the following NMP MRs:
- !1234 - description
- !1235 - description
- !1236 - description
EOF
)"
git push -u origin HEAD
```

---

## Workflow: Push Changes to NMP

When a change is made in Safe-Synthesizer that needs to go back to NMP.

### Step 1: Sync to NMP

```bash
cd /Users/aagonzales/dev/Safe-Synthesizer
export NMP_REPO_PATH="$HOME/dev/aire/microservices/nmp"

make synchronize-to-nmp
```

### Step 2: Review Changes in NMP

```bash
cd "$NMP_REPO_PATH"
git diff --stat -- packages/nemo_safe_synthesizer/
```

### Step 3: Create NMP Branch and Commit

```bash
cd "$NMP_REPO_PATH"
git checkout -b <issue-number>-sync-from-oss/$USER
git add packages/nemo_safe_synthesizer/
git commit -m "chore(nss): sync from Safe-Synthesizer"
git push -u origin HEAD
```

### Step 4: Create NMP MR

Create a merge request in the NMP GitLab project for the new branch.

---

## Workflow: Verify Sync Consistency

Check that the two repos are in sync by comparing file trees.

### Quick Diff

```bash
export NMP_REPO_PATH="$HOME/dev/aire/microservices/nmp"

# Compare src/ directories
diff -rq \
  src/nemo_safe_synthesizer/ \
  "$NMP_REPO_PATH/packages/nemo_safe_synthesizer/src/nemo_safe_synthesizer/" \
  --exclude='__pycache__' --exclude='*.pyc'

# Compare tests/ directories
diff -rq \
  tests/ \
  "$NMP_REPO_PATH/packages/nemo_safe_synthesizer/tests/" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache'
```

If no output, the directories are identical.

### Detailed Diff

```bash
diff -r \
  src/nemo_safe_synthesizer/ \
  "$NMP_REPO_PATH/packages/nemo_safe_synthesizer/src/nemo_safe_synthesizer/" \
  --exclude='__pycache__' --exclude='*.pyc' | head -100
```

---

## Workflow: Pre-Sync NMP MR Investigation

Before syncing, investigate what an NMP MR changed and whether it's safe to sync.

Use the GitLab MCP server or browse the MR on `gitlab-master.nvidia.com` to:

1. Confirm the MR is merged and its pipeline passed
2. Review changed files -- look for changes to public interfaces (`cli/`, `config/`, `errors.py`)
3. Note any files outside `src/` and `tests/` that will need manual sync
