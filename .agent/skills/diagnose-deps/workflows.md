# Dependency Diagnosis Workflows

Extended step-by-step workflows for `tools/diff-lockfile.py`. See [SKILL.md](SKILL.md) for quick-reference commands.

## Bisect: Find Which Commit Broke a Test via Dep Changes

Full lifecycle -- bisect is error-prone, so follow every step.

```bash
# 1. Identify the known-good ref (last green CI, last working commit, etc.)
GOOD_REF=origin/main

# 2. Start bisect
git bisect start
git bisect bad HEAD
git bisect good $GOOD_REF

# 3. Run the bisect driver -- each step prints a lockfile diff then runs the test
git bisect run uv run tools/diff-lockfile.py --run tests/path/to/test.py::test_name

# 4. Read the bisect result (git prints the first bad commit)

# 5. Cleanup (ALWAYS do this, even if bisect errors out)
git bisect reset
```

Exit code reference:

| Code | Meaning | Bisect action |
|------|---------|---------------|
| 0 | Test passed | Mark commit as good |
| 1-124, 126-127 | Test failed | Mark commit as bad |
| 125 | `uv sync --frozen` failed | Skip this commit |

Common pitfalls:
- Forgetting `git bisect reset` leaves the repo in detached HEAD state.
- If bisect keeps skipping (exit 125), the lockfile is invalid at those commits -- narrow the range with a more recent `GOOD_REF`.
- The `--run` flag is required to activate bisect mode; without it the script only prints the diff.

## Cross-Skill: Correlate PR CI Failure with Dep Changes

Combine with the `github-cli` skill to check whether a PR's CI failure is caused by dependency changes.

```bash
# 1. Get the PR's base and head SHAs
BASE=$(gh pr view 123 --json baseRefOid -q .baseRefOid)
HEAD=$(gh pr view 123 --json headRefOid -q .headRefOid)

# 2. Fetch if needed
git fetch origin $BASE $HEAD

# 3. Count total dep changes
uv run tools/diff-lockfile.py $BASE --head $HEAD --json | jq 'length'

# 4. Check for GPU/ML-related changes (most likely CI breakers)
uv run tools/diff-lockfile.py $BASE --head $HEAD --json | \
  jq '[.[] | select(.name | test("torch|cuda|nvidia|vllm|transformers|unsloth"))]'

# 5. Check for major version bumps
uv run tools/diff-lockfile.py $BASE --head $HEAD --json | \
  jq '[.[] | select(.change == "upgraded") | select((.old.version | split(".")[0]) != (.new.version | split(".")[0]))]'
```

If CI failed and the diff shows relevant dep changes, the root cause is likely a transitive dependency upgrade.

## Pre-Merge Dependency Review

Before merging a branch that updates `uv.lock`, review what changed:

```bash
# 1. See the full human-readable diff
uv run tools/diff-lockfile.py origin/main

# 2. Check for anything suspicious in JSON
uv run tools/diff-lockfile.py origin/main --json | jq '
  [.[] | select(
    .change == "downgraded" or
    (.change == "removed") or
    (.change == "upgraded" and ((.old.version | split(".")[0]) != (.new.version | split(".")[0])))
  )]
'

# 3. Verify override-dependencies are still respected
#    Check pyproject.toml for overrides, then confirm those packages
#    didn't change unexpectedly in the diff
uv run tools/diff-lockfile.py origin/main --json | \
  jq '.[] | select(.name | test("outlines_core|flashinfer"))'
```

## Debugging a Specific Test Failure

Step-by-step when a test broke and you suspect transitive deps:

```bash
# 1. Run the diff to see what changed
uv run tools/diff-lockfile.py origin/main

# 2. Identify the failing test's imports (e.g., torch, transformers)
#    then filter the diff for those packages
uv run tools/diff-lockfile.py --json | \
  jq '[.[] | select(.name | test("torch|transformers|peft|accelerate"))]'

# 3. If a suspicious package upgraded, check its changelog
#    (agent can web-search "<package> changelog <new_version>")

# 4. If the cause is unclear, use bisect to pinpoint the commit
git bisect start
git bisect bad HEAD
git bisect good origin/main
git bisect run uv run tools/diff-lockfile.py --run tests/path/to/failing_test.py
git bisect reset
```
