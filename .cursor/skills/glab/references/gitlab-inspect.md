<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->


# GitLab Inspection and Debugging

Inspect GitLab merge requests, check approval status, investigate pipeline failures, and debug CI/CD jobs using the `glab` CLI.

## Prerequisites

The `glab` CLI must be configured for `gitlab-master.nvidia.com`. See the Setup section below.

## Shell Permissions

When running `glab` commands, always use `required_permissions: ["all"]` to avoid TLS certificate verification issues with the corporate GitLab instance.

## Setup

If `glab` is not configured:

```bash
# Install glab (if needed)
brew install glab

# Configure for Nvidia GitLab
glab config set -g host gitlab-master.nvidia.com

# Authenticate
glab auth login --hostname gitlab-master.nvidia.com
```

If `glab` is aliased to "op plugin run --glab":

```bash
# use this glab or antoher unaliased version
/opt/homebrew/bin/glab
```


## Understanding the jq Commands

All jq commands in this skill include defensive checks to handle API failures gracefully:

- **Array checks**: `if type == "array" and length > 0 then ... else empty end`
  - Prevents errors when the API returns empty arrays or non-array responses
  - Returns `empty` (no output) instead of an error
- **Field checks**: `if .field then ... else {} end`
  - Ensures fields exist before accessing nested properties
  - Returns empty objects `{}` instead of errors

These patterns make the commands more robust when running in scripts or when the API returns unexpected results.

## Project References

When using the GitLab API directly, you can reference the NMP project in different ways:

- **`:id`** - Works inside the cloned NMP repository (recommended when inside the repo)
- **`aire%2Fmicroservices%2Fnmp`** - URL-encoded path, works anywhere
- **`150981`** - Numeric project ID, works anywhere

**Examples for NMP:**

```bash
# Inside NMP repo (most common, recommended)
glab api "projects/:id/pipelines"

# Outside repo or in scripts (use URL-encoded path)
glab api "projects/aire%2Fmicroservices%2Fnmp/pipelines"

# Using numeric project ID
glab api "projects/150981/pipelines"
```

**Note:** All examples below use `:id` assuming you're working inside the NMP repo. Replace with `aire%2Fmicroservices%2Fnmp` or `150981` if running commands outside the repo.

## Listing Merge Requests

### Your Open MRs (All Projects)

List all your open MRs across all projects you have access to:

```bash
glab api "merge_requests?state=opened&author_username=$(glab api user | jq -r .username)&per_page=20" | \
  jq -r 'if type == "array" and length > 0 then .[] | "\(.project_id)\t\(.iid)\t\(.references.full)\t\(.title)" else empty end'
```

This shows: `<project-id>  <mr-iid>  <full-reference>  <title>`

### MRs in Current Repository

```bash
# Your MRs in current repo
glab mr list --author=@me

# All open MRs
glab mr list --state=opened

# MRs by assignee
glab mr list --assignee="username"
```

## Viewing MR Details

### Quick View

```bash
# Inside NMP repo
glab mr view <mr-iid>

# Example: View MR !5603
glab mr view 5603
```

### Full MR Data (API)

```bash
# Get all MR details as JSON
glab api "projects/:id/merge_requests/<mr-iid>" | jq '.'

# Example: Get details for MR !5603
glab api "projects/:id/merge_requests/5603" | jq '.'
```

### MR Comments and Discussion

```bash
# List all comments on an MR
glab api "projects/:id/merge_requests/<mr-iid>/notes" | \
  jq 'if type == "array" and length > 0 then .[] | {author: .author.username, created: .created_at, body: .body} else empty end'

# Example: Comments on MR !5603
glab api "projects/:id/merge_requests/5603/notes" | \
  jq 'if type == "array" and length > 0 then .[] | {author: .author.username, created: .created_at, body: .body} else empty end'
```

## Checking Approval Status

### Quick Approval Check

```bash
# Check if MR has required approvals
glab api "projects/:id/merge_requests/<mr-iid>/approvals" | \
  jq 'if .approved_by then {approved, approvals_required, approvals_left, approved_by: [.approved_by[].user.username]} else {approved, approvals_required, approvals_left, approved_by: []} end'

# Example: Check approvals for MR !5603
glab api "projects/:id/merge_requests/5603/approvals" | \
  jq 'if .approved_by then {approved, approvals_required, approvals_left, approved_by: [.approved_by[].user.username]} else {approved, approvals_required, approvals_left, approved_by: []} end'
```

**Example output:**

```json
{
  "approved": true,
  "approvals_required": 2,
  "approvals_left": 0,
  "approved_by": ["schapman", "bmccown"]
}
```

### Approval Rules (CODEOWNERS)

Check detailed approval rules including CODEOWNERS requirements:

```bash
# See each approval rule separately
glab api "projects/:id/merge_requests/<mr-iid>/approval_state" | \
  jq 'if .rules and (.rules | type == "array") and (.rules | length > 0) then .rules[] | {name, approvals_required, approvals_left, approved_by: [.approved_by[].username]} else empty end'

# Example: Check approval rules for MR !5603
glab api "projects/:id/merge_requests/5603/approval_state" | \
  jq 'if .rules and (.rules | type == "array") and (.rules | length > 0) then .rules[] | {name, approvals_required, approvals_left, approved_by: [.approved_by[].username]} else empty end'
```

This shows each approval rule separately (e.g., "Any eligible user", "CODEOWNERS").

## Checking Merge Readiness

### Quick Merge Status

```bash
# Check if MR is ready to merge
glab api "projects/:id/merge_requests/<mr-iid>" | \
  jq '{merge_status, has_conflicts, draft, pipeline: (if .head_pipeline then .head_pipeline.status else null end), blocking_discussions_resolved}'

# Example: Check merge readiness for MR !5603
glab api "projects/:id/merge_requests/5603" | \
  jq '{merge_status, has_conflicts, draft, pipeline: (if .head_pipeline then .head_pipeline.status else null end), blocking_discussions_resolved}'
```

**Example output:**

```json
{
  "merge_status": "can_be_merged",
  "has_conflicts": false,
  "draft": false,
  "pipeline": "success",
  "blocking_discussions_resolved": true
}
```

**Note:** If no pipeline exists, `pipeline` will be `null`.

**Merge status values:**

- `can_be_merged` - Ready to merge
- `cannot_be_merged` - Has conflicts
- `cannot_be_merged_recheck` - Checking for conflicts
- `unchecked` - Not yet checked

### Comprehensive Readiness Check (All Your MRs)

Check the merge readiness of all your open MRs at once:

```bash
glab api "merge_requests?state=opened&author_username=$(glab api user | jq -r .username)&per_page=20" | \
  jq -c 'if type == "array" and length > 0 then .[] | {project_id, iid, ref: .references.full} else empty end' | \
  while read -r mr; do
    pid=$(echo "$mr" | jq -r '.project_id')
    iid=$(echo "$mr" | jq -r '.iid')
    ref=$(echo "$mr" | jq -r '.ref')
    echo "--- $ref ---"
    glab api "projects/$pid/merge_requests/$iid" | \
      jq '{merge_status, has_conflicts, draft, pipeline: (if .head_pipeline then .head_pipeline.status else null end), discussions_resolved: .blocking_discussions_resolved}'
    glab api "projects/$pid/merge_requests/$iid/approvals" | \
      jq 'if .approved_by then {approved, approvals_left, approved_by: [.approved_by[].user.username]} else {approved, approvals_left, approved_by: []} end'
  done
```

This is useful for getting a dashboard view of all your MRs.

## Investigating Pipelines

### List MR Pipelines

Get all pipelines for an MR:

```bash
# List all pipelines for an MR
glab api "projects/:id/merge_requests/<mr-iid>/pipelines" | \
  jq 'if type == "array" and length > 0 then .[] | {id, status, source, created_at} else empty end'

# Example: Pipelines for MR !5603
glab api "projects/:id/merge_requests/5603/pipelines" | \
  jq 'if type == "array" and length > 0 then .[] | {id, status, source, created_at} else empty end'
```

**Note:** MR pipelines are triggered by `merge_request_event`, separate from branch push pipelines.

### List Branch Pipelines

```bash
# In NMP repo - list recent pipelines
glab ci list

# Specific branch (API)
glab api "projects/:id/pipelines?ref=<branch-name>&per_page=20" | \
  jq 'if type == "array" and length > 0 then .[] | {id, status, created_at} else empty end'

# Example: Pipelines for main branch
glab api "projects/:id/pipelines?ref=main&per_page=20" | \
  jq 'if type == "array" and length > 0 then .[] | {id, status, created_at} else empty end'
```

### List Failed Pipelines

```bash
# List recent failed pipelines in NMP
glab api "projects/:id/pipelines?status=failed&per_page=20" | \
  jq 'if type == "array" and length > 0 then .[] | {id, ref, created_at} else empty end'
```

### Get Pipeline Details

```bash
# Get full details for a specific pipeline
glab api "projects/:id/pipelines/<pipeline-id>" | jq '.'

# Example: Details for pipeline 123456
glab api "projects/:id/pipelines/123456" | jq '.'
```

## Investigating Failed Jobs

### List All Jobs in a Pipeline

```bash
# List all jobs in a pipeline
glab api "projects/:id/pipelines/<pipeline-id>/jobs" | \
  jq 'if type == "array" and length > 0 then .[] | {id, status, name, stage} else empty end'

# Example: Jobs for pipeline 123456
glab api "projects/:id/pipelines/123456/jobs" | \
  jq 'if type == "array" and length > 0 then .[] | {id, status, name, stage} else empty end'
```

### List Only Failed Jobs

```bash
# Find which jobs failed in a pipeline
glab api "projects/:id/pipelines/<pipeline-id>/jobs" | \
  jq 'if type == "array" and length > 0 then .[] | select(.status == "failed") | {id, name, stage, failure_reason} else empty end'

# Example: Failed jobs in pipeline 123456
glab api "projects/:id/pipelines/123456/jobs" | \
  jq 'if type == "array" and length > 0 then .[] | select(.status == "failed") | {id, name, stage, failure_reason} else empty end'
```

### Get Job Log

```bash
# Get the full log output from a job
glab api "projects/:id/jobs/<job-id>/trace"

# Example: Log for job 789012
glab api "projects/:id/jobs/789012/trace"
```

**Tip:** Pipe to `less` for large logs:

```bash
glab api "projects/:id/jobs/789012/trace" | less
```

Or save to file:

```bash
glab api "projects/:id/jobs/789012/trace" > job-log.txt
```

### Investigating Child/Bridge Pipeline Failures

Sometimes a parent pipeline shows as failed, but all direct jobs passed. This indicates a child pipeline (triggered via bridge job) failed:

```bash
# Check for failed bridge/child pipelines
glab api "projects/:id/pipelines/<pipeline-id>/bridges" | \
  jq 'if type == "array" and length > 0 then .[] | select(.status == "failed") | {name, failure_reason} else empty end'

# Example: Bridge failures in pipeline 123456
glab api "projects/:id/pipelines/123456/bridges" | \
  jq 'if type == "array" and length > 0 then .[] | select(.status == "failed") | {name, failure_reason} else empty end'
```

## Common Workflows

### Workflow: Check MR Before Merging

Complete pre-merge check for an NMP MR:

```bash
# Example: Check MR !5603 before merging
MR_IID=5603

# 1. Check approval status
echo "=== Approvals ==="
glab api "projects/:id/merge_requests/$MR_IID/approvals" | \
  jq 'if .approved_by then {approved, approvals_left, approved_by: [.approved_by[].user.username]} else {approved, approvals_left, approved_by: []} end'

# 2. Check pipeline status
echo "=== Pipeline ==="
glab api "projects/:id/merge_requests/$MR_IID/pipelines" | \
  jq 'if type == "array" and length > 0 then .[0] | {status, id, created_at} else {} end'

# 3. Check merge readiness
echo "=== Merge Status ==="
glab api "projects/:id/merge_requests/$MR_IID" | \
  jq '{merge_status, has_conflicts, blocking_discussions_resolved, pipeline: (if .head_pipeline then .head_pipeline.status else null end)}'
```

### Workflow: Debug Failed Pipeline

Step-by-step pipeline debugging:

```bash
# Example: Debug failed pipeline for MR !5603
MR_IID=5603

# 1. Get the latest pipeline ID
PIPELINE_ID=$(glab api "projects/:id/merge_requests/$MR_IID/pipelines" | \
  jq -r 'if type == "array" and length > 0 then .[0].id else "" end')

if [ -z "$PIPELINE_ID" ]; then
  echo "No pipelines found for this MR"
  exit 1
fi

echo "Checking pipeline: $PIPELINE_ID"

# 2. List all failed jobs
echo "=== Failed Jobs ==="
glab api "projects/:id/pipelines/$PIPELINE_ID/jobs" | \
  jq 'if type == "array" and length > 0 then .[] | select(.status == "failed") | {id, name, stage} else empty end'

# 3. Get log for the first failed job
echo "=== First Failed Job Log ==="
JOB_ID=$(glab api "projects/:id/pipelines/$PIPELINE_ID/jobs" | \
  jq -r 'if type == "array" and length > 0 then .[] | select(.status == "failed") | .id else "" end' | head -1)

if [ -n "$JOB_ID" ]; then
  glab api "projects/:id/jobs/$JOB_ID/trace"
else
  echo "No failed jobs found"
fi
```

### Workflow: Monitor Pipeline Until Complete

Wait for a pipeline to finish:

```bash
# Example: Monitor pipeline 123456
PIPELINE_ID=123456

# Poll until complete
while true; do
  STATUS=$(glab api "projects/:id/pipelines/$PIPELINE_ID" | jq -r '.status')
  echo "$(date '+%H:%M:%S') Pipeline $PIPELINE_ID: $STATUS"
  case "$STATUS" in
    success|failed|canceled) break ;;
    *) sleep 30 ;;
  esac
done

echo "Pipeline finished with status: $STATUS"
```

## Reference Tables

### Pipeline Status Values

| Status | Description |
|--------|-------------|
| `created` | Pipeline created but not yet running |
| `waiting_for_resource` | Waiting for available runner |
| `preparing` | Preparing to run |
| `pending` | Jobs are pending |
| `running` | Pipeline is running |
| `success` | All jobs passed |
| `failed` | At least one job failed |
| `canceled` | Pipeline was canceled |
| `skipped` | Pipeline was skipped |
| `manual` | Waiting for manual action |

### Job Status Values

| Status | Description |
|--------|-------------|
| `created` | Job created |
| `pending` | Waiting for runner |
| `running` | Job is running |
| `success` | Job passed |
| `failed` | Job failed |
| `canceled` | Job was canceled |
| `skipped` | Job was skipped |
| `manual` | Waiting for manual trigger |

## Tips and Best Practices

1. **Use `:id` inside the NMP repo** - Most convenient when running commands from within the cloned repository
2. **Use `aire%2Fmicroservices%2Fnmp` or `150981` for scripts** - More portable for automation that runs outside the repo
3. **Check child pipelines** - If parent shows failed but all jobs passed, check bridge jobs with the bridges endpoint
4. **Monitor with polling** - Use short sleep intervals (30s) when waiting for pipelines to complete
5. **Save logs for debugging** - Pipe job traces to files for easier analysis and sharing
6. **Batch checks** - When checking multiple MRs, script the checks in a loop (see "Comprehensive Readiness Check" example)
7. **Reference real MR numbers** - Use actual MR IIDs (like `5603`) not URLs when using `glab` commands
