---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: glab
description: "Interact with Nvidia's GitLab (gitlab-master.nvidia.com) using the glab CLI. Activate when users want to find merge requests, check approval status, check merge readiness, check pipeline status, investigate job failures, read MR comments, or view job logs. Trigger keywords - merge request, MR, pipeline, CI, job failure, failed job, job log, glab, gitlab, approval, approvals, ready to merge, CODEOWNERS."
---

# GitLab CLI (glab)

## Detailed References

- **[GitLab Inspect](./references/gitlab-inspect.md)** - Comprehensive MR inspection, approval checks, pipeline debugging, and failed job investigation (includes defensive jq patterns)

**Note:** The commands below are quick references. For production use or scripts, use the defensive patterns from the GitLab Inspect reference to handle empty arrays and missing fields gracefully.

## Setup
```bash
# osx or homebrew
brew install glab
glab config set -g host gitlab-master.nvidia.com
glab auth login --hostname gitlab-master.nvidia.com
```

```bash
# linux
wget https://gitlab.com/gitlab-org/cli/-/releases/v1.80.4/downloads/glab_1.80.4_linux_amd64.tar.gz
tar -xzvf glab_1.80.4_linux_amd64.tar.gz --strip-components=1 -C . $HOME/.local/bin
rm glab_1.80.4_linux_amd64.tar.gz
glab config set -g host gitlab-master.nvidia.com
glab auth login --hostname gitlab-master.nvidia.com
```

If `glab` is aliased to "op plugin run --glab", which might be the case if '[ERROR] 2026/02/04 09:13:18 authorization prompt dismissed, please try again' shows up, then use the following unaliased version.

```bash
# use this glab or another unaliased version for all commands instead of glab
/opt/homebrew/bin/glab
```

## Project References
- `:id` - works inside cloned repo only
- `aire%2Fmicroservices%2Fnmp` - URL-encoded path, works anywhere
- `150981` - numeric project ID, works anywhere

## Merge Requests

```bash
# Your open MRs (all projects)
glab api "merge_requests?state=opened&author_username=$(glab api user | jq -r .username)&per_page=20" | jq -r '.[] | "\(.project_id)\t\(.iid)\t\(.references.full)\t\(.title)"'

# MRs in current repo
glab mr list --author=@me

# MR details
glab mr view <mr-iid>

# MR comments
glab api "projects/<project-id>/merge_requests/<mr-iid>/notes" | jq '.[] | {author: .author.username, body: .body}'
```

## Approvals & Merge Readiness

```bash
# Approval status
glab api "projects/<project-id>/merge_requests/<mr-iid>/approvals" | jq '{approved, approvals_required, approvals_left, approved_by: [.approved_by[].user.username]}'

# Approval rules (CODEOWNERS, etc)
glab api "projects/<project-id>/merge_requests/<mr-iid>/approval_state" | jq '.rules[] | {name, approvals_required, approvals_left, approved_by: [.approved_by[].username]}'

# MR merge readiness
glab api "projects/<project-id>/merge_requests/<mr-iid>" | jq '{merge_status, has_conflicts, draft, pipeline: .head_pipeline.status, blocking_discussions_resolved}'

# All your open MRs - full readiness check
glab api "merge_requests?state=opened&author_username=$(glab api user | jq -r .username)&per_page=20" | jq -c '.[] | {project_id, iid, ref: .references.full}' | while read -r mr; do
  pid=$(echo "$mr" | jq -r '.project_id')
  iid=$(echo "$mr" | jq -r '.iid')
  ref=$(echo "$mr" | jq -r '.ref')
  echo "--- $ref ---"
  glab api "projects/$pid/merge_requests/$iid" | jq '{merge_status, has_conflicts, draft, pipeline: .head_pipeline.status, discussions_resolved: .blocking_discussions_resolved}'
  glab api "projects/$pid/merge_requests/$iid/approvals" | jq '{approved, approvals_left, approved_by: [.approved_by[].user.username]}'
done
```

## Pipelines

```bash
# MR pipelines
glab api "projects/<project-id>/merge_requests/<mr-iid>/pipelines" | jq '.[] | {id, status, source}'

# Branch pipelines
glab ci list
glab api "projects/:id/pipelines?ref=<branch>&per_page=20" | jq '.[] | {id, status, created_at}'

# Failed pipelines
glab api "projects/:id/pipelines?status=failed&per_page=20" | jq '.[] | {id, ref, created_at}'
```

Note: `glab ci status -b <branch>` shows branch pipelines (push-triggered), not MR pipelines (merge_request_event-triggered). These are separate.

## Jobs

```bash
# List jobs in pipeline
glab api "projects/<project-id>/pipelines/<pipeline-id>/jobs" | jq '.[] | {status, name, stage}'

# Failed jobs
glab api "projects/<project-id>/pipelines/<pipeline-id>/jobs" | jq '.[] | select(.status == "failed") | {id, name, stage, failure_reason}'

# Job log
glab api "projects/<project-id>/jobs/<job-id>/trace"

# Child/bridge pipeline failures (when parent shows failed but direct jobs passed)
glab api "projects/<project-id>/pipelines/<pipeline-id>/bridges" | jq '.[] | select(.status == "failed") | {name, failure_reason}'
```

## Issues

```bash
glab issue list
glab issue view <issue-number>

# branches in this repo typically have a prefix of an issue number like:
# 3219-some-name/someauthor
git branch --show-current | sed -e 's:-.*$::' | xargs glab issue view
```

## URL Encoding

Project paths with slashes must be URL-encoded for API calls:
- `group/project` → `group%2Fproject`
- `group/subgroup/project` → `group%2Fsubgroup%2Fproject`

```bash
glab api "projects/group%2Fsubgroup%2Fproject/pipelines"
```
