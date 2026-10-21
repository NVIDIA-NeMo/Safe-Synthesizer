<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- Thank you for contributing to Safe Synthesizer! -->
<!-- External contributors: before submitting (including drafts), link an existing, open issue labeled `triaged` by a maintainer. Newly opened or reopened PRs without one are automatically closed. See CONTRIBUTING.md. -->

# Summary
<!-- Brief description of changes -->

## Related Issue

<!-- Use Fixes #NNN, Closes #NNN, or Resolves #NNN. At least one issue must be open and labeled `triaged` for external contributions. Check the issue for that label before reopening an automatically closed PR. -->
- Closes #<issue>

## Human review

Select exactly one. The default is no human review; uncheck it when choosing another level.

- [x] ⚪ No human review; automated review only
- [ ] 🟡 Partially reviewed or spot-checked by a human
- [ ] 🔵 Complete diff reviewed by a human
- [ ] 🟢 Complete diff reviewed and verified by a human

Verification performed:

## Pre-Review Checklist

<!-- These checks should be completed before a PR is reviewed, -->
<!-- but you can submit a draft early to indicate that the issue is being worked on. -->

Ensure that the following pass:

- [ ] `mise run format` leaves only intended formatting changes.
- [ ] `mise run check ::: test` passes locally.
- [ ] `mise run test:e2e` passes locally
- [ ] `mise run test:ci-container` passes locally (recommended)
- [ ] GPU CI status check passes -- comment `/sync` on this PR to trigger a run (auto-triggers on ready-for-review)

## Pre-Merge Checklist

<!-- These checks need to be completed before a PR is merged, -->
<!-- but as PRs often change significantly during review, -->
<!-- it's OK for them to be incomplete when review is first requested. -->

- [ ] New or updated tests for any fix or new behavior
- [ ] Updated documentation for new features and behaviors, including docstrings for API docs.

## Other Notes
