<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- Thank you for contributing to Safe Synthesizer! -->

# Summary
<!-- Brief description of changes -->

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

- [ ] `mise run format && mise run check` or via prek validation.
- [ ] `mise run test` passes locally
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

<!-- Please add the issue number that should be closed when this PR is merged. -->
- Closes #<issue>
