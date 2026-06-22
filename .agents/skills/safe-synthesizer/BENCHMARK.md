<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluation Plan

Pre-publication evaluation plan for the `safe-synthesizer` skill.

The source eval dataset is checked in at `evals/evals.json`. Final NVSkills-Eval execution, benchmark results, and signing are still pending.

## Evaluation Summary

- Skill: `safe-synthesizer`
- Evaluation date: pending
- NVSkills-Eval profile: pending
- Environment: pending
- Dataset: 6 evaluation tasks
- Attempts per task: pending
- Pass threshold: pending
- Overall verdict: pending

## Agents Used

- `claude-code` (pending)
- `codex` (pending)

## Metrics Used

Planned benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Expected evaluation signals:

- `security`: checks for unsafe operations, secret leakage, and unauthorized access.
- `skill_execution`: verifies that the agent loaded the expected skill and workflow.
- `skill_efficiency`: checks routing quality, decoy avoidance, and redundant tool usage.
- `accuracy`: grades final-answer correctness against the reference answer.
- `goal_accuracy`: checks whether the overall user task completed successfully.
- `behavior_check`: verifies expected behavior steps, including safety expectations.
- `token_efficiency`: compares token usage with and without the skill.

## Test Tasks

The dataset covers the four routed workflows in `SKILL.md`:

| ID | Route | Purpose |
|---|---|---|
| `safe-synthesizer-run-dp-cli` | `run.md` | Run the CLI with differential privacy overrides. |
| `safe-synthesizer-config-num-records` | `config.md` | Set `generation.num_records` from the CLI and SDK. |
| `safe-synthesizer-diagnose-generation-oom` | `diagnose.md` | Triage generation-phase out-of-memory failures. |
| `safe-synthesizer-artifact-locations` | `artifacts.md` | Find generated data, reports, metrics, logs, and adapters. |
| `safe-synthesizer-negative-react-ui` | negative | Avoid activation for general React UI work. |
| `safe-synthesizer-negative-general-dp` | negative | Avoid activation for a general differential privacy explainer. |

## Results

Pending NVSkills-Eval run.

## Publication Readiness

Source-side artifacts added in this branch:

- `evals/evals.json`
- `skill-card.md`
- `BENCHMARK.md`

Remaining external publication step:

- Run NVSkills validation and signing so `skill.oms.sig` lands beside `SKILL.md`.
