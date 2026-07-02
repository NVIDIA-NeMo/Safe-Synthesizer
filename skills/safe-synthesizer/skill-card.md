<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Description

Use NeMo Safe Synthesizer through task-specific routing: running the CLI or SDK, configuring parameters, troubleshooting runtime failures, inspecting artifacts, and interpreting evaluation outputs.

This skill is ready for commercial/non-commercial use.

## Owner

NVIDIA

### License/Terms of Use

Apache 2.0

## Use Case

Developers and engineers using NeMo Safe Synthesizer to generate private, safe synthetic tabular datasets with differential privacy support for compliance and sensitive information protection.

### Deployment Geography for Use

Global

## Known Risks and Mitigations

Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills.

Mitigation: Review and scan skill before deployment.

## Reference(s)

- [Run Safe Synthesizer](references/run.md)
- [Configure Safe Synthesizer](references/config.md)
- [Diagnose Safe Synthesizer](references/diagnose.md)
- [Inspect Safe Synthesizer Artifacts](references/artifacts.md)
- [Getting Started](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/getting-started/)
- [Safe Synthesizer 101 Tutorial](https://nvidia-nemo.github.io/Safe-Synthesizer/tutorials/safe-synthesizer-101/)

## Skill Output

**Output Type(s):** Shell commands, configuration instructions, diagnostic guidance

**Output Format:** Markdown with inline bash and Python code blocks

**Output Parameters:** Single markdown response per invocation

**Other Properties Related to Output:** None

## Evaluation Agents Used

- Claude Code (`claude-code`)
- Codex (`codex`)

## Evaluation Tasks

Evaluated against 6 NVSkills-Eval tasks (4 positive activation, 2 negative activation) with 1 attempt per task and a 50% pass threshold.

## Evaluation Metrics Used

Reported benchmark dimensions:

- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it.
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- `security`: Checks for unsafe operations, secret leakage, and unauthorized access.
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow.
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage.
- `accuracy`: Grades final-answer correctness against the reference answer.
- `goal_accuracy`: Checks whether the overall user task completed successfully.
- `behavior_check`: Verifies expected behavior steps, including safety expectations.
- `token_efficiency`: Compares token usage with and without the skill.

## Evaluation Results

| Dimension | Tasks | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 6 | 100% (+0%) | 100% (+0%) |
| Correctness | 6 | 98% (+52%) | 96% (+20%) |
| Discoverability | 6 | 96% (+54%) | 93% (+32%) |
| Effectiveness | 6 | 89% (+45%) | 90% (+26%) |
| Efficiency | 6 | 87% (+37%) | 89% (+29%) |

Security is retained in the table even though uplift is 0% because both baseline
and skill-assisted runs already scored 100%.

## Skill Version(s)

v0.1.3 (source: git tag)

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

(For Release on NVIDIA Platforms Only)

Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
