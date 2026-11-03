<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Description: <br>
Use NeMo Safe Synthesizer through task-specific routing: running the CLI or SDK, configuring parameters, troubleshooting runtime failures, inspecting artifacts, and interpreting evaluation outputs. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data engineers use this skill to have an agent run, configure, troubleshoot, and inspect outputs of NeMo Safe Synthesizer, which generates private synthetic versions of sensitive tabular datasets with optional differential privacy and PII replacement. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Run Safe Synthesizer](references/run.md) <br>
- [Configure Safe Synthesizer](references/config.md) <br>
- [Diagnose Safe Synthesizer](references/diagnose.md) <br>
- [Inspect Safe Synthesizer Artifacts](references/artifacts.md) <br>
- [Getting Started](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/getting-started/) <br>
- [Running Safe Synthesizer](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/running/) <br>
- [Configuration Reference](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/configuration/) <br>
- [Troubleshooting](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/troubleshooting/) <br>
- [Environment Variables](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/environment/) <br>
- [Synthetic Data Quality](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/evaluating-data/) <br>
- [Configuration Management Guide](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/docs/developer-guide/configuration_management.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code, Analysis] <br>
**Output Format:** [Markdown with inline bash and Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
Evaluated against 6 evaluation tasks (4 positive, 2 negative) with 1 attempt per task, each run in an isolated sandbox pod (`k8s-sandbox`), using evaluator version 1.6.6. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal (100%). <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal (100%). <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` (tool-call productivity) plus 50% `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 93.3% (+33.3 points) | 83.3% → 93.3% (+10.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 38.5% → 85.7% (+47.2 points) | 68.1% → 79.6% (+11.5 points) |
| Efficiency | 75.9% — baseline ran, but no comparable score was available; uplift unavailable | 71.6% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.7.0.post1+cu130 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
