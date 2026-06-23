## Description: <br>
Use NeMo Safe Synthesizer through task-specific routing: running the CLI or SDK, configuring parameters, troubleshooting runtime failures, inspecting artifacts, and interpreting evaluation outputs. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers using NeMo Safe Synthesizer to generate private, safe synthetic tabular datasets with differential privacy support for compliance and sensitive information protection. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Run Safe Synthesizer](references/run.md) <br>
- [Configure Safe Synthesizer](references/config.md) <br>
- [Diagnose Safe Synthesizer](references/diagnose.md) <br>
- [Inspect Safe Synthesizer Artifacts](references/artifacts.md) <br>
- [Getting Started](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/getting-started/) <br>
- [Safe Synthesizer 101 Tutorial](https://nvidia-nemo.github.io/Safe-Synthesizer/tutorials/safe-synthesizer-101/) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Diagnostic guidance] <br>
**Output Format:** [Markdown with inline bash and Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 6 NVSkills-Eval tasks (4 positive activation, 2 negative activation) with 1 attempt per task and a 50% pass threshold. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 6 | 100% (+0%) | 100% (+0%) |
| Correctness | 6 | 98% (+52%) | 96% (+20%) |
| Discoverability | 6 | 96% (+54%) | 93% (+32%) |
| Effectiveness | 6 | 89% (+45%) | 90% (+26%) |
| Efficiency | 6 | 87% (+37%) | 89% (+29%) |

## Skill Version(s): <br>
v0.1.3 (source: git tag) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
