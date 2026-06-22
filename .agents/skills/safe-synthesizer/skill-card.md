## Description: <br>
Use NeMo Safe Synthesizer through task-specific routing for CLI runs, SDK usage, configuration, troubleshooting, artifacts, and evaluation outputs. <br>

This skill is pending final NVSkills signing and publication validation. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>

## Use Case: <br>
Developers and engineers who need to run, configure, diagnose, or inspect outputs from NeMo Safe Synthesizer without re-reading the full user guide. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Guidance can cause expensive GPU runs or incorrect privacy settings if an agent skips prerequisites or parameter precedence. <br>
Mitigation: The skill routes agents to the source user-guide pages, requires explicit commands or file paths, and separates usage guidance from source-code changes. <br>

Risk: Users can confuse general synthetic data or differential privacy questions with product-specific NeMo Safe Synthesizer workflows. <br>
Mitigation: The eval dataset includes negative routing cases for general React UI and differential privacy explainer prompts. <br>

## Reference(s): <br>
- [Getting Started](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/getting-started/) <br>
- [Running Safe Synthesizer](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/running/) <br>
- [Configuration](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/configuration/) <br>
- [Troubleshooting](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/troubleshooting/) <br>
- [Synthetic Data Quality](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/evaluating-data/) <br>
- [Environment Variables](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/environment/) <br>

## Skill Output: <br>
**Output Type(s):** [Shell commands, Python SDK snippets, Configuration instructions, Diagnostics, Artifact paths] <br>
**Output Format:** [Markdown with inline bash, Python, and YAML snippets] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Answers cite Safe Synthesizer docs and include one next action unless the user asks for a full walkthrough] <br>

## Evaluation Agents Used: <br>
- `claude-code` (pending NVSkills-Eval run) <br>
- `codex` (pending NVSkills-Eval run) <br>

## Evaluation Tasks: <br>
Evaluation dataset contains 6 tasks: 4 positive skill-activation tasks for run, config, diagnose, and artifacts workflows, plus 2 negative activation tasks. Final NVSkills-Eval attempts and pass threshold are pending. <br>

## Evaluation Metrics Used: <br>
Planned benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals expected for this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>

## Evaluation Results: <br>
Pending NVSkills-Eval run. <br>

## Skill Version(s): <br>
c5f3a21d (source: git SHA, committed 2026-06-15) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and has established policies and practices to support AI development across many applications. Developers should work with their internal teams to confirm that this skill meets requirements for their industry and use case, including privacy requirements for sensitive datasets. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
